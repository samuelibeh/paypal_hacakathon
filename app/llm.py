"""Thin wrapper over the Bedrock Converse API with a tool-use loop."""

import json
import logging
from typing import Callable

import boto3

from app import config, events

logger = logging.getLogger(__name__)

ToolHandler = Callable[[dict], object]


class LLM:
    def __init__(self, client=None, model_id: str = config.BEDROCK_MODEL_ID):
        self.client = client or boto3.client("bedrock-runtime", region_name=config.AWS_REGION)
        self.model_id = model_id

    def complete(self, system: str, prompt: str, max_tokens: int = 512) -> str:
        """Single-turn text completion."""
        resp = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens},
        )
        return _text_of(resp["output"]["message"])

    def run_tools(
        self,
        system: str,
        messages: list,
        tools: dict[str, tuple],
        max_turns: int = config.MAX_AGENT_TURNS,
        max_tokens: int = 1024,
    ) -> tuple[str, list]:
        """Run a Converse tool-use loop until the model stops calling tools.

        tools maps name -> (toolSpec, handler) or (toolSpec, handler, view). view, if given,
        turns the handler's output into what the model is shown, e.g. a compact list; the
        trace (and so the app) keeps the full output. Returns (final_text, trace).
        """
        tool_config = {"tools": [{"toolSpec": entry[0]} for entry in tools.values()]}
        trace = []
        _mend(messages)  # a chat saved with an empty message would fail every turn after it
        start = len(messages)  # what came before is earlier turns
        for _ in range(max_turns):
            events.check_stopped()  # the user pressed Stop: start nothing new
            resp = self.client.converse(
                modelId=self.model_id,
                system=[{"text": system}],
                messages=messages,
                toolConfig=tool_config,
                inferenceConfig={"maxTokens": max_tokens},
            )
            message = resp["output"]["message"]
            messages.append(message)
            if resp["stopReason"] != "tool_use":
                # The model sometimes says its piece next to a tool call and then ends with nothing
                # new; the reply is then the last thing it did say, not an empty message. The
                # history keeps that reply too: Bedrock rejects a message with no content.
                reply = _text_of(message) or _last_text(messages[start:-1])
                if not _text_of(message):
                    _mend([message], reply)
                return reply, trace

            results = []
            for block in message["content"]:
                if block.get("text", "").strip():  # what the model says before using tools
                    events.emit("note", text=block["text"].strip())
                if "toolUse" not in block:
                    continue
                events.check_stopped()
                call = block["toolUse"]
                # gpt-oss sometimes leaks its channel marker into the name ("search_store<|channel|>
                # commentary"); fixed in the history too, or Bedrock rejects the next turn.
                call["name"] = call["name"].split("<|")[0]
                with events.step(call["name"], call["input"]) as outcome:
                    output, status = self._call(tools, call["name"], call["input"])
                    outcome.update(status=status, output=output)
                trace.append({"tool": call["name"], "input": call["input"], "output": output})
                view = tools.get(call["name"], (None, None, None))[2:3]
                shown = view[0](output) if view and view[0] and status == "success" else output
                results.append({
                    "toolResult": {
                        "toolUseId": call["toolUseId"],
                        "content": [{"json": {"result": shown}}],
                        "status": status,
                    }
                })
            messages.append({"role": "user", "content": results})
        return "I couldn't finish that request within the step limit.", trace

    @staticmethod
    def _call(tools, name, args):
        if name not in tools:
            return f"Unknown tool {name}", "error"
        try:
            output = tools[name][1](args)
            json.dumps(output)  # Converse requires JSON-serialisable results
            return output, "success"
        except events.Stopped:  # a sub-agent inside this tool saw Stop: not a tool error
            raise
        except Exception as e:  # surface tool failures to the model, don't crash the loop
            logger.warning("tool %s failed: %s: %s", name, type(e).__name__, e)
            return f"{type(e).__name__}: {e}", "error"


def _mend(messages: list, filler: str = "…") -> None:
    """Drop blank text blocks, and give a message left with no content the filler: Bedrock
    rejects empty content, and dropping the message would break the user/assistant turns."""
    for m in messages:
        m["content"] = [b for b in m.get("content") or [] if "text" not in b or b["text"].strip()]
        if not m["content"]:
            m["content"] = [{"text": filler or "…"}]


def _last_text(messages: list) -> str:
    said = [_text_of(m) for m in messages if m.get("role") == "assistant"]
    return next((t for t in reversed(said) if t), "")


# gpt-oss writes narrow and no-break spaces ("$2,500 – 2011 Nissan"); plain spaces read the same
# everywhere the text goes (cards, descriptions, the JSON it's parsed from).
_SPACES = str.maketrans({" ": " ", " ": " ", " ": " ", " ": " "})


def _text_of(message: dict) -> str:
    return "".join(b["text"] for b in message["content"] if "text" in b).translate(_SPACES).strip()
