"""Thin wrapper over the Bedrock Converse API with a tool-use loop."""

import json
import logging
from typing import Callable

import boto3

from app import config

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
        tools: dict[str, tuple[dict, ToolHandler]],
        max_turns: int = config.MAX_AGENT_TURNS,
        max_tokens: int = 1024,
    ) -> tuple[str, list]:
        """Run a Converse tool-use loop until the model stops calling tools.

        tools maps name -> (toolSpec, handler). Returns (final_text, trace) where
        trace lists each tool call with its input and output.
        """
        tool_config = {"tools": [{"toolSpec": spec} for spec, _ in tools.values()]}
        trace = []
        for _ in range(max_turns):
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
                return _text_of(message), trace

            results = []
            for block in message["content"]:
                if "toolUse" not in block:
                    continue
                call = block["toolUse"]
                output, status = self._call(tools, call["name"], call["input"])
                trace.append({"tool": call["name"], "input": call["input"], "output": output})
                results.append({
                    "toolResult": {
                        "toolUseId": call["toolUseId"],
                        "content": [{"json": {"result": output}}],
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
        except Exception as e:  # surface tool failures to the model, don't crash the loop
            logger.warning("tool %s failed: %s: %s", name, type(e).__name__, e)
            return f"{type(e).__name__}: {e}", "error"


def _text_of(message: dict) -> str:
    return "".join(b["text"] for b in message["content"] if "text" in b).strip()
