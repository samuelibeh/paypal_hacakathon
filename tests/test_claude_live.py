"""Live smoke test: call a Haiku model directly through the Anthropic API (not Bedrock).

Needs ANTHROPIC_API_KEY in .env or the environment; skipped otherwise, so plain `pytest`
stays offline. Run it on its own with:

    pytest tests/test_claude_live.py -v -s

Override the model with CLAUDE_TEST_MODEL.
"""

import os

import pytest
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("CLAUDE_TEST_MODEL", "claude-haiku-5-5")

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY is not set"
)


def make_client() -> Anthropic:
    headers = {}
    if os.getenv("ANTHROPIC_WORKSPACE_ID"):
        headers["anthropic-workspace-id"] = os.environ["ANTHROPIC_WORKSPACE_ID"]
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=headers or None)


@pytest.fixture(scope="module")
def client():
    return make_client()


def test_haiku_replies(client):
    resp = client.messages.create(
        model=MODEL,
        max_tokens=50,
        messages=[{"role": "user", "content": "Reply with exactly one word: pong"}],
    )
    text = resp.content[0].text
    print(f"\n[{resp.model}] {text!r}  usage={resp.usage.input_tokens}in/{resp.usage.output_tokens}out")

    assert resp.stop_reason == "end_turn"
    assert "pong" in text.lower()


def test_haiku_can_call_a_tool(client):
    """The shopping agents rely on tool use, so check Haiku emits a tool_use block."""
    tool = {
        "name": "web_search",
        "description": "Search the web. Returns titles, URLs and snippets.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
    resp = client.messages.create(
        model=MODEL,
        max_tokens=200,
        tools=[tool],
        tool_choice={"type": "tool", "name": "web_search"},
        messages=[{"role": "user", "content": "Find cheap fridges in Lagos"}],
    )
    call = next(b for b in resp.content if b.type == "tool_use")
    print(f"\n[tool] {call.name}({call.input})")

    assert call.name == "web_search"
    assert isinstance(call.input.get("query"), str) and call.input["query"]


def main():
    """Run the checks without pytest: python tests/test_claude_live.py"""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set (put it in .env)")
    client = make_client()
    for test in (test_haiku_replies, test_haiku_can_call_a_tool):
        test(client)
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
