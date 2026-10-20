import json

from fastapi.testclient import TestClient

from app import events
from app.agents.deal_agent import DealAgent
from app.agents.main_agent import MainAgent
from app.llm import LLM
from app.main import app, get_main_agent
from tests.fakes import FakeBedrock, fake_store, text_reply, tool_call


def _agent():
    """Main agent whose find_deals runs a real deal sub-agent, all on one scripted client."""
    find_deals = tool_call("find_deals", {"product": "purple dress"}, "a")
    find_deals["output"]["message"]["content"].insert(0, {"text": "Let me look for dresses."})
    client = FakeBedrock(
        find_deals,                                            # main agent
        tool_call("web_search", {"query": "purple dress sale"}, "b"),  # deal sub-agent
        text_reply('{"deals": [], "summary": "none"}'),        # deal sub-agent
        text_reply("I couldn't find any."),                    # main agent
    )
    llm = LLM(client, "m")
    return MainAgent(llm, fake_store(), DealAgent(llm, search=lambda q: [{"title": "t", "url": "u", "content": "c"}]),
                     contacts=None, search=lambda q: [])


def test_sub_agent_steps_nest_under_the_step_that_started_them():
    seen = []
    with events.listen(seen.append):
        _agent().chat("u1", "purple dress for a wedding")

    assert [(e["type"], e.get("tool")) for e in seen] == [
        ("note", None), ("step", "find_deals"), ("step", "web_search"),
        ("step_done", "web_search"), ("step_done", "find_deals"),
    ]
    note, find, search, search_done, find_done = seen
    assert note["text"] == "Let me look for dresses."
    assert find["parent"] is None and search["parent"] == find["id"]
    assert search_done["id"] == search["id"] and search_done["status"] == "success"
    assert find_done["output"]["deals"] == []


def test_no_listener_means_no_events_and_no_errors():
    assert _agent().chat("u1", "purple dress")["reply"] == "I couldn't find any."


def test_chat_stream_sends_progress_then_the_reply():
    app.dependency_overrides[get_main_agent] = _agent
    try:
        with TestClient(app).stream("POST", "/chat/stream", json={"user_id": "u1", "message": "dress"}) as resp:
            assert resp.headers["content-type"].startswith("text/event-stream")
            sent = [json.loads(line[len("data: "):]) for line in resp.iter_lines() if line.startswith("data: ")]
    finally:
        app.dependency_overrides.clear()

    assert [e["type"] for e in sent] == ["note", "step", "step", "step_done", "step_done", "done"]
    assert sent[-1]["reply"] == "I couldn't find any."
    assert [t["tool"] for t in sent[-1]["tool_calls"]] == ["find_deals"]
