import json

from fastapi.testclient import TestClient

from app import main

from app import events
from app.agents.deal_agent import DealAgent
from app.agents.main_agent import MainAgent
from app.llm import LLM
from app.main import app, get_main_agent
from tests.fakes import FakeBedrock, fake_store, signed_in, text_reply, tool_call


def _agent():
    """Main agent whose find_deals runs a real deal sub-agent, all on one scripted client."""
    find_deals = tool_call("find_deals", {"product": "purple dress"}, "a")
    find_deals["output"]["message"]["content"].insert(0, {"text": "Let me look for dresses."})
    client = FakeBedrock(
        find_deals,                                            # main agent
        tool_call("web_search", {"query": "purple dress sale"}, "b"),  # deal sub-agent
        text_reply('{"deals": [{"store": "Shop", "title": "Purple dress", "price": "$40", "amount": 40, '
                   '"currency": "USD", "url": "https://shop.com/d"}], "summary": "one"}'),  # deal sub-agent
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
    assert [d["title"] for d in find_done["output"]["deals"]] == ["Purple dress"] and find_done["output"]["deals"][0]["n"] == 1


def test_no_listener_means_no_events_and_no_errors():
    assert _agent().chat("u1", "purple dress")["reply"] == "I couldn't find any."


def test_chat_stream_sends_progress_then_the_reply(monkeypatch):
    store = fake_store()  # not the database in your .env
    monkeypatch.setattr(main, "get_store", lambda: store)
    monkeypatch.setattr(main, "_name_chat", lambda *args: None)  # no real Claude call from the test
    app.dependency_overrides[get_main_agent] = _agent
    try:
        with signed_in(app, "u1").stream("POST", "/chat/stream", json={"user_id": "u1", "message": "dress"}) as resp:
            assert resp.headers["content-type"].startswith("text/event-stream")
            sent = [json.loads(line[len("data: "):]) for line in resp.iter_lines() if line.startswith("data: ")]
    finally:
        app.dependency_overrides.clear()

    assert [e["type"] for e in sent] == ["note", "step", "step", "step_done", "step_done", "done"]
    assert sent[-1]["reply"] == "I couldn't find any."
    assert [t["tool"] for t in sent[-1]["tool_calls"]] == ["find_deals"]

    assert store.current_chat("u1")[0]["message"] == "dress"  # the turn is saved


def test_stop_ends_the_turn_before_the_next_step_and_keeps_nothing():
    import threading

    from app import events
    from app.llm import LLM

    stop = threading.Event()
    client = FakeBedrock(tool_call("web_search", {"query": "fridge"}), text_reply("never reached"))

    def search(args):
        stop.set()  # the user presses Stop while the search runs
        return []

    spec = {"name": "web_search", "inputSchema": {"json": {"type": "object"}}}
    try:
        with events.stoppable(stop):
            LLM(client, "m").run_tools("sys", [], {"web_search": (spec, search)})
    except events.Stopped:
        pass
    else:
        raise AssertionError("expected Stopped")
    assert len(client.requests) == 1  # no model call after Stop


def test_stop_route_flags_the_reply_in_progress():
    import threading

    stop = main._replies["u1"] = threading.Event()
    try:
        assert signed_in(app, "u1").post("/chat/u1/stop").json() == {"stopping": True}
        assert stop.is_set()
    finally:
        main._replies.pop("u1", None)
    assert signed_in(app, "u1").post("/chat/u1/stop").json() == {"stopping": False}
    assert signed_in(app, "u2").post("/chat/u1/stop").status_code == 403


def test_status_says_whether_a_reply_is_still_coming_after_the_stream_dropped():
    import threading

    main._replies["u1"] = threading.Event()
    try:
        assert signed_in(app, "u1").get("/chat/u1/status").json() == {"working": True}
    finally:
        main._replies.pop("u1", None)
    assert signed_in(app, "u1").get("/chat/u1/status").json() == {"working": False}
    assert signed_in(app, "u2").get("/chat/u1/status").status_code == 403
