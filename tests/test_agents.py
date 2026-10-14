from app.agents.contact_agent import ContactAgent
from app.agents.deal_agent import DealAgent
from app.agents.main_agent import MainAgent
from app.llm import LLM
from tests.fakes import FakeBedrock, FakeEmbedder, fake_store, text_reply, tool_call


def test_tool_loop_feeds_results_back_and_records_trace():
    client = FakeBedrock(tool_call("echo", {"x": 1}), text_reply("done"))
    spec = {"name": "echo", "description": "", "inputSchema": {"json": {"type": "object"}}}
    reply, trace = LLM(client, "m").run_tools("sys", [], {"echo": (spec, lambda a: a["x"] + 1)})

    assert reply == "done"
    assert trace == [{"tool": "echo", "input": {"x": 1}, "output": 2}]
    result = client.requests[1]["messages"][-1]["content"][0]["toolResult"]
    assert result["content"] == [{"json": {"result": 2}}] and result["status"] == "success"


def test_tool_errors_are_reported_to_the_model_not_raised():
    client = FakeBedrock(tool_call("boom", {}), text_reply("sorry"))
    spec = {"name": "boom", "description": "", "inputSchema": {"json": {"type": "object"}}}
    reply, trace = LLM(client, "m").run_tools("sys", [], {"boom": (spec, lambda a: 1 / 0)})

    assert reply == "sorry"
    assert client.requests[1]["messages"][-1]["content"][0]["toolResult"]["status"] == "error"


def test_deal_agent_parses_json_and_lists_searches():
    final = '{"deals": [{"store": "Amazon", "title": "Fridge", "price": "$499", "url": "u", "note": ""}], "summary": "s"}'
    client = FakeBedrock(tool_call("web_search", {"query": "fridge sale"}), text_reply(final))
    searched = []
    agent = DealAgent(LLM(client, "m"), search=lambda q: searched.append(q) or [{"title": "t", "url": "u", "content": "c"}])

    result = agent.find_deals("fridge", location="Lagos")

    assert result["deals"][0]["price"] == "$499"
    assert result["searches"] == ["fridge sale"] == searched
    assert "Lagos" in client.requests[0]["messages"][0]["content"][0]["text"]


def test_contact_import_and_similarity_search():
    store = fake_store()
    client = FakeBedrock(
        text_reply("Morris runs an appliance shop selling fridges and freezers."),
        text_reply("Bob is a wedding photographer."),
    )
    agent = ContactAgent(LLM(client, "m"), store, FakeEmbedder(), search=lambda q, max_results: [])
    agent.import_contacts("u1", [{"name": "Morris", "notes": "appliance shop"}, {"name": "Bob"}])

    hits = agent.find_contacts("u1", "fridges freezers appliance")

    assert [h["name"] for h in hits] == ["Morris"]
    assert agent.find_contacts("u2", "fridges") == []  # contacts are per-user


def test_main_agent_delegates_to_both_subagents():
    store = fake_store()
    store.remember("u1", "city", "Lagos")
    client = FakeBedrock(
        tool_call("find_deals", {"product": "fridge"}, "a"),
        tool_call("find_contacts", {"need": "fridge"}, "b"),
        text_reply("Best deal: Amazon $499. Morris sells fridges."),
    )

    class Deals:
        def find_deals(self, product, location):
            return {"deals": [{"store": "Amazon", "price": "$499", "product": product, "loc": location}]}

    class Contacts:
        def find_contacts(self, user_id, need):
            return [{"name": "Morris", "user": user_id}]

    agent = MainAgent(LLM(client, "m"), store, Deals(), Contacts(), search=lambda q: [])
    out = agent.chat("u1", "Where can I buy a fridge?", location="Ikeja, Lagos")

    assert [t["tool"] for t in out["tool_calls"]] == ["find_deals", "find_contacts"]
    assert out["tool_calls"][0]["output"]["deals"][0]["loc"] == "Ikeja, Lagos"
    assert out["tool_calls"][1]["output"] == [{"name": "Morris", "user": "u1"}]
    assert out["reply"].startswith("Best deal")


def test_remember_persists_user_facts():
    store = fake_store()
    store.remember("u1", "name", "Sam")
    store.remember("u1", "budget", "$600")
    assert store.get_profile("u1") == {"name": "Sam", "budget": "$600"}
    assert store.get_profile("nobody") == {}
