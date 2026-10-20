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
    agent.import_contacts("u1", [{"name": "Morris", "notes": "appliance shop"}, {"name": "Bob"}], enrich_with_web=False)

    hits = agent.find_contacts("u1", "fridges freezers appliance")

    assert [h["name"] for h in hits] == ["Morris"]
    assert agent.find_contacts("u2", "fridges") == []  # contacts are per-user


def test_contact_enrichment_model_writes_query_we_search_model_writes_profile():
    client = FakeBedrock(
        text_reply('"Morris Adeyemi" Ikeja appliance shop'),            # Morris: search query
        text_reply("Morris runs Adeyemi Appliances in Ikeja, selling fridges."),
        text_reply("NONE"),                                             # Bob: not identifiable
        text_reply("Little is known about Bob."),
    )
    searched = []

    def search(query, max_results):
        searched.append(query)
        return [{"title": "Adeyemi Appliances", "content": "Fridges and ACs in Ikeja"}]

    agent = ContactAgent(LLM(client, "m"), fake_store(), FakeEmbedder(), search=search)
    imported = agent.import_contacts("u1", [
        {"name": "Morris Adeyemi", "notes": "appliance shop in Ikeja"},
        {"name": "Bob"},
    ])

    assert searched == ['"Morris Adeyemi" Ikeja appliance shop']  # one search, and none for Bob
    assert [c["search_query"] for c in imported] == ['"Morris Adeyemi" Ikeja appliance shop', None]
    profile_prompt = client.requests[1]["messages"][0]["content"][0]["text"]
    assert "Adeyemi Appliances: Fridges and ACs in Ikeja" in profile_prompt
    assert "Search results" not in client.requests[3]["messages"][0]["content"][0]["text"]


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


def test_main_agent_sees_selection_and_photo_and_can_add_to_cart():
    store = fake_store()
    client = FakeBedrock(
        tool_call("find_deals", {"product": "fridge"}, "a"),
        text_reply("1. Amazon $499. 2. Walmart $650. Which fits the space?"),
    )

    class Deals:
        def find_deals(self, product, location):
            return {"deals": [
                {"store": "Amazon", "title": "Slim fridge", "price": "$499", "amount": 499, "currency": "USD", "url": "a"},
                {"store": "Walmart", "title": "Wide fridge", "price": "$650", "amount": 650, "currency": "USD", "url": "w"},
            ]}

    agent = MainAgent(LLM(client, "m"), store, Deals(), contacts=None, search=lambda q: [])
    out = agent.chat("u1", "fridge please")
    slim, wide = out["tool_calls"][0]["output"]["deals"]
    assert slim["deal_id"] and slim["payable"] and slim["amount"] == "499.00"

    store.add_to_cart("u1", slim["deal_id"])  # user taps Select in the UI
    client.responses += [tool_call("add_to_cart", {"deal_id": wide["deal_id"]}, "b"), text_reply("Added.")]
    out = agent.chat("u1", "Here's the spot. I'll take the wide one too", history=out["history"],
                     images=[("jpeg", b"\xff\xd8photo")])

    sent = client.requests[2]["messages"][-1]["content"]
    assert sent[0] == {"image": {"format": "jpeg", "source": {"bytes": b"\xff\xd8photo"}}}
    assert f"Selected in the app: [{slim['deal_id']}] Slim fridge" in sent[1]["text"]
    assert [i["title"] for i in out["tool_calls"][0]["output"]["items"]] == ["Slim fridge", "Wide fridge"]


def test_remember_persists_user_facts():
    store = fake_store()
    store.remember("u1", "name", "Sam")
    store.remember("u1", "budget", "$600")
    assert store.get_profile("u1") == {"name": "Sam", "budget": "$600"}
    assert store.get_profile("nobody") == {}
