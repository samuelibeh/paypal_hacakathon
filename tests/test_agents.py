from app.agents.deal_agent import DealAgent
from app.agents.main_agent import MainAgent
from app.llm import LLM
from app.merchants import Merchant, Merchants
from tests.fakes import FakeBedrock, fake_store, text_reply, tool_call


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
    assert client.requests[0]["inferenceConfig"]["maxTokens"] == 4096  # five deals don't fit in 1024


def test_deal_agent_stops_searching_at_the_limit():
    final = '{"deals": [], "summary": "s"}'
    client = FakeBedrock(*[tool_call("web_search", {"query": f"q{n}"}) for n in range(9)], text_reply(final))
    searched = []
    agent = DealAgent(LLM(client, "m"), search=lambda q: searched.append(q) or [])

    result = agent.find_deals("fridge")

    assert len(searched) == 8  # the ninth search isn't run
    assert result["summary"] == "s"
    assert "8 searches" in client.requests[0]["system"][0]["text"]


def test_main_agent_delegates_to_both_subagents():
    store = fake_store()
    store.remember("u1", "city", "Lagos")
    client = FakeBedrock(
        tool_call("find_deals", {"product": "fridge"}, "a"),
        tool_call("find_contacts", {"need": "fridge"}, "b"),
        text_reply("Best deal: Amazon $499. Morris sells fridges."),
    )

    class Deals:
        def find_deals(self, product, location, budget=None, must_have=None, brief=None):
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
        text_reply("1. Slim $499. 2. Wide $650. Which fits the space?"),
    )

    class Deals:
        def find_deals(self, product, location, budget=None, must_have=None, brief=None):
            return {"deals": [
                {"store": "Shop", "title": "Slim fridge", "price": "$499", "amount": 499, "currency": "USD", "url": "https://shop.com/s"},
                {"store": "Shop", "title": "Wide fridge", "price": "$650", "amount": 650, "currency": "USD", "url": "https://shop.com/w"},
            ]}

    shop = Merchant("shop", "Shop", ("shop.com",), "payee@shop.com", "help@shop.com", ("shop.com",))
    agent = MainAgent(LLM(client, "m"), store, Deals(), contacts=None, search=lambda q: [], merchants=Merchants([shop]))
    out = agent.chat("u1", "fridge please")
    slim, wide = out["tool_calls"][0]["output"]["deals"]
    assert slim["deal_id"] and slim["payable"] and slim["amount"] == "499.00"

    store.add_to_cart("u1", slim["deal_id"])  # user taps Add to cart in the UI
    client.responses += [tool_call("get_cart", {}, "c"), tool_call("add_to_cart", {"deal_id": wide["deal_id"]}, "b"),
                         text_reply("Added.")]
    out = agent.chat("u1", "Here's the spot. I'll take the wide one too", history=out["history"],
                     images=[("jpeg", b"\xff\xd8photo")])

    sent = client.requests[2]["messages"][-1]["content"]
    assert sent[0] == {"image": {"format": "jpeg", "source": {"bytes": b"\xff\xd8photo"}}}
    assert "Selected in the app" not in sent[1]["text"]  # the cart isn't sent with every message
    assert out["tool_calls"][0]["output"] == [f"[{slim['deal_id']}] Slim fridge from Shop, $499"]  # looked up on request
    assert [i["title"] for i in out["tool_calls"][1]["output"]["items"]] == ["Slim fridge", "Wide fridge"]


def test_remember_persists_user_facts():
    store = fake_store()
    store.remember("u1", "name", "Sam")
    store.remember("u1", "budget", "$600")
    assert store.get_profile("u1") == {"name": "Sam", "budget": "$600"}
    assert store.get_profile("nobody") == {}


class ScriptedDeals:
    """A deal agent that returns one scripted round per call and records what it was given."""

    def __init__(self, *rounds):
        self.rounds, self.calls = list(rounds), []

    def find_deals(self, product, location, budget=None, must_have=None, brief=None):
        self.calls.append({"product": product, "budget": budget, "must_have": must_have, "brief": brief})
        return self.rounds.pop(0)


def _macbook(store, price, amount, url, **extra):
    return {"store": store, "title": "MacBook Pro M1 Pro", "price": price, "amount": amount, "currency": "NGN",
            "url": url, **extra}


def _hunt_agent(deals, *calls):
    from app.agents.main_agent import MainAgent as Agent
    client = FakeBedrock(*calls, text_reply("Here's what I found."))
    store = fake_store()
    return Agent(LLM(client, "m"), store, deals, contacts=None, search=lambda q: []), store


BUDGET = {"product": "MacBook Pro M1 Pro", "budget": {"min": 700000, "max": 800000, "currency": "NGN"},
          "must_have": ["M1 Pro chip"]}


def test_the_main_agent_retries_once_broadened_with_a_brief_and_the_numbers_carry_on():
    deals = ScriptedDeals(
        {"deals": [_macbook("Lagos Tech", "₦1,040,000", 1040000, "https://ig.com/a")], "searches": ["MacBook M1 Pro discount Lagos"],
         "pages": [{"title": "Lagos Tech listing", "url": "https://ig.com/a"}]},
        {"deals": [_macbook("Dex Gadgets", "₦780,000", 780000, "https://dex.ng/m1", missing="M1, not M1 Pro"),
                   _macbook("Lagos Tech", "₦1,040,000", 1040000, "https://ig.com/a")],  # shown already: dropped
         "searches": ["MacBook M1 sale Lagos"], "pages": []},
        {"deals": [], "searches": ["q"], "pages": []},
    )
    broadened = {"product": "MacBook M1", "budget": BUDGET["budget"], "same_hunt": True}
    agent, store = _hunt_agent(deals, tool_call("find_deals", BUDGET, "a"), tool_call("find_deals", broadened, "b"),
                               tool_call("find_deals", broadened, "c"))
    out = agent.chat("u1", "MacBook M1 Pro around 700k-800k", location="Ikeja, Lagos")

    assert len(deals.calls) == 2  # the third call in one message isn't run
    first, second = deals.calls
    assert first["brief"] is None and first["budget"]["max"] == 800000 and first["must_have"] == ["M1 Pro chip"]
    assert second["product"] == "MacBook M1" and '"MacBook M1 Pro discount Lagos"' in second["brief"]
    assert "https://ig.com/a" in second["brief"]
    one, two, three = (c["output"] for c in out["tool_calls"])
    assert [d["n"] for d in one["deals"]] == [1] and [d["n"] for d in two["deals"]] == [2]  # numbered on, as on the cards
    assert two["deals"][0]["missing"] == "M1, not M1 Pro"
    assert (one["searches_left"], two["searches_left"]) == (1, 0) and "Already searched 2 times" in three["summary"]

    agent.llm.client.responses += [tool_call("find_deals", {**BUDGET, "same_hunt": True}, "d"), text_reply("More.")]
    out = agent.chat("u1", "search more")
    assert len(deals.calls) == 3 and out["tool_calls"][0]["output"]["deals"] == []  # a new message, a new allowance
    assert "Today: " in agent.llm.client.requests[-1]["messages"][-3]["content"][0]["text"]  # the date goes with it


def test_search_more_continues_the_hunt_and_a_new_item_starts_clean():
    deals = ScriptedDeals(
        {"deals": [_macbook("Dex Gadgets", "₦780,000", 780000, "https://dex.ng/a")], "searches": ["q1"], "pages": []},
        {"deals": [_macbook("Slot", "₦760,000", 760000, "https://slot.ng/b")], "searches": ["q2"], "pages": []},
        {"deals": [{"store": "Jumia", "title": "Fridge", "price": "₦300,000", "amount": 300000, "currency": "NGN",
                    "url": "https://jumia.com/f"}], "searches": ["fridge"], "pages": []},
    )
    agent, store = _hunt_agent(deals, tool_call("find_deals", BUDGET, "a"))
    agent.chat("u1", "MacBook M1 Pro 700k-800k")
    agent.llm.client.responses += [tool_call("find_deals", {**BUDGET, "same_hunt": True}, "b"), text_reply("More.")]
    agent.chat("u1", "search more")
    agent.llm.client.responses += [tool_call("find_deals", {"product": "fridge"}, "c"), text_reply("Fridges.")]
    agent.chat("u1", "now a fridge")

    assert deals.calls[1]["brief"] and '"q1"' in deals.calls[1]["brief"]
    assert deals.calls[2]["brief"] is None  # a new item: no brief from the MacBook hunt
    assert store.get_hunt("u1")["product"] == "fridge"


def test_a_reply_said_alongside_a_tool_call_isnt_lost():
    said_first = tool_call("remember", {"key": "city", "value": "Lagos"}, "a")
    said_first["output"]["message"]["content"].insert(0, {"text": "Here are the deals."})
    client = FakeBedrock(said_first, text_reply(""))
    reply, _ = LLM(client, "m").run_tools("sys", [], {"remember": ({"name": "remember"}, lambda a: "ok")})
    assert reply == "Here are the deals."


def test_the_main_agent_sees_a_compact_numbered_list_and_the_app_keeps_the_full_deals():
    deal = _macbook("Dex Gadgets", "₦780,000", 780000, "https://dex.ng/m1pro",
                    details="UK used, 16GB RAM, 512GB SSD, 14-inch, battery cycle count 120, comes with charger",
                    note="WhatsApp 0800 000 0000, Computer Village", discount="was ₦850,000")
    deals = ScriptedDeals({"deals": [deal], "searches": ["q"], "pages": []})
    agent, _ = _hunt_agent(deals, tool_call("find_deals", BUDGET, "a"))
    out = agent.chat("u1", "MacBook M1 Pro 700k-800k")

    full = out["tool_calls"][0]["output"]["deals"][0]
    assert full["note"].startswith("WhatsApp") and full["url"] == "https://dex.ng/m1pro"  # the card keeps it all
    shown = agent.llm.client.requests[1]["messages"][-1]["content"][0]["toolResult"]["content"][0]["json"]["result"]
    [compact] = shown["deals"]
    assert compact["n"] == 1 and compact["price"] == "₦780,000" and compact["discount"] == "was ₦850,000"
    assert shown["searches_left"] == 1
    assert "note" not in compact and "url" not in compact and len(compact["detail"]) <= 120


def test_the_deal_agent_is_told_the_date_and_starts_with_discounts():
    from datetime import date

    from app.agents.deal_agent import SYSTEM, _request

    assert "Today: 30 October 2026 (look for 2026 prices)" in _request("cake", "Ikeja", None, None, None, date(2026, 10, 30))
    assert "Always start with discounts" in SYSTEM and '"discount": str or null' in SYSTEM
