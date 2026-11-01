"""Run the main agent and the deal agent for real and show the whole hunt: each step as it
happens, every Tavily search with what it returned, each store the deal agent looked inside
and what the store agent answered, what the deal agent is asked each call
(including the brief of earlier rounds), the deals it returns with their facts (discount, what
each lacks, the price's year), the short list the main agent judges and writes its reply from,
whether it retried with the item or the area broadened, and the reply.

Each message after the first continues the same chat, so "search more" carries the hunt.
Uses Claude on Bedrock and Tavily (real calls; up to 2 deal-agent calls per message, each up
to 8 searches) and a
throwaway in-memory database, so nothing is saved to MongoDB.

    python -m scripts.try_deal_hunt "Where can I get a macbook m1 pro of around 700k-800k"
    python -m scripts.try_deal_hunt "macbook m1 pro 700k-800k" "search more"
    python -m scripts.try_deal_hunt "Where can I get a senator?"
    python -m scripts.try_deal_hunt "fridge under 300k" --no-area
"""

import argparse
import json
import sys

import mongomock

from app import config, events
from app.agents import deal_agent
from app.agents.deal_agent import DealAgent
from app.agents.main_agent import MainAgent, _deals_for_model
from app.agents.store_agent import StoreAgent
from app.llm import LLM
from app.store import Store
from app.tools.web_search import read_page, web_search


def heading(text: str, line: str = "=") -> None:
    print(f"\n{line * 78}\n{text}\n{line * 78}")


def tavily(query: str, max_results: int = 5, domains: list[str] | None = None) -> list[dict]:
    results = web_search(query, max_results=max_results, domains=domains)
    print(f"\n    TAVILY: {query!r}" + (f" on {', '.join(domains)}" if domains else "") + f" -> {len(results)} results")
    for n, r in enumerate(results, 1):
        print(f"      [{n}] {r.get('title', '')}\n          {r.get('url', '')}")
        print("          " + (r.get("content") or "")[:300].replace("\n", " ") + " …")
    return results


def reading(url: str) -> str:
    text = read_page(url)
    print(f"\n      READ PAGE: {url} -> " + (f"{len(text):,} characters" if text else "wouldn't open"))
    return text


class ShowingStores(StoreAgent):
    """The real store agent, printing each store the deal agent asks it to look inside."""

    def check(self, url, looking_for):
        print(f"\n    CHECK STORE: {url}\n      looking for: {looking_for}")
        result = super().check(url, looking_for)
        print(f"      answer: {result.get('answer')}")
        for item in result["items"]:
            print(f"        - {item.get('title')} | {item.get('price')} | {item.get('url')}")
        return result


class ShowingDeals(DealAgent):
    """The real deal agent, printing what it's asked and what it answers each round."""

    round = 0

    def find_deals(self, product, location=None, budget=None, must_have=None, brief=None, today=None):
        ShowingDeals.round += 1
        heading(f"DEAL AGENT, CALL {ShowingDeals.round}: what it's asked", "-")
        print(deal_agent._request(product, location, budget, must_have, brief, today))
        result = super().find_deals(product, location, budget, must_have, brief, today)
        heading(f"DEAL AGENT, CALL {ShowingDeals.round}: what it answered", "-")
        print(json.dumps({k: result.get(k) for k in ("deals", "summary")}, indent=2, ensure_ascii=False))
        print(f"(searches: {len(result.get('searches', []))}, store checks: {len(result.get('checks', []))}, "
              f"pages seen: {len(result.get('pages', []))})")
        return result


class NoContacts:
    def find_contacts(self, user_id, need):
        return []


# Steps the helpers above print in full already.
QUIET = {"web_search", "check_store", "read_page", "search_store"}


def show_event(event: dict) -> None:
    if event["type"] == "note":
        print(f"\n  Scout says: {event['text']}")
    elif event["type"] == "step" and event["tool"] not in QUIET:
        print(f"\n> {event['tool']} {json.dumps(event['input'], ensure_ascii=False)}")
    elif event["type"] == "step_done" and event["tool"] not in QUIET:
        print(f"< {event['tool']} done ({event.get('status')})")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # ₦ and friends in Windows terminals
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("messages", nargs="*", default=["Where can I get a macbook m1 pro of around 700k-800k"])
    parser.add_argument("--area", default="Oregun, Ikeja", help="your area, as the app sends it")
    parser.add_argument("--no-area", action="store_true", help="send no area (Scout should ask about the currency)")
    args = parser.parse_args()

    llm = LLM()
    store = Store(client=mongomock.MongoClient())
    print(f"Main agent: {config.MAIN_MODEL_ID}\nDeal agent: {config.BEDROCK_MODEL_ID}\nStore agent: {config.STORE_MODEL_ID}")
    stores = ShowingStores(LLM(model_id=config.STORE_MODEL_ID), read=reading, search=tavily)
    agent = MainAgent(LLM(model_id=config.MAIN_MODEL_ID), store, ShowingDeals(llm, search=tavily, stores=stores),
                      NoContacts(), search=tavily)
    location = None if args.no_area else args.area
    history = None
    for message in args.messages:
        heading(f"YOU: {message}" + (f"   (area: {location})" if location else "   (no area)"))
        ShowingDeals.round = 0
        with events.listen(show_event):
            out = agent.chat("you", message, location, history)
        history = out["history"]

        calls = [c for c in out["tool_calls"] if c["tool"] == "find_deals"]
        for number, call in enumerate(calls, 1):
            result = call["output"]
            if not isinstance(result, dict):
                print(f"\nfind_deals failed: {result}")
                continue
            asked = call["input"]
            heading(f"FIND_DEALS CALL {number} of {len(calls)}: {asked.get('product')!r} in "
                    f"{asked.get('location') or location or 'online'}"
                    + (" (same hunt)" if asked.get("same_hunt") else "") + f", {len(result['deals'])} deals", "-")
            for d in result["deals"]:
                facts = [f"lacks: {d['missing']}" if d.get("missing") else "",
                         f"discount: {d['discount']}" if d.get("discount") else "",
                         f"price from {d['price_year']}" if d.get("price_year") else ""]
                print(f"#{d.get('n')} {d.get('store')} | {d.get('title')} | {d.get('price')}"
                      + "".join(f"   <- {f}" for f in facts if f))
            heading("WHAT THE MAIN AGENT WAS SHOWN (the cards get the full deals)", "-")
            print(json.dumps(_deals_for_model(result), indent=2, ensure_ascii=False))
        heading("SCOUT'S REPLY")
        print(out["reply"])
    print(f"\nRemembered about you: {store.get_profile('you') or 'nothing'}")


if __name__ == "__main__":
    main()
