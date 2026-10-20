"""Sub-agent: searches the web for discounted prices on a product."""

import json
import re

from app.llm import LLM
from app.tools.web_search import WEB_SEARCH_SPEC, web_search

SYSTEM = """You find the best current deals for a product a shopper wants.
Use web_search several times with different queries (discounts, coupons, sale, compare prices).
If a location is given, prefer stores and marketplaces that serve that location.
If no location is given, search major online stores (Amazon, Walmart, Temu, eBay, Best Buy, etc.).
Never invent prices or stores; only report what the search results show.

Finish with ONLY a JSON object, no prose:
{"deals": [{"store": str, "title": str, "price": str or null, "amount": number or null,
            "currency": str or null, "url": str, "details": str, "note": str}],
 "summary": str}
price is the price as shown ("$499.99"); amount is the same price as a plain number (499.99)
and currency its ISO 4217 code ("USD"). Use null for both if no price is shown.
details lists what the results say that helps someone choose between items: dimensions,
capacity, colours available, material, sizes, standout review points. Leave it empty rather
than guess; the shopper may rely on it to check whether something fits.
List at most 5 deals, cheapest credible offer first."""


class DealAgent:
    def __init__(self, llm: LLM, search=web_search):
        self.llm = llm
        self.search = search

    def find_deals(self, product: str, location: str | None = None) -> dict:
        ask = f"Product: {product}\nLocation: {location or 'not shared - search online stores'}"
        tools = {"web_search": (WEB_SEARCH_SPEC, lambda a: self.search(a["query"]))}
        text, trace = self.llm.run_tools(SYSTEM, [{"role": "user", "content": [{"text": ask}]}], tools)
        result = _parse_json(text) or {"deals": [], "summary": text}
        result["searches"] = [t["input"]["query"] for t in trace]
        return result


def _parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and "deals" in data else None
