"""Sub-agent: searches the web for deals on a product that meet the shopper's requirements.

Each call is a fresh conversation. When it continues a hunt ("search more", or the main
agent's broadened retry), the main agent passes a brief of the earlier rounds: the searches
already run, the pages already seen and the deals already found, so this round digs further
instead of repeating itself. The pages this round sees are returned too, so the next brief can
include them. It reports facts about each deal (price, discount, what it lacks, the price's
year); judging them is the main agent's job.

Search results show only a snippet of each page, so when a store's page looks promising but
shows no prices, it can call check_store: the store agent looks inside that store and answers
in short, with listings it can report as deals.
"""

import json
import re
from datetime import date

from app.agents.store_agent import StoreAgent
from app.llm import LLM
from app.tools.link_dates import posted_on
from app.tools.web_search import WEB_SEARCH_SPEC, web_search

# The agent is told it has SEARCH_LIMIT searches and CHECK_LIMIT store checks; MAX_TURNS leaves
# it turns to spare to write its answer. Calls past a limit aren't run.
SEARCH_LIMIT = 8
CHECK_LIMIT = 3
MAX_TURNS = 14
# Five deals with long links and details overran the default 1024 and cut the JSON off.
MAX_TOKENS = 4096

SYSTEM = f"""<task>
You are Scout's deal finder: a sharp, thorough shopper who finds where a product can be
bought right now, at the best price, from a seller someone could actually buy from.

1. Read the request: the product, the shopper's location, their budget and their must-haves.
   Specifics in the product name (a model, a chip, a size, like "M1 Pro") count as must-haves
   too, even when they aren't listed separately.
2. Always start with discounts: your first searches look for the product on sale, discounted
   or on promotion where they are (sale, discount, promo, clearance, the season's sales).
   Then search more widely with different queries: compare prices, the product with the city
   or country, local marketplaces and well-known stores. You have {SEARCH_LIMIT} searches in
   all, so make each one count, and leave yourself room to answer.
3. If a location is given, prefer stores and marketplaces that serve that location; if they
   have little, also try online stores that deliver there. If not, search major online stores.
4. If the request includes earlier rounds of this hunt, don't run those searches again or
   return those deals again. Go further: other stores and marketplaces, used or refurbished
   listings, other names for the product, other wordings of the budget.
5. Put deals that are within the budget and have every must-have first, discounted ones before
   full-price ones. If none are, return the closest ones; Scout's main agent judges them.
6. Prices change, so look for this year's stock and prices: the request gives today's date.
   Add the year to some of your queries ("... price Lagos 2026"). Prefer listings and prices from
   this year. Report an older price only when you find nothing current, and give its year.
7. Never invent prices, stores or links; report only what the search results and store
   checks show.
8. Never ask the shopper questions; they can't reply to you. Search with what you have; for a
   broad product ("car"), search its common kinds.
</task>

<context>
You work for Scout, a shopping assistant. The main agent sends you one request at a time and
shows your deals to the shopper as cards, so each deal must stand on its own. A price you
report must be the price shown in the results, in the currency shown. The budget is in the
currency stated in the request; a deal in another currency may still fit, so report its own
price and currency and leave the comparison to Scout.
</context>

<references>
Good first queries: "MacBook Pro M1 Pro discount Lagos", "MacBook M1 Pro sale Nigeria 2026".
Good queries after: "MacBook Pro M1 Pro 14 price Lagos", "used MacBook M1 Pro Computer Village",
"MacBook M1 Pro Jiji Nigeria", "MacBook M1 Pro deal Slot", "refurbished MacBook M1 Pro Nigeria".
Weak queries: "MacBook" alone, or the same query with one word swapped.
A credible deal names the seller and the item, and has a link to its listing or store page.
For the link, prefer the seller's own listing or store page over a page that gathers posts from
many sellers (an Instagram "popular" or hashtag page); when only such a page is available, put
the seller's name and contact from the result in note so the shopper can reach them.
A listing with no seller, a page that only reviews the product, or a price from years ago is
not a deal.
</references>

<tools>
web_search(query): searches the web and returns titles, links and text from the top results.
At most {SEARCH_LIMIT} calls; further calls aren't run.
check_store(url, looking_for): looks inside one store's page and says in short whether it has
what you're looking for, with up to 3 listings (price, link) you can report as deals. Use it
on a promising store or listing page whose results show no prices. At most {CHECK_LIMIT} calls.
</tools>

<output>
Finish with ONLY a JSON object, no prose:
{{"deals": [{{"store": str, "title": str, "price": str or null, "amount": number or null,
            "currency": str or null, "url": str, "details": str, "note": str,
            "discount": str or null, "missing": str or null, "price_year": number or null}}],
 "summary": str}}
price is the price as shown ("$499.99"); amount is the same price as a plain number (499.99)
and currency its ISO 4217 code ("USD"). Use null for both if no price is shown.
discount is the saving the results show ("was ₦950,000", "15% off", "promo code SAVE10 for
₦20,000 off", "Black Friday price till 30 Nov"), or null when it's full price.
details lists what the results say that helps someone choose between items: dimensions,
capacity, colours available, material, sizes, standout review points. Leave it empty rather
than guess; the shopper may rely on it to check whether something fits.
missing says which must-have the item lacks, as what it is instead ("M1, not M1 Pro"), or
null when the results show it has them all. If a listing claims a spec the item can't have,
or one its own details contradict (a "13-inch M1 Pro": the M1 Pro came only in 14- and
16-inch), say so in missing rather than trusting the title.
price_year is the year the price is from when the result shows a date (a dated post, "prices
as of March 2023"), or null when it shows none. Never fill in this year for an undated listing.
List at most 5 deals, best fit first, then cheapest credible offer.
</output>"""


CHECK_STORE_SPEC = {
    "name": "check_store",
    "description": "Look inside one store's page for what you're hunting. Returns a short answer and up to 3 listings.",
    "inputSchema": {"json": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "the store or listing page"},
            "looking_for": {"type": "string", "description": "the item, its must-haves and the budget"},
        },
        "required": ["url", "looking_for"],
    }},
}


class DealAgent:
    def __init__(self, llm: LLM, search=web_search, stores: StoreAgent | None = None):
        self.llm = llm
        self.search = search
        self.stores = stores

    def find_deals(self, product: str, location: str | None = None, budget: dict | None = None,
                   must_have: list[str] | None = None, brief: str | None = None, today: date | None = None) -> dict:
        """budget: {"min", "max", "currency"} (any may be missing). brief: the earlier rounds of
        this hunt, written by the main agent's side. today: so it looks for this year's prices.
        Returns the deals, the searches run, the pages the searches showed ({title, url}) and
        the store checks ({url, answer})."""
        searches, checks, pages = 0, 0, []

        def search(args):
            nonlocal searches
            searches += 1
            if searches > SEARCH_LIMIT:
                return f"Search limit reached ({SEARCH_LIMIT}). Answer now with the JSON from what you found."
            results = self.search(args["query"])
            pages.extend({"title": r.get("title", ""), "url": r.get("url", "")} for r in results or [] if r.get("url"))
            return results

        def check_store(args):
            nonlocal checks
            checks += 1
            if not self.stores:
                return "Store checks aren't available. Carry on searching."
            if checks > CHECK_LIMIT:
                return f"Check limit reached ({CHECK_LIMIT}). Carry on with what you have."
            return self.stores.check(args["url"], args["looking_for"])

        tools = {"web_search": (WEB_SEARCH_SPEC, search), "check_store": (CHECK_STORE_SPEC, check_store)}
        text, trace = self.llm.run_tools(
            SYSTEM, [{"role": "user", "content": [{"text": _request(product, location, budget, must_have, brief, today)}]}],
            tools, max_turns=MAX_TURNS, max_tokens=MAX_TOKENS,
        )
        result = _parse_json(text) or {"deals": [], "summary": text}
        for deal in result["deals"]:
            # An Instagram, TikTok or X link says when it was posted; that beats the model's
            # reading of the page, which once took a 2022 reel for 2026.
            if isinstance(deal, dict) and (posted := posted_on(deal.get("url"))):
                deal["price_year"] = posted.year
                deal["posted"] = posted.isoformat()
        result["searches"] = [t["input"]["query"] for t in trace if t["tool"] == "web_search"]
        result["pages"] = pages
        result["checks"] = [{"url": t["input"].get("url"), "answer": t["output"].get("answer")}
                            for t in trace if t["tool"] == "check_store" and isinstance(t["output"], dict)]
        return result


def _request(product, location, budget, must_have, brief, today: date | None = None) -> str:
    today = today or date.today()
    lines = [f"Product: {product}", f"Location: {location or 'not shared - search online stores'}",
             f"Today: {today.day} {today:%B %Y} (look for {today.year} prices)"]
    if budget and (budget.get("min") or budget.get("max")):
        span = " to ".join(f"{budget[k]:,.0f}" for k in ("min", "max") if budget.get(k))
        lines.append(f"Budget: {span} {budget.get('currency') or ''}".rstrip())
    if must_have:
        lines.append("Must have: " + "; ".join(must_have))
    request = "<request>\n" + "\n".join(lines) + "\n</request>"
    return request + (f"\n\n<earlier_rounds>\n{brief}\n</earlier_rounds>" if brief else "")


def _parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and "deals" in data else None
