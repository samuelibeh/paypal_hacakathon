"""Sub-agent of the deal agent: checks one store for what the deal agent is hunting.

Search results only show a snippet of a store's page ("1229 vehicles under $2,500"), not the
listings on it. When the deal agent finds a promising page, it calls check_store, and this
agent reads the page (or searches within the store's site) and answers in short: whether the
store has it, and up to ITEMS listings with their price and link. The page's full text stays
here, so the deal agent's context holds only the answer.

It has CALL_LIMIT calls in all. Network blips are retried inside a call and don't count.
"""

import json
import re
from urllib.parse import urlparse

from app.llm import LLM
from app.tools.web_search import read_page, web_search

CALL_LIMIT = 2
MAX_TURNS = 4
# gpt-oss reasons before answering, and the reasoning counts toward this.
MAX_TOKENS = 2048
ITEMS = 3

SYSTEM = f"""<task>
You check one online store for Scout's deal finder. You're given a page of the store and what
the shopper is looking for. Read the page. If it doesn't show what you need (it wouldn't open,
or shows no prices), search the store's site. Then say whether the store has it.
Never invent items, prices or links; report only what the page or the search shows.
</task>

<tools>
read_page(url): the page's text, with its links. search_store(query): searches only this
store's site. {CALL_LIMIT} calls in all.
</tools>

<output>
Finish with ONLY a JSON object, no prose:
{{"found": bool, "answer": str, "items": [{{"title": str, "price": str, "url": str, "details": str}}]}}
answer is one or two sentences: what the store has that fits, or why not ("the cheapest is
$3,400", "the page wouldn't open"). items are at most {ITEMS} listings that fit, best fit first,
each with its own listing link where the page gives one, else the page's link; details are
facts that help choose (year, mileage, condition, location). items is empty when none fit.
</output>"""

READ_PAGE_SPEC = {
    "name": "read_page",
    "description": "Read a page of the store: its text, with its links.",
    "inputSchema": {"json": {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    }},
}

SEARCH_STORE_SPEC = {
    "name": "search_store",
    "description": "Search only this store's site. Returns titles, URLs and snippets.",
    "inputSchema": {"json": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }},
}

# gpt-oss writes prices with narrow and no-break spaces ("$2,500 – 2011 Nissan").
_SPACES = str.maketrans({" ": " ", " ": " ", " ": " ", " ": " "})


class StoreAgent:
    def __init__(self, llm: LLM, read=read_page, search=web_search):
        self.llm = llm
        self.read = read
        self.search = search

    def check(self, url: str, looking_for: str) -> dict:
        """Returns {"found", "answer", "items": [{title, price, url, details}]}."""
        calls = 0
        domain = urlparse(url).netloc.removeprefix("www.")

        def limited(run):
            def handler(args):
                nonlocal calls
                calls += 1
                if calls > CALL_LIMIT:
                    return f"Call limit reached ({CALL_LIMIT}). Answer now with the JSON."
                return run(args)
            return handler

        tools = {
            "read_page": (READ_PAGE_SPEC, limited(lambda a: self.read(a["url"]) or "The page wouldn't open.")),
            "search_store": (SEARCH_STORE_SPEC, limited(lambda a: self.search(a["query"], domains=[domain]))),
        }
        request = f"<request>\nStore page: {url}\nLooking for: {looking_for}\n</request>"
        text, _ = self.llm.run_tools(SYSTEM, [{"role": "user", "content": [{"text": request}]}],
                                     tools, max_turns=MAX_TURNS, max_tokens=MAX_TOKENS)
        text = text.translate(_SPACES)
        result = _parse_json(text) or {"found": False, "answer": text or "The store check gave no answer.", "items": []}
        result["items"] = [i for i in result.get("items") or [] if isinstance(i, dict)][:ITEMS]
        result["found"] = bool(result.get("found")) and bool(result["items"])
        return result


def _parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and "answer" in data else None
