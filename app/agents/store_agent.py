"""Sub-agent of the deal agent: checks one store for what the deal agent is hunting.

Search results only show a snippet of a store's page ("1229 vehicles under $2,500"), not the
listings on it. When the deal agent finds a promising page, it calls check_store, and this
agent reads the page (or searches within the store's site) and answers in short: whether the
store has it, and up to ITEMS listings with their price and link. The page's full text stays
here, so the deal agent's context holds only the answer.

It has CALL_LIMIT calls in all. Network blips are retried inside a call and don't count.
"""

import json
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
        # Never the raw text: it's shown under the check, and it may be half a JSON object.
        result = _parse_json(text) or {"found": False, "answer": "The store check didn't give a clear answer.", "items": []}
        result["items"] = _distinct([i for i in result.get("items") or [] if isinstance(i, dict)])[:ITEMS]
        result["found"] = bool(result.get("found")) and bool(result["items"])
        return result


def _distinct(items: list[dict]) -> list[dict]:
    """gpt-oss sometimes lists one listing several times; the same link counts once."""
    seen, kept = set(), []
    for item in items:
        key = item.get("url") or (item.get("title"), item.get("price"))
        if key not in seen:
            seen.add(key)
            kept.append(item)
    return kept


def _parse_json(text: str) -> dict | None:
    """The answer's JSON object. gpt-oss sometimes leaves off the closing brackets
    ('..."items": [{...}]' with no final '}'), so a text that doesn't read is tried again
    with the brackets it left open closed."""
    start = text.find("{")
    if start < 0:
        return None
    body = text[start:]
    for candidate in (body, body + _closers(body)):
        try:
            data, _ = json.JSONDecoder().raw_decode(candidate)  # ignores anything after the object
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "answer" in data:
            return data
    return None


def _closers(text: str) -> str:
    """What closes the strings, lists and objects a JSON text left open, innermost first."""
    open_, in_string, escaped = [], False, False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            open_.append("}" if ch == "{" else "]")
        elif ch in "}]" and open_:
            open_.pop()
    return ('"' if in_string else "") + "".join(reversed(open_))
