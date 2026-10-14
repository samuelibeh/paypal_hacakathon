"""Tavily web search."""

from functools import lru_cache

from app import config


class WebSearchUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _client():
    if not config.TAVILY_API_KEY:
        raise WebSearchUnavailable("TAVILY_API_KEY is not set")
    from tavily import TavilyClient

    return TavilyClient(api_key=config.TAVILY_API_KEY)


def web_search(query: str, max_results: int = 5) -> list[dict]:
    resp = _client().search(query=query, max_results=max_results, search_depth="basic")
    return [
        {"title": r.get("title"), "url": r.get("url"), "content": (r.get("content") or "")[:500]}
        for r in resp.get("results", [])
    ]


WEB_SEARCH_SPEC = {
    "name": "web_search",
    "description": "Search the web. Returns titles, URLs and snippets.",
    "inputSchema": {"json": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }},
}
