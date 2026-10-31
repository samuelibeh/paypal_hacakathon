"""Tavily web search."""

import logging
import time
from functools import lru_cache

import requests

from app import config

logger = logging.getLogger(__name__)

# A dropped connection or a timeout is usually a blip: try again after these pauses (seconds)
# before giving up. Other errors (a bad key, a bad request) aren't retried.
RETRY_PAUSES = (1, 3)
_BLIPS = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)


class WebSearchUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _client():
    if not config.TAVILY_API_KEY:
        raise WebSearchUnavailable("TAVILY_API_KEY is not set")
    from tavily import TavilyClient

    return TavilyClient(api_key=config.TAVILY_API_KEY)


def web_search(query: str, max_results: int = 5) -> list[dict]:
    for attempt, pause in enumerate((*RETRY_PAUSES, None), 1):
        try:
            resp = _client().search(query=query, max_results=max_results, search_depth="basic")
            break
        except _BLIPS as e:
            if pause is None:
                raise
            logger.warning("Tavily search failed (%s), retrying in %ss (attempt %d of %d)",
                           type(e).__name__, pause, attempt, len(RETRY_PAUSES) + 1)
            time.sleep(pause)
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
