"""Tavily web search, and reading one page."""

import logging
import re
import time
from functools import lru_cache
from html import unescape
from urllib.parse import urljoin

import requests

from app import config

logger = logging.getLogger(__name__)

# A dropped connection or a timeout is usually a blip: try again after these pauses (seconds)
# before giving up. Other errors (a bad key, a bad request) aren't retried.
RETRY_PAUSES = (1, 3)
_BLIPS = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)

# A store's listing page runs 5-50k characters; the listings are rarely past this.
PAGE_CHARS = 40_000
# Some stores refuse requests that don't look like a browser.
BROWSER = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
           "Chrome/126.0 Safari/537.36")


class WebSearchUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _client():
    if not config.TAVILY_API_KEY:
        raise WebSearchUnavailable("TAVILY_API_KEY is not set")
    from tavily import TavilyClient

    return TavilyClient(api_key=config.TAVILY_API_KEY)


def _retrying(what: str, call):
    for attempt, pause in enumerate((*RETRY_PAUSES, None), 1):
        try:
            return call()
        except _BLIPS as e:
            if pause is None:
                raise
            logger.warning("%s failed (%s), retrying in %ss (attempt %d of %d)",
                           what, type(e).__name__, pause, attempt, len(RETRY_PAUSES) + 1)
            time.sleep(pause)


def web_search(query: str, max_results: int = 5, domains: list[str] | None = None) -> list[dict]:
    """domains: search only these sites (one store's)."""
    extra = {"include_domains": domains} if domains else {}
    resp = _retrying("Tavily search", lambda: _client().search(query=query, max_results=max_results, search_depth="basic", **extra))
    return [
        {"title": r.get("title"), "url": r.get("url"), "content": (r.get("content") or "")[:500]}
        for r in resp.get("results", [])
    ]


def read_page(url: str) -> str:
    """The text of one page with its links, cut to PAGE_CHARS, or "" when it won't open.
    Tavily's advanced extract renders pages a plain fetch can't; when it fails, the page is
    fetched directly, which works for smaller stores."""
    try:
        resp = _retrying("Tavily extract", lambda: _client().extract(urls=[url], extract_depth="advanced"))
        text = next((r["raw_content"] for r in resp.get("results", []) if r.get("raw_content")), "")
    except Exception as e:  # Tavily refusing a page isn't the end: try it directly
        logger.warning("Tavily couldn't read %s: %s: %s", url, type(e).__name__, e)
        text = ""
    if not text:
        try:
            page = _retrying("Fetching the page", lambda: requests.get(url, headers={"User-Agent": BROWSER}, timeout=20))
            page.raise_for_status()
            text = _html_text(page.text, url)
        except requests.exceptions.RequestException as e:
            logger.warning("couldn't fetch %s: %s: %s", url, type(e).__name__, e)
    return text[:PAGE_CHARS]


def _html_text(html: str, base: str) -> str:
    """Visible text, one block per line, with each link as "text (url)"."""
    html = re.sub(r"(?is)<(script|style|noscript|svg|head)\b.*?</\1>", " ", html)
    html = re.sub(r'(?is)<a\b[^>]*?href="([^"#][^"]*)"[^>]*>(.*?)</a>',
                  lambda m: f" {m.group(2)} ({urljoin(base, unescape(m.group(1)))}) ", html)
    html = re.sub(r"(?i)<br\s*/?>|</(p|div|li|tr|h\d|section|article)>", "\n", html)
    text = unescape(re.sub(r"<[^>]+>", " ", html))
    lines = (re.sub(r"\s+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


WEB_SEARCH_SPEC = {
    "name": "web_search",
    "description": "Search the web. Returns titles, URLs and snippets.",
    "inputSchema": {"json": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }},
}
