"""Finds a deal's product photo: the preview image its page publishes for link sharing
(og:image, or twitter:image). Deal URLs come from the model, so only public http(s)
addresses are fetched, redirects included, with a short timeout and a size cap."""

import html
import ipaddress
import re
import socket
from functools import lru_cache
from urllib.parse import urljoin, urlsplit

import httpx

MAX_BYTES = 600_000  # the preview tags sit in <head>, well before this
MAX_REDIRECTS = 3
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ScoutShoppingAgent/1.0)",
    "Accept": "text/html,application/xhtml+xml",
}
# In order of preference.
IMAGE_KEYS = ("og:image:secure_url", "og:image", "og:image:url", "twitter:image", "twitter:image:src")

_META = re.compile(r"<meta\b[^>]*>", re.I)
_ATTR = re.compile(r"""([\w:-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""")


@lru_cache(maxsize=1)
def _client() -> httpx.Client:
    return httpx.Client(timeout=5, follow_redirects=False, headers=HEADERS)


def find_image(url: str | None, http: httpx.Client | None = None, resolve=socket.getaddrinfo) -> str | None:
    """The product photo's URL, or None if the page has none or can't be read."""
    http = http or _client()
    for _ in range(MAX_REDIRECTS + 1):
        if not url or not _is_public(url, resolve):
            return None
        try:
            with http.stream("GET", url) as resp:
                if resp.is_redirect:
                    url = urljoin(url, resp.headers.get("location", ""))
                    continue
                if resp.status_code != 200 or "html" not in resp.headers.get("content-type", ""):
                    return None
                page = _read(resp)
        except httpx.HTTPError:
            return None
        return image_from_html(page, url)
    return None


def image_from_html(page: str, page_url: str) -> str | None:
    found = {}
    for tag in _META.findall(page):
        attrs = {m.group(1).lower(): next(v for v in m.groups()[1:] if v is not None) for m in _ATTR.finditer(tag)}
        key = (attrs.get("property") or attrs.get("name") or "").lower()
        if key in IMAGE_KEYS and attrs.get("content", "").strip():
            found.setdefault(key, attrs["content"].strip())
    for key in IMAGE_KEYS:
        if key in found:
            image = urljoin(page_url, html.unescape(found[key]))
            if urlsplit(image).scheme in ("http", "https"):
                return image
    return None


def _is_public(url: str, resolve) -> bool:
    """True only for http(s) URLs whose host resolves to public internet addresses."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    try:
        addresses = {info[4][0] for info in resolve(parts.hostname, None)}
        return bool(addresses) and all(ipaddress.ip_address(a.split("%")[0]).is_global for a in addresses)
    except (OSError, UnicodeError, ValueError):
        return False


def _read(resp: httpx.Response) -> str:
    chunks, size = [], 0
    for chunk in resp.iter_bytes():
        chunks.append(chunk)
        size += len(chunk)
        if size >= MAX_BYTES:
            break
    return b"".join(chunks).decode(resp.charset_encoding or "utf-8", errors="replace")
