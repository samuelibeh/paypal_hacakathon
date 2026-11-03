"""When a post went up, read from its link. Some sites put the time inside the post's ID, so the
link itself says when it was posted, with no page to read and no model to trust:

  Instagram  /p/, /reel/, /tv/ codes decode to a media ID whose top bits are milliseconds
             since Instagram's epoch (24 Aug 2011).
  TikTok     /video/ IDs: the top 32 bits are seconds since 1970. Short links
             (vm.tiktok.com/...) redirect to the full one, so they're followed first.
  X/Twitter  /status/ IDs: the top bits are milliseconds since Twitter's epoch (4 Nov 2010).

Facebook IDs carry no time, so its links give nothing here. Checked against real posts: an
Instagram reel from 18 Aug 2022, and TikTok videos from election night, 5-6 Nov 2024.
"""

import logging
import re
from datetime import date, datetime, timedelta, timezone

import requests

logger = logging.getLogger(__name__)

_SHORTCODE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
_INSTAGRAM_EPOCH_MS = 1314220021721
_TWITTER_EPOCH_MS = 1288834974657

_INSTAGRAM = re.compile(r"instagram\.com/(?:[\w.]+/)?(?:p|reels?|tv)/([A-Za-z0-9_-]+)")
_TIKTOK = re.compile(r"tiktok\.com/.*?/(?:video|photo)/(\d{15,20})")
_TIKTOK_SHORT = re.compile(r"(?:vm|vt)\.tiktok\.com/|tiktok\.com/t/")
_TWITTER = re.compile(r"(?:twitter|x)\.com/\w+/status/(\d{15,20})")


def posted_on(url: str | None) -> date | None:
    """The day the post at url went up, or None when its link doesn't say."""
    if not url:
        return None
    if _TIKTOK_SHORT.search(url):
        url = _follow(url)
    when = None
    if m := _INSTAGRAM.search(url):
        code = m.group(1)[:11]  # private posts' codes run longer; the media ID is the first 11
        media_id = 0
        for ch in code:
            media_id = media_id * 64 + _SHORTCODE.index(ch)
        when = _from_ms((media_id >> 23) + _INSTAGRAM_EPOCH_MS)
    elif m := _TIKTOK.search(url):
        when = _from_ms((int(m.group(1)) >> 32) * 1000)
    elif m := _TWITTER.search(url):
        when = _from_ms((int(m.group(1)) >> 22) + _TWITTER_EPOCH_MS)
    # An ID from before these sites existed, or from the future, wasn't a post ID after all.
    if when and date(2010, 1, 1) <= when <= date.today() + timedelta(days=1):
        return when
    return None


def _from_ms(ms: int) -> date | None:
    try:
        return datetime.fromtimestamp(ms / 1000, timezone.utc).date()
    except (OverflowError, OSError, ValueError):
        return None


def _follow(url: str) -> str:
    """Where a short link leads, or the link itself when it can't be followed."""
    try:
        return requests.head(url, allow_redirects=True, timeout=5).url
    except requests.exceptions.RequestException as e:
        logger.warning("couldn't follow %s: %s", url, type(e).__name__)
        return url
