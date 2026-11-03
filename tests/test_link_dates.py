from datetime import date

import requests

from app.agents.deal_agent import DealAgent
from app.llm import LLM
from app.tools import link_dates
from app.tools.link_dates import posted_on
from tests.fakes import FakeBedrock, text_reply


def test_instagram_tiktok_and_x_links_say_when_they_were_posted():
    # Real posts, with dates checked against the posts themselves.
    assert posted_on("https://www.instagram.com/reel/ChZiIAWDF7V") == date(2022, 8, 18)
    assert posted_on("https://www.instagram.com/fastestcakes/p/ChZiIAWDF7V/?igsh=x") == date(2022, 8, 18)
    assert posted_on("https://www.tiktok.com/@abcnews/video/7433885220236987679") == date(2024, 11, 5)
    assert posted_on("https://x.com/jack/status/20") is None  # 2006: before X's IDs held a time
    assert posted_on("https://x.com/elonmusk/status/1585341984679469056") == date(2022, 10, 26)


def test_links_without_a_post_date_give_none():
    assert posted_on("https://www.instagram.com/fastestcakes/") is None  # a profile, not a post
    assert posted_on("https://www.facebook.com/marketplace/item/1234567890123456") is None
    assert posted_on("https://jiji.ng/lagos/cakes/birthday-cake.html") is None
    assert posted_on(None) is None


def test_a_tiktok_short_link_is_followed_to_the_video(monkeypatch):
    class Followed:
        url = "https://www.tiktok.com/@abcnews/video/7433885220236987679"

    monkeypatch.setattr(link_dates.requests, "head", lambda url, **kw: Followed())
    assert posted_on("https://vm.tiktok.com/ZMabc123/") == date(2024, 11, 5)

    def offline(url, **kw):
        raise requests.exceptions.ConnectionError("offline")

    monkeypatch.setattr(link_dates.requests, "head", offline)
    assert posted_on("https://vm.tiktok.com/ZMabc123/") is None


def test_the_link_date_overrides_the_year_the_model_gave():
    final = ('{"deals": [{"store": "Fastest Cakes", "title": "Cake", "price": "₦6,500", "price_year": 2026,'
             ' "url": "https://www.instagram.com/reel/ChZiIAWDF7V"},'
             ' {"store": "Jiji", "title": "Cake", "price": "₦7,000", "price_year": 2026,'
             ' "url": "https://jiji.ng/lagos/cakes/a.html"}], "summary": "s"}')
    agent = DealAgent(LLM(FakeBedrock(text_reply(final)), "m"), search=lambda q: [])

    cake, other = agent.find_deals("cake")["deals"]

    assert cake["price_year"] == 2022 and cake["posted"] == "2022-08-18"
    assert other["price_year"] == 2026 and "posted" not in other  # no date in its link: the model's stands
