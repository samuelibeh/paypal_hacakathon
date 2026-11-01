import pytest
import requests

from app.tools import web_search as ws


class FlakyTavily:
    """Fails with the given errors first, then answers."""

    def __init__(self, *errors):
        self.errors, self.calls = list(errors), 0

    def search(self, **kwargs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return {"results": [{"title": "t", "url": "https://a.com", "content": "c"}]}


def _use(monkeypatch, tavily):
    monkeypatch.setattr(ws, "_client", lambda: tavily)
    monkeypatch.setattr(ws.time, "sleep", lambda s: None)


def test_a_dropped_connection_is_retried(monkeypatch):
    tavily = FlakyTavily(requests.exceptions.ConnectionError("reset"), requests.exceptions.Timeout("slow"))
    _use(monkeypatch, tavily)
    assert ws.web_search("q") == [{"title": "t", "url": "https://a.com", "content": "c"}]
    assert tavily.calls == 3


def test_it_gives_up_after_the_retries_and_other_errors_arent_retried(monkeypatch):
    reset = requests.exceptions.ConnectionError("reset")
    tavily = FlakyTavily(reset, reset, reset)
    _use(monkeypatch, tavily)
    with pytest.raises(requests.exceptions.ConnectionError):
        ws.web_search("q")
    assert tavily.calls == 3

    tavily = FlakyTavily(ValueError("bad key"))
    _use(monkeypatch, tavily)
    with pytest.raises(ValueError):
        ws.web_search("q")
    assert tavily.calls == 1


class Page:
    def __init__(self, text, status=200):
        self.text, self.status_code = text, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(str(self.status_code))


def test_a_page_tavily_cant_read_is_fetched_directly_with_its_links(monkeypatch):
    class Refusing:
        def extract(self, **kwargs):
            return {"results": [], "failed_results": [{"url": kwargs["urls"][0], "error": "Failed to fetch url"}]}

    _use(monkeypatch, Refusing())
    html = ('<html><head><title>x</title></head><body><script>var a=1</script>'
            '<div><a href="/car/1">2011 Nissan Sentra</a> $2,500</div><p>Ikeja</p></body></html>')
    monkeypatch.setattr(ws.requests, "get", lambda url, **kw: Page(html))

    assert ws.read_page("https://shop.ng/cars") == "2011 Nissan Sentra (https://shop.ng/car/1) $2,500\nIkeja"


def test_a_page_that_wont_open_reads_as_empty(monkeypatch):
    class Broken:
        def extract(self, **kwargs):
            raise RuntimeError("Tavily is down")

    _use(monkeypatch, Broken())
    monkeypatch.setattr(ws.requests, "get", lambda url, **kw: Page("", status=403))
    assert ws.read_page("https://www.cars.com/x") == ""
