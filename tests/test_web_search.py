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
