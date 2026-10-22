import base64
import email
import json
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from app import gmail as gmail_module
from app.gmail import Gmail, GmailError, GmailNotConnected
from tests.fakes import fake_store


def b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


class FakeGoogle:
    def __init__(self):
        self.grants, self.sent = [], []
        self.refresh_ok = True

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == gmail_module.TOKEN_URL:
            form = parse_qs(request.content.decode())
            self.grants.append(form["grant_type"][0])
            if form["grant_type"] == ["refresh_token"] and not self.refresh_ok:
                return httpx.Response(400, json={"error": "invalid_grant", "error_description": "Token revoked"})
            return httpx.Response(200, json={"access_token": "tok", "refresh_token": "refresh", "expires_in": 3600})
        assert request.headers["authorization"] == "Bearer tok"
        path = url.removeprefix(gmail_module.API_URL).split("?")[0]
        if path == "/profile":
            return httpx.Response(200, json={"emailAddress": "sam@gmail.com"})
        if path == "/messages":
            return httpx.Response(200, json={"messages": [{"id": "m1"}]})
        if path == "/messages/m1":
            return httpx.Response(200, json={"id": "m1", "snippet": "s", "payload": {
                "mimeType": "multipart/alternative",
                "headers": [{"name": "From", "value": "Shop <orders@shop.com>"}, {"name": "Subject", "value": "Order confirmed"}],
                "parts": [{"mimeType": "text/html", "body": {"data": b64("<style>x{}</style><p>Fridge &amp; ice</p>")}}],
            }})
        if path == "/messages/send":
            self.sent.append(json.loads(request.content)["raw"])
            return httpx.Response(200, json={"id": "sent-1"})
        return httpx.Response(404, json={"error": {"message": "not found"}})


def make():
    google = FakeGoogle()
    client = Gmail(fake_store(), "cid", "secret", "http://localhost/cb", secret="s3cret",
                   http=httpx.Client(transport=httpx.MockTransport(google.handler)))
    return client, google


def state_of(url: str) -> str:
    return parse_qs(urlparse(url).query)["state"][0]


def test_connect_search_and_send():
    client, google = make()
    url = client.auth_url("u:1")
    assert "gmail.readonly" in url and "gmail.send" in url and "access_type=offline" in url

    assert client.connect("code", state_of(url)) == ("u:1", "sam@gmail.com")
    assert client.connected_email("u:1") == "sam@gmail.com"

    [found] = client.search("u:1", "from:shop.com")
    assert found == {"id": "m1", "from": "Shop <orders@shop.com>", "subject": "Order confirmed", "date": "", "text": "Fridge & ice"}

    client.send("u:1", "help@shop.com", "Order confirmation", "Hello,\nI paid.")
    sent = email.message_from_bytes(base64.urlsafe_b64decode(google.sent[0]))
    assert sent["To"] == "help@shop.com" and sent["Subject"] == "Order confirmation"
    assert sent.get_payload(decode=True).decode().strip() == "Hello,\nI paid."


def test_tampered_or_expired_state_is_rejected(monkeypatch):
    client, _ = make()
    state = state_of(client.auth_url("u1"))
    issued, mac, _ = base64.urlsafe_b64decode(state).decode().split(":", 2)
    forged = base64.urlsafe_b64encode(f"{issued}:{mac}:someone-else".encode()).decode()

    with pytest.raises(GmailError, match="invalid"):
        client.connect("code", forged)
    with pytest.raises(GmailError, match="invalid"):
        client.connect("code", "not-a-state")

    monkeypatch.setattr(gmail_module.time, "time", lambda: int(issued) + gmail_module.STATE_MAX_AGE + 1)
    with pytest.raises(GmailError, match="expired"):
        client.connect("code", state)


def test_tokens_are_refreshed_and_revoked_access_disconnects():
    client, google = make()
    client.connect("code", state_of(client.auth_url("u1")))

    client._tokens.clear()  # as after a restart
    client.search("u1", "q")
    assert google.grants == ["authorization_code", "refresh_token"]

    client._tokens.clear()
    google.refresh_ok = False
    with pytest.raises(GmailNotConnected):
        client.search("u1", "q")
    assert client.connected_email("u1") is None

    with pytest.raises(GmailNotConnected):
        client.send("nobody", "a@b.c", "s", "b")
