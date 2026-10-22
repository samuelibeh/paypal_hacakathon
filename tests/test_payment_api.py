from fastapi.testclient import TestClient

from app import main
from app.gmail import Gmail
from app.store import get_store


def test_google_connect_redirects_to_consent(monkeypatch):
    monkeypatch.setattr(main, "get_gmail", lambda: Gmail(get_store(), "cid", "secret", "http://localhost/cb"))

    resp = TestClient(main.app).get("/google/connect?user_id=u1", follow_redirects=False)

    assert resp.status_code == 307
    assert resp.headers["location"].startswith("https://accounts.google.com/o/oauth2/v2/auth?")


def test_google_connect_without_credentials_explains_what_to_set():
    resp = TestClient(main.app).get("/google/connect?user_id=u1", follow_redirects=False)

    assert resp.status_code == 400 and "GOOGLE_CLIENT_ID" in resp.json()["detail"]


def test_cancelled_consent_returns_to_the_app():
    resp = TestClient(main.app).get("/google/callback?error=access_denied", follow_redirects=False)

    assert resp.headers["location"] == "/?gmail=cancelled"


def test_push_subscription_is_stored_and_service_worker_served():
    client = TestClient(main.app)
    sub = {"endpoint": "https://push.test/abc", "expirationTime": None, "keys": {"p256dh": "p", "auth": "a"}}

    assert client.post("/push/subscribe", json={"user_id": "api-user", "subscription": sub}).status_code == 200
    assert get_store().list_push_subscriptions("api-user") == [{"endpoint": sub["endpoint"], "keys": sub["keys"]}]

    worker = client.get("/sw.js")
    assert worker.headers["content-type"].startswith("text/javascript") and "showNotification" in worker.text
    assert client.get("/purchases/api-user").json() == {"purchases": []}
