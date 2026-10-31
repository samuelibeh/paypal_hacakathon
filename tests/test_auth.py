from fastapi.testclient import TestClient

from app import auth, main
from tests.fakes import fake_store, signed_in


def _client(monkeypatch):
    store = fake_store()
    monkeypatch.setattr(main, "get_store", lambda: store)
    return TestClient(main.app), store


def test_passwords_are_hashed_and_checked():
    stored = auth.hash_password("correct horse")
    assert "correct horse" not in stored
    assert auth.check_password("correct horse", stored)
    assert not auth.check_password("wrong horse", stored)


def test_session_tokens_cant_be_edited_and_expire():
    token = auth.session_token("u1", now=1000)
    assert auth.user_from_token(token, now=2000) == "u1"
    assert auth.user_from_token(token.replace("u1", "u2", 1), now=2000) is None
    assert auth.user_from_token(token, now=1000 + auth.SESSION_DAYS * 86400 + 1) is None
    assert auth.user_from_token("junk") is None


def test_sign_up_sign_out_and_sign_in(monkeypatch):
    client, store = _client(monkeypatch)
    assert client.get("/auth/me").json()["guest"] is True

    resp = client.post("/auth/signup", json={
        "name": "Samuel", "email": " Sam@Example.com ", "password": "longenough", "location": "Ikeja, Lagos"})
    assert resp.status_code == 200
    me = client.get("/auth/me").json()
    assert me == {"user_id": "demo-user", "email": "sam@example.com", "name": "Samuel", "location": "Ikeja, Lagos",
                  "guest": False}
    assert store.get_profile("demo-user")["name"] == "Samuel"  # signs emails sent for them

    client.post("/auth/signout")
    assert client.get("/auth/me").json()["guest"] is True  # signed out: a guest again
    assert client.post("/auth/signin", json={"email": "sam@example.com", "password": "nope"}).status_code == 401
    assert client.post("/auth/signin", json={"email": "SAM@example.com", "password": "longenough"}).status_code == 200
    assert client.get("/auth/me").json()["name"] == "Samuel"


def test_sign_up_rules(monkeypatch):
    client, _ = _client(monkeypatch)
    ok = {"name": "A", "email": "a@example.com", "password": "longenough"}
    assert client.post("/auth/signup", json={**ok, "password": "short"}).status_code == 400
    assert client.post("/auth/signup", json={**ok, "email": "not-an-email"}).status_code == 400
    assert client.post("/auth/signup", json=ok).status_code == 200
    assert client.post("/auth/signup", json=ok).status_code == 409  # email taken
    second = TestClient(main.app).post("/auth/signup", json={**ok, "email": "b@example.com"}).json()
    assert second["user_id"] != "demo-user"  # only the first account takes over the demo data


def test_editing_name_and_area(monkeypatch):
    client, store = _client(monkeypatch)
    client.post("/auth/signup", json={"name": "Sam", "email": "s@example.com", "password": "longenough"})

    assert client.patch("/auth/me", json={"location": "Lekki, Lagos"}).json()["location"] == "Lekki, Lagos"
    assert client.patch("/auth/me", json={"name": "Samuel B"}).json()["name"] == "Samuel B"
    assert store.get_profile("demo-user")["name"] == "Samuel B"
    assert client.patch("/auth/me", json={"location": " "}).json()["location"] is None


def test_you_can_only_reach_your_own_data(monkeypatch):
    _, store = _client(monkeypatch)
    client = signed_in(main.app, "alice")

    assert client.get("/cart/alice").status_code == 200
    assert client.get("/cart/bob").status_code == 403
    assert client.get("/contacts/bob").status_code == 403
    assert client.post("/paypal/orders", json={"user_id": "bob"}).status_code == 403
    assert TestClient(main.app).get("/cart/alice").status_code == 401  # not signed in


def test_area_name_from_an_openstreetmap_address():
    assert main._area_name({"suburb": "Ikeja", "city": "Lagos", "country": "Nigeria"}) == "Ikeja, Lagos"
    assert main._area_name({"city": "Brooklyn", "state": "New York"}) == "Brooklyn"
    assert main._area_name({"town": "Ojo", "state": "Lagos State"}) == "Ojo, Lagos State"
    assert main._area_name({"country": "Nigeria"}) == "Nigeria"


def test_google_sign_in_creates_then_reuses_the_account(monkeypatch):
    client, store = _client(monkeypatch)

    class FakeGoogle:
        def contacts_user(self, state):
            return None

        def is_sign_in(self, state):
            return state == "signin-state"

        def sign_in(self, code, state):
            return {"email": "sam@gmail.com", "name": "Sam B"}

    monkeypatch.setattr(main, "get_gmail", lambda: FakeGoogle())
    first = client.get("/google/callback?code=c&state=signin-state", follow_redirects=False)
    assert first.headers["location"] == "/?signin=new"  # goes on to "where do you shop?"
    assert client.get("/auth/me").json()["email"] == "sam@gmail.com"

    client.post("/auth/signout")
    again = client.get("/google/callback?code=c&state=signin-state", follow_redirects=False)
    assert again.headers["location"] == "/?signin=google"
    assert client.get("/auth/me").json()["name"] == "Sam B"
    # a Google account has no password, so password sign-in never matches it
    assert client.post("/auth/signin", json={"email": "sam@gmail.com", "password": ""}).status_code == 401


def test_guests_get_an_id_for_this_browser_and_can_shop_but_not_keep_contacts(monkeypatch):
    client, store = _client(monkeypatch)
    first = client.get("/auth/me")
    guest = first.json()["user_id"]
    assert guest.startswith("guest-")
    assert "max-age" not in first.headers["set-cookie"].lower()  # ends when the browser closes
    assert client.get("/auth/me").json()["user_id"] == guest  # same guest on the next visit

    assert client.get(f"/cart/{guest}").status_code == 200
    assert client.get(f"/contacts/{guest}").status_code == 403
    assert client.get(f"/purchases/{guest}").status_code == 403
    assert client.patch("/auth/me", json={"location": "Ikeja"}).status_code == 403


def test_signing_up_keeps_what_the_guest_did(monkeypatch):
    client, store = _client(monkeypatch)
    guest = client.get("/auth/me").json()["user_id"]
    [deal] = store.save_deals(guest, [{"title": "Fridge", "amount": "10.00", "currency": "USD", "payable": True}])
    store.add_to_cart(guest, deal["deal_id"])
    store.save_chat_turn(guest, "find me a fridge", False, {"reply": "Here you go"}, [{"role": "user", "content": []}])
    store.remember(guest, "budget", "$600")

    account = client.post("/auth/signup", json={"name": "Sam", "email": "s@example.com", "password": "longenough"}).json()

    user = account["user_id"]
    assert [i["title"] for i in store.get_cart(user)] == ["Fridge"]
    assert store.current_chat(user)[0]["reply"] == {"reply": "Here you go"}
    assert store.get_profile(user) == {"name": "Sam", "budget": "$600"}
    assert store.get_cart(guest) == [] and store.current_chat(guest) == []


def test_chats_are_saved_without_photos_and_new_chat_starts_fresh():
    store = fake_store()
    history = [{"role": "user", "content": [{"text": "this one?"}, {"image": {"format": "jpeg", "source": {"bytes": b"..."}}}]}]
    store.save_chat_turn("u1", "this one?", True, {"reply": "A red bag"}, history)

    assert store.chat_history("u1")[0]["content"] == [{"text": "this one?"}, {"text": "[The user attached a photo here.]"}]
    assert store.current_chat("u1") == [{"message": "this one?", "photo": True, "reply": {"reply": "A red bag"}}]

    store.end_chat("u1")
    assert store.current_chat("u1") == [] and store.chat_history("u1") is None


def test_past_chats_are_listed_newest_first_and_can_be_reopened(monkeypatch):
    client, store = _client(monkeypatch)
    client.post("/auth/signup", json={"name": "Sam", "email": "s@example.com", "password": "longenough"})
    store.save_chat_turn("demo-user", "Where can I get the cheapest MacBook around me", False, {"reply": "a"}, [])
    store.end_chat("demo-user")
    store.save_chat_turn("demo-user", "white sticky notes", False, {"reply": "b"}, [])

    chats = client.get("/chat/demo-user/list").json()["chats"]
    assert [c["title"] for c in chats] == ["white sticky notes", "Where can I get the cheapest MacBook around me"]
    assert [c["current"] for c in chats] == [True, False]

    turns = client.post(f"/chat/demo-user/open/{chats[1]['chat_id']}").json()["turns"]
    assert turns[0]["reply"] == {"reply": "a"}
    assert [c["current"] for c in client.get("/chat/demo-user/list").json()["chats"]] == [False, True]
    assert client.post("/chat/demo-user/open/nope").status_code == 404
    assert signed_in(main.app, "guest-x").get("/chat/guest-x/list").status_code == 403


def test_a_new_chat_is_named_once_after_its_first_reply():
    store = fake_store()
    long = "I am looking for a really affordable laptop for school that can handle coding and design work please"
    assert store.save_chat_turn("u1", long, False, {"reply": "a"}, []) is True
    assert store.save_chat_turn("u1", "under 500k", False, {"reply": "b"}, []) is False
    assert store.list_chats("u1")[0]["title"].endswith("…")  # the first message, cut, until it's named
    store.set_chat_title("u1", "Laptop for school and coding")
    assert store.list_chats("u1")[0]["title"] == "Laptop for school and coding"


def test_gmail_and_google_contacts_wait_for_google_approval(monkeypatch):
    from app import config
    client = signed_in(main.app, "u1")
    monkeypatch.setattr(main, "get_store", fake_store)
    monkeypatch.setattr(config, "GOOGLE_APP_VERIFIED", False)
    for url in ("/google/connect?user_id=u1", "/contacts/google/connect?user_id=u1"):
        resp = client.get(url, follow_redirects=False)
        assert resp.status_code == 403 and "Coming soon" in resp.json()["detail"]
    assert client.get("/health").json()["google_verified"] is False
