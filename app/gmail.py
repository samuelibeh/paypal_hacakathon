"""Gmail for the payment agent: find a store's receipt in the user's inbox, and email the
store from the user's own account when no receipt arrives.

Users connect with Google OAuth, granting read access to their mail and permission to send.
Their refresh token is kept in storage; access tokens are cached in memory.
"""

import base64
import hashlib
import hmac
import html
import re
import time
from email.message import EmailMessage
from urllib.parse import urlencode

import httpx

from app import config

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
API_URL = "https://gmail.googleapis.com/gmail/v1/users/me"
SCOPES = "https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/gmail.send"
STATE_MAX_AGE = 600


class GmailError(RuntimeError):
    pass


class GmailNotConnected(GmailError):
    pass


class Gmail:
    def __init__(
        self,
        store,
        client_id: str | None = config.GOOGLE_CLIENT_ID,
        client_secret: str | None = config.GOOGLE_CLIENT_SECRET,
        redirect_uri: str = config.GOOGLE_REDIRECT_URI,
        secret: str = config.APP_SECRET,
        http: httpx.Client | None = None,
    ):
        self.store = store
        self.client_id, self.client_secret, self.redirect_uri = client_id, client_secret, redirect_uri
        self.secret = secret.encode()
        self.http = http or httpx.Client(timeout=30)
        self._tokens: dict[str, tuple[str, float]] = {}

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    # --- connecting ----------------------------------------------------------

    def auth_url(self, user_id: str) -> str:
        if not self.configured:
            raise GmailError("Gmail isn't set up: set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.")
        return AUTH_URL + "?" + urlencode({
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": SCOPES,
            "access_type": "offline",
            "prompt": "consent",  # so Google returns a refresh token every time
            "state": self._sign(user_id),
        })

    def connect(self, code: str, state: str) -> tuple[str, str]:
        """Finish the OAuth flow. Returns (user_id, gmail address)."""
        user_id = self._verify(state)
        tokens = self._send("POST", TOKEN_URL, data={
            "code": code, "client_id": self.client_id, "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri, "grant_type": "authorization_code",
        })
        if "refresh_token" not in tokens:
            raise GmailError("Google didn't grant offline access. Try connecting again.")
        self._tokens[user_id] = (tokens["access_token"], time.time() + tokens.get("expires_in", 0) - 60)
        email = self._api(user_id, "GET", "/profile")["emailAddress"]
        self.store.save_google_account(user_id, email, tokens["refresh_token"])
        return user_id, email

    def connected_email(self, user_id: str) -> str | None:
        account = self.store.get_google_account(user_id)
        return account["email"] if account else None

    def disconnect(self, user_id: str) -> None:
        self._tokens.pop(user_id, None)
        self.store.delete_google_account(user_id)

    # --- mail ----------------------------------------------------------------

    def search(self, user_id: str, query: str, limit: int = 10) -> list[dict]:
        """Messages matching a Gmail search, newest first: [{id, from, subject, date, text}]."""
        found = self._api(user_id, "GET", "/messages", params={"q": query, "maxResults": limit})
        return [
            _parse_message(self._api(user_id, "GET", f"/messages/{m['id']}", params={"format": "full"}))
            for m in found.get("messages", [])
        ]

    def send(self, user_id: str, to: str, subject: str, body: str) -> str:
        """Send a plain-text email from the user's Gmail. Returns the message id."""
        message = EmailMessage()
        message["To"] = to
        message["Subject"] = subject
        message.set_content(body)
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        return self._api(user_id, "POST", "/messages/send", json={"raw": raw})["id"]

    # --- plumbing ------------------------------------------------------------

    def _mac(self, issued: str, user_id: str) -> str:
        return hmac.new(self.secret, f"{issued}:{user_id}".encode(), hashlib.sha256).hexdigest()

    def _sign(self, user_id: str) -> str:
        issued = str(int(time.time()))
        return base64.urlsafe_b64encode(f"{issued}:{self._mac(issued, user_id)}:{user_id}".encode()).decode()

    def _verify(self, state: str) -> str:
        try:
            issued, mac, user_id = base64.urlsafe_b64decode(state.encode()).decode().split(":", 2)
            age = time.time() - int(issued)
        except ValueError:
            raise GmailError("That Gmail sign-in link is invalid. Start again from the app.")
        if not hmac.compare_digest(mac, self._mac(issued, user_id)):
            raise GmailError("That Gmail sign-in link is invalid. Start again from the app.")
        if age > STATE_MAX_AGE:
            raise GmailError("That Gmail sign-in link has expired. Start again from the app.")
        return user_id

    def _access_token(self, user_id: str) -> str:
        token, expiry = self._tokens.get(user_id, (None, 0.0))
        if token and time.time() < expiry:
            return token
        account = self.store.get_google_account(user_id)
        if not account:
            raise GmailNotConnected("Gmail isn't connected for this user.")
        try:
            data = self._send("POST", TOKEN_URL, data={
                "client_id": self.client_id, "client_secret": self.client_secret,
                "refresh_token": account["refresh_token"], "grant_type": "refresh_token",
            })
        except GmailError as e:
            if "invalid_grant" in str(e):  # the user revoked access
                self.disconnect(user_id)
                raise GmailNotConnected("Gmail access was revoked. Connect it again in the app.") from e
            raise
        self._tokens[user_id] = (data["access_token"], time.time() + data.get("expires_in", 0) - 60)
        return data["access_token"]

    def _api(self, user_id: str, method: str, path: str, **kwargs) -> dict:
        headers = {"Authorization": f"Bearer {self._access_token(user_id)}"}
        return self._send(method, API_URL + path, headers=headers, **kwargs)

    def _send(self, method: str, url: str, **kwargs) -> dict:
        try:
            resp = self.http.request(method, url, **kwargs)
        except httpx.HTTPError as e:
            raise GmailError(f"Couldn't reach Google: {e}") from e
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if resp.status_code >= 400:
            error = data.get("error")
            detail = error.get("message") if isinstance(error, dict) else f"{error}: {data.get('error_description', '')}"
            raise GmailError(f"Google error {resp.status_code}: {detail or resp.text[:200]}")
        return data


def _parse_message(data: dict) -> dict:
    payload = data.get("payload") or {}
    headers = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}
    return {
        "id": data["id"],
        "from": headers.get("from", ""),
        "subject": headers.get("subject", ""),
        "date": headers.get("date", ""),
        "text": (_body_text(payload) or data.get("snippet", ""))[:4000],
    }


def _body_text(payload: dict) -> str:
    plain, rich = [], []

    def walk(part):
        data = (part.get("body") or {}).get("data")
        if data and part.get("mimeType") == "text/plain":
            plain.append(_decode(data))
        elif data and part.get("mimeType") == "text/html":
            rich.append(_decode(data))
        for child in part.get("parts") or []:
            walk(child)

    walk(payload)
    return "\n".join(plain).strip() or _strip_html("\n".join(rich))


def _decode(data: str) -> str:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", "replace")


def _strip_html(markup: str) -> str:
    markup = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", markup)
    text = html.unescape(re.sub(r"(?s)<[^>]+>", " ", markup))
    return re.sub(r"\s+", " ", text).strip()
