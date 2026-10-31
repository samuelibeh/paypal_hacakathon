"""Accounts and guests: sign up and sign in with email and password (or Google), kept
signed in by a cookie. Anyone can use Scout as a guest first: they get a guest id in the
same kind of cookie, but one that ends when the browser closes.

Passwords are stored as scrypt hashes, never as typed. The cookie holds the user id, an
expiry and an HMAC signature made with APP_SECRET, so it can't be forged or edited. Every
route that takes a user id checks it against the signed-in user, so nobody can see someone
else's cart, contacts or purchases by changing the id.
"""

import hashlib
import hmac
import secrets
import time

from fastapi import Depends, HTTPException, Request

from app import config

COOKIE = "scout_session"
GUEST_PREFIX = "guest-"
SESSION_DAYS = 30
MIN_PASSWORD = 8


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    return f"scrypt${salt.hex()}${_scrypt(password, salt).hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        _, salt, digest = stored.split("$")
        return hmac.compare_digest(_scrypt(password, bytes.fromhex(salt)).hex(), digest)
    except ValueError:
        return False


def _scrypt(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)


def session_token(user_id: str, now: float | None = None) -> str:
    expires = int((now or time.time()) + SESSION_DAYS * 86400)
    payload = f"{user_id}.{expires}"
    return f"{payload}.{_sign(payload)}"


def user_from_token(token: str, now: float | None = None) -> str | None:
    try:
        user_id, expires, signature = token.rsplit(".", 2)
        valid = hmac.compare_digest(signature, _sign(f"{user_id}.{expires}"))
        return user_id if valid and int(expires) > (now or time.time()) else None
    except ValueError:
        return None


def _sign(payload: str) -> str:
    return hmac.new(config.APP_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()


def current_user(request: Request) -> str:
    """The signed-in user's id; 401 when nobody is signed in."""
    user_id = user_from_token(request.cookies.get(COOKIE) or "")
    if not user_id:
        raise HTTPException(401, "Sign in to continue.")
    return user_id


def owner(user_id: str, me: str = Depends(current_user)) -> str:
    """For routes with a user_id (in the path or query): it must be the signed-in user's."""
    return same_user(user_id, me)


def is_guest(user_id: str) -> bool:
    return user_id.startswith(GUEST_PREFIX)


def new_guest_id() -> str:
    return GUEST_PREFIX + secrets.token_hex(12)


def member(user_id: str, me: str = Depends(current_user)) -> str:
    """Like owner, for things only accounts have: contacts, purchases, Gmail, notifications."""
    same_user(user_id, me)
    if is_guest(me):
        raise HTTPException(403, "Sign up to use this.")
    return me


def same_user(user_id: str, me: str) -> str:
    if user_id != me:
        raise HTTPException(403, "That belongs to another account.")
    return me
