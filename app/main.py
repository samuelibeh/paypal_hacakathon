"""FastAPI entry point. Run with: uvicorn app.main:app --reload"""

import base64
import binascii
import json
import logging
import queue
import threading
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from app import auth, checkout, config, contact_sources, events, push
from app.auth import current_user, member, owner, same_user
from app.agents.contact_agent import ContactAgent
from app.agents.deal_agent import DealAgent
from app.agents.main_agent import MainAgent
from app.agents.payment_agent import PaymentAgent
from app.agents.store_agent import StoreAgent
from app.embeddings import get_embedder
from app.gmail import Gmail, GmailError
from app.llm import LLM
from app.paypal import PayPal, PayPalError
from app.store import get_store
from app.tools import product_image
from app.tools.web_search import _client as web_search_client

logger = logging.getLogger(__name__)
startup_log = logging.getLogger("uvicorn.error")  # shown in the uvicorn console
STATIC = Path(__file__).parent / "static"


def _watch_receipts(stop: threading.Event) -> None:
    while not stop.wait(config.RECEIPT_POLL_SECONDS):
        try:
            get_payment_agent().check_due()
        except Exception:
            logger.exception("receipt check failed")


def _look_up_contacts(stop: threading.Event) -> None:
    """Works through queued contact lookups, one at a time, in the background."""
    try:
        get_store().requeue_interrupted_lookups()
    except Exception:
        logger.exception("couldn't requeue interrupted contact lookups")
    while not stop.is_set():
        try:
            busy = get_contact_agent().look_up_next()
        except Exception:
            logger.exception("contact lookup failed")
            busy = False
        stop.wait(1 if busy else 5)


def _warm_up() -> None:
    """Load the slow pieces before the first request instead of during it: the embedding
    model (and its first encode), the database connection and the API clients. A piece that
    fails is logged and loads on first use instead, as before."""
    steps = [
        ("embedding model", lambda: get_embedder().embed_query("warm up")),
        ("database", _ping_database),
        ("Claude client", get_llm),
        ("payment agent", get_payment_agent),
    ]
    if config.TAVILY_API_KEY:
        steps.append(("web search client", web_search_client))
    for name, load in steps:
        started = time.perf_counter()
        try:
            load()
            startup_log.info("warm-up: %s ready in %.1fs", name, time.perf_counter() - started)
        except Exception as e:
            startup_log.warning("warm-up: %s failed, it'll load on first use: %s: %s", name, type(e).__name__, e)


def _ping_database() -> None:
    store = get_store()
    if config.MONGODB_URI:  # the in-memory store has nothing to connect
        import pymongo

        with pymongo.timeout(5):  # don't hold up startup for 30s when Atlas is unreachable
            store.users.database.client.admin.command("ping")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _warm_up()
    stop = threading.Event()
    threading.Thread(target=_watch_receipts, args=(stop,), daemon=True, name="receipt-watcher").start()
    threading.Thread(target=_look_up_contacts, args=(stop,), daemon=True, name="contact-lookups").start()
    yield
    stop.set()


app = FastAPI(title="PayPal Shopping Agent", lifespan=lifespan)

# Image formats Bedrock accepts, and its per-image size limit.
IMAGE_FORMATS = {"image/png": "png", "image/jpeg": "jpeg", "image/gif": "gif", "image/webp": "webp"}
MAX_IMAGE_BYTES = 3_750_000


@lru_cache(maxsize=1)
def get_llm() -> LLM:
    return LLM()


@lru_cache(maxsize=1)
def get_main_llm() -> LLM:
    return LLM(model_id=config.MAIN_MODEL_ID)


@lru_cache(maxsize=1)
def get_store_llm() -> LLM:
    return LLM(model_id=config.STORE_MODEL_ID)


@lru_cache(maxsize=1)
def get_paypal() -> PayPal:
    return PayPal()


@lru_cache(maxsize=1)
def get_gmail() -> Gmail:
    return Gmail(get_store())


@lru_cache(maxsize=1)
def get_payment_agent() -> PaymentAgent:
    return PaymentAgent(get_llm(), get_store(), get_gmail(), push.Notifier(get_store()))


def get_contact_agent() -> ContactAgent:
    return ContactAgent(get_llm(), get_store(), get_embedder())


def get_main_agent(contacts: ContactAgent = Depends(get_contact_agent)) -> MainAgent:
    return MainAgent(get_main_llm(), get_store(), DealAgent(get_llm(), stores=StoreAgent(get_store_llm())), contacts)


@app.exception_handler(checkout.CheckoutError)
def checkout_error(request: Request, exc: checkout.CheckoutError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(PayPalError)
def paypal_error(request: Request, exc: PayPalError):
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.exception_handler(GmailError)
def gmail_error(request: Request, exc: GmailError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


class SignUp(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    email: str = Field(max_length=200)
    password: str = Field(max_length=200)
    location: str | None = Field(None, max_length=120, description="The area they shop in, e.g. Ikeja, Lagos")


class SignIn(BaseModel):
    email: str
    password: str


class AccountEdit(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=80)
    location: str | None = Field(None, max_length=120, description="Empty string clears it")


class ImageIn(BaseModel):
    media_type: str = Field(description="image/png, image/jpeg, image/gif or image/webp")
    data: str = Field(description="Base64-encoded image bytes")
    thumb: str | None = Field(None, max_length=200_000, description="A small JPEG data URL, saved with the chat")


class ChatRequest(BaseModel):
    user_id: str
    message: str
    location: str | None = Field(None, description="Only send if the user allowed location access")
    image: ImageIn | None = Field(None, description="Optional photo, e.g. of the space an appliance goes in")


class Contact(BaseModel):
    name: str | None = None
    phone: str | None = None
    email: str | None = None
    company: str | None = None
    title: str | None = None
    notes: str | None = Field(None, description="What the user knows about this contact")


class ImportRequest(BaseModel):
    user_id: str
    contacts: list[Contact] = Field(max_length=5000)
    source: str = Field("manual", max_length=20, description="google, phone, file or manual")


class ContactFile(BaseModel):
    filename: str = Field(max_length=200)
    text: str = Field(max_length=5_000_000, description="The .vcf or .csv file's contents")


class PeopleAnswer(BaseModel):
    yes: list[str] = Field(default_factory=list, description="Ids of the people that are this contact")
    no: list[str] = Field(default_factory=list, description="Ids of the people that aren't")


CLASH_CHOICES = Literal["replace", "keep_both"]


class ContactAdd(BaseModel):
    contact: Contact
    on_conflict: CLASH_CHOICES | None = Field(None, description="When the number is already saved: "
                                                                "replace that contact, or keep both")


class ContactEdit(BaseModel):
    name: str | None = Field(None, max_length=200)
    phone: str | None = Field(None, max_length=50)
    email: str | None = Field(None, max_length=200)
    company: str | None = Field(None, max_length=200)
    profile: str = Field("", max_length=1000, description="The description; empty clears it")
    on_conflict: CLASH_CHOICES | None = None


class CartItemRequest(BaseModel):
    deal_id: str


class CheckoutRequest(BaseModel):
    user_id: str


class PushKeys(BaseModel):
    p256dh: str
    auth: str


class PushSubscription(BaseModel):
    endpoint: str
    keys: PushKeys


class PushSubscribeRequest(BaseModel):
    user_id: str
    subscription: PushSubscription


def _decode_image(image: ImageIn) -> tuple[str, bytes]:
    fmt = IMAGE_FORMATS.get(image.media_type)
    if not fmt:
        raise HTTPException(400, "Images must be PNG, JPEG, GIF or WebP.")
    try:
        data = base64.b64decode(image.data, validate=True)
    except binascii.Error:
        raise HTTPException(400, "Image data isn't valid base64.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(400, "Images must be under 3.75 MB.")
    return fmt, data


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/sw.js", include_in_schema=False)
def service_worker():
    """Served from the root so it can show notifications for the whole app."""
    return FileResponse(STATIC / "sw.js", media_type="text/javascript")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "storage": "mongodb" if config.MONGODB_URI else "in-memory",
        "web_search": bool(config.TAVILY_API_KEY),
        "embedding_model": config.EMBEDDING_MODEL,
        "paypal": bool(config.PAYPAL_CLIENT_ID and config.PAYPAL_CLIENT_SECRET),
        "paypal_env": config.PAYPAL_ENV,
        "paypal_currency": config.PAYPAL_CURRENCY,
        "gmail": get_gmail().configured,
        "google_verified": config.GOOGLE_APP_VERIFIED,
        "push": push.configured(),
    }


def _start_session(response: Response, request: Request, user_id: str) -> None:
    """Sign this account in. If a guest was using this browser, their chat, cart and the rest
    move into the account."""
    was = auth.user_from_token(request.cookies.get(auth.COOKIE) or "")
    if was and auth.is_guest(was) and not auth.is_guest(user_id):
        get_store().adopt_guest(was, user_id)
    _set_cookie(response, request, user_id)


def _set_cookie(response: Response, request: Request, user_id: str) -> None:
    guest = auth.is_guest(user_id)
    response.set_cookie(
        auth.COOKIE, auth.session_token(user_id),
        max_age=None if guest else auth.SESSION_DAYS * 86400,  # a guest's ends when the browser closes
        httponly=True, samesite="lax", secure=request.url.scheme == "https",
    )


GUEST = {"email": None, "name": None, "location": None, "guest": True}

# The Stop for each user's reply in progress.
_replies: dict[str, threading.Event] = {}


@app.post("/auth/signup")
def sign_up(req: SignUp, request: Request, response: Response):
    email = req.email.strip().lower()
    if "@" not in email or "." not in email.rsplit("@", 1)[-1]:
        raise HTTPException(400, "Enter a valid email address.")
    if len(req.password) < auth.MIN_PASSWORD:
        raise HTTPException(400, f"Use a password of at least {auth.MIN_PASSWORD} characters.")
    store = get_store()
    account = store.create_account(email, req.name.strip(), auth.hash_password(req.password),
                                   (req.location or "").strip() or None)
    if not account:
        raise HTTPException(409, "There's already an account with that email. Sign in instead.")
    store.remember(account["user_id"], "name", account["name"])  # signs the emails sent for them
    _start_session(response, request, account["user_id"])
    return {**account, "guest": False}


@app.post("/auth/signin")
def sign_in(req: SignIn, request: Request, response: Response):
    account = get_store().account_by_email(req.email.strip().lower())
    if not account or not auth.check_password(req.password, account.get("password_hash") or ""):
        raise HTTPException(401, "That email and password don't match an account.")
    _start_session(response, request, account["user_id"])
    return {**get_store().get_account(account["user_id"]), "guest": False}


@app.post("/auth/signout")
def sign_out(response: Response):
    response.delete_cookie(auth.COOKIE)
    return {"status": "signed out"}


@app.get("/auth/me")
def me(request: Request, response: Response):
    """The signed-in account, or the guest using this browser (made on the first visit)."""
    user_id = auth.user_from_token(request.cookies.get(auth.COOKIE) or "")
    account = get_store().get_account(user_id) if user_id and not auth.is_guest(user_id) else None
    if account:
        return {**account, "guest": False}
    if not (user_id and auth.is_guest(user_id)):
        user_id = auth.new_guest_id()
        _set_cookie(response, request, user_id)
    return {"user_id": user_id, **GUEST}


@app.patch("/auth/me")
def edit_me(req: AccountEdit, user_id: str = Depends(current_user)):
    if auth.is_guest(user_id):
        raise HTTPException(403, "Sign up to save your details.")
    fields = req.model_dump(exclude_none=True)
    if "name" in fields:
        fields["name"] = fields["name"].strip()
        get_store().remember(user_id, "name", fields["name"])
    if "location" in fields:
        fields["location"] = fields["location"].strip() or None
    return {**get_store().update_account(user_id, **fields), "guest": False}


@app.get("/geo/reverse")
def reverse_geocode(lat: float, lon: float, _: str = Depends(current_user)):
    """Turn the browser's coordinates into an area name like "Ikeja, Lagos" (OpenStreetMap).
    The coordinates aren't stored; only the area name is used."""
    try:
        resp = httpx.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 14, "accept-language": "en"},
            headers={"User-Agent": "ScoutShoppingAgent/1.0"}, timeout=8,
        )
        resp.raise_for_status()
        address = resp.json().get("address", {})
    except (httpx.HTTPError, ValueError):
        raise HTTPException(502, "Couldn't look up your area. Type it instead.")
    return {"place": _area_name(address)}


def _area_name(address: dict) -> str | None:
    local = next((address[k] for k in ("suburb", "neighbourhood", "city_district", "town", "village")
                  if address.get(k)), None)
    region = address.get("city") or address.get("state")
    parts = [p for p in (local, region) if p]
    return ", ".join(dict.fromkeys(parts)) or address.get("country")


@app.post("/chat")
def chat(req: ChatRequest, agent: MainAgent = Depends(get_main_agent), me: str = Depends(current_user)):
    same_user(req.user_id, me)
    images = [_decode_image(req.image)] if req.image else None
    return _chat_turn(agent, req, images)


def _chat_turn(agent: MainAgent, req: ChatRequest, images) -> dict:
    """One message and reply, carrying on the user's current chat and saving it."""
    store = get_store()
    result = agent.chat(req.user_id, req.message, req.location, store.chat_history(req.user_id), images)
    history = result.pop("history")
    thumb = req.image.thumb if req.image and (req.image.thumb or "").startswith("data:image/jpeg;base64,") else None
    first = store.save_chat_turn(req.user_id, req.message, thumb or bool(images), result, history)
    if first:  # name the chat in the background, so the reply isn't held up
        threading.Thread(target=_name_chat, args=(req.user_id, req.message, result.get("reply", "")), daemon=True).start()
    return result


CHAT_NAME = """<task>
Name a shopping chat in 2 to 5 words from the user's first message and Scout's reply.
</task>

<context>
The name is shown in the list of the user's chats, so it should say what they were shopping for.
</context>

<references>
"Cheap MacBook near Ikeja", "White sticky notes", "Laptop for school and coding".
</references>

<tools>
None.
</tools>

<output>
Plain words, no quotes, no full stop. Reply with only the name.
</output>"""


def _name_chat(user_id: str, message: str, reply: str) -> None:
    try:
        name = get_llm().complete(CHAT_NAME, f"Message: {message[:1000]}\n\nReply: {reply[:600]}", max_tokens=20)
        name = name.strip().strip('"\'.').splitlines()[0][:60] if name.strip() else ""
        if name:
            get_store().set_chat_title(user_id, name)
    except Exception:
        logger.exception("couldn't name the chat; it keeps its first message as the title")


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, agent: MainAgent = Depends(get_main_agent), me: str = Depends(current_user)):
    """Same as /chat, but streams progress as server-sent events while the agent works:
    note, step and step_done events, then one done (the /chat response) or error event."""
    same_user(req.user_id, me)
    images = [_decode_image(req.image)] if req.image else None
    updates: queue.Queue = queue.Queue()
    stop = _replies[req.user_id] = threading.Event()

    def run():
        with events.listen(updates.put), events.stoppable(stop):
            try:
                updates.put({"type": "done", **_chat_turn(agent, req, images)})
            except events.Stopped:  # nothing was saved: the turn is forgotten
                updates.put({"type": "stopped"})
            except Exception as e:
                logger.exception("chat failed")
                updates.put({"type": "error", "detail": f"{type(e).__name__}: {e}"})
            finally:
                if _replies.get(req.user_id) is stop:
                    del _replies[req.user_id]
                updates.put(None)

    def stream():
        threading.Thread(target=run, daemon=True).start()
        while (event := updates.get()) is not None:
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/chat/{user_id}/stop")
def stop_reply(user_id: str, _: str = Depends(owner)):
    """Stop the reply in progress. Scout stops before its next step and keeps nothing from
    the turn, so the model never sees the stopped message."""
    stop = _replies.get(user_id)
    if stop:
        stop.set()
    return {"stopping": bool(stop)}


@app.post("/chat/{user_id}/reset")
def reset_chat(user_id: str, _: str = Depends(owner)):
    get_store().end_chat(user_id)
    return {"status": "reset"}


@app.get("/chat/{user_id}/list")
def list_chats(user_id: str, _: str = Depends(member)):
    """Past chats, for the menu. Guests' chats aren't kept, so this is for accounts."""
    return {"chats": get_store().list_chats(user_id)}


@app.post("/chat/{user_id}/open/{chat_id}")
def open_chat(user_id: str, chat_id: str, _: str = Depends(member)):
    turns = get_store().open_chat(user_id, chat_id)
    if turns is None:
        raise HTTPException(404, "No chat with that id.")
    return {"turns": turns}


@app.get("/chat/{user_id}/current")
def current_chat(user_id: str, _: str = Depends(owner)):
    """The current chat's turns, to show again after a reload."""
    return {"turns": get_store().current_chat(user_id)}


@app.get("/contacts/google/connect", include_in_schema=False)
def google_contacts_connect(user_id: str, _: str = Depends(member)):
    _need_google_approval()
    return RedirectResponse(get_gmail().contacts_url(user_id))


@app.post("/contacts/import")
def import_contacts(req: ImportRequest, agent: ContactAgent = Depends(get_contact_agent),
                    me: str = Depends(current_user)):
    """Save the contacts the user chose. Lookups run in the background."""
    member(req.user_id, me)
    return agent.import_contacts(req.user_id, [c.model_dump() for c in req.contacts], req.source)


@app.post("/contacts/{user_id}/parse-file")
def parse_contact_file(user_id: str, req: ContactFile, _: str = Depends(member)):
    """Read a .vcf or .csv into contacts for the user to review; nothing is saved yet."""
    contacts = contact_sources.parse_file(req.filename, req.text)
    if not contacts:
        raise HTTPException(400, "Couldn't find any contacts in that file. Use a .vcf or .csv export.")
    return {"contacts": contacts}


@app.get("/contacts/{user_id}/pending")
def pending_google_contacts(user_id: str, _: str = Depends(member)):
    """Google Contacts fetched during the consent redirect, waiting for the user to review."""
    return {"contacts": get_store().take_pending_import(user_id)}


@app.post("/contacts/{user_id}/lookup-more")
def look_up_more_contacts(user_id: str, agent: ContactAgent = Depends(get_contact_agent), _: str = Depends(member)):
    return {"queued": agent.look_up_more(user_id)}


@app.get("/contacts/{user_id}")
def list_contacts(user_id: str, _: str = Depends(member)):
    store = get_store()
    return {"contacts": store.list_contacts(user_id), "counts": store.contact_counts(user_id)}


@app.post("/contacts/{user_id}/{contact_id}/people")
def answer_people(user_id: str, contact_id: str, req: PeopleAnswer,
                  agent: ContactAgent = Depends(get_contact_agent), _: str = Depends(member)):
    """Which of the people found online are this contact."""
    contact = agent.confirm_people(user_id, contact_id, req.yes, req.no)
    if not contact:
        raise HTTPException(404, "No contact with that id.")
    return contact


@app.post("/contacts/{user_id}/add")
def add_contact(user_id: str, req: ContactAdd, agent: ContactAgent = Depends(get_contact_agent),
                _: str = Depends(member)):
    """One contact typed in by hand. A number already saved comes back as {"conflict": ...}
    until the user picks on_conflict."""
    try:
        return agent.add_contact(user_id, req.contact.model_dump(), req.on_conflict)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.put("/contacts/{user_id}/{contact_id}")
def edit_contact(user_id: str, contact_id: str, req: ContactEdit,
                 agent: ContactAgent = Depends(get_contact_agent), _: str = Depends(member)):
    """The whole card: name, phone, email, company and description."""
    try:
        contact = agent.edit_contact(user_id, contact_id, req.model_dump(exclude={"on_conflict"}), req.on_conflict)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not contact:
        raise HTTPException(404, "No contact with that id.")
    return contact


@app.post("/contacts/{user_id}/{contact_id}/search-again")
def search_contact_again(user_id: str, contact_id: str, agent: ContactAgent = Depends(get_contact_agent),
                         _: str = Depends(member)):
    try:
        contact = agent.search_again(user_id, contact_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not contact:
        raise HTTPException(404, "No contact with that id.")
    return contact


@app.delete("/contacts/{user_id}/{contact_id}")
def delete_contact(user_id: str, contact_id: str, agent: ContactAgent = Depends(get_contact_agent),
                   _: str = Depends(member)):
    if not agent.delete_contact(user_id, contact_id):
        raise HTTPException(404, "No contact with that id.")
    return {"deleted": contact_id}


@app.get("/contacts/{user_id}/search")
def search_contacts(user_id: str, need: str, agent: ContactAgent = Depends(get_contact_agent),
                    _: str = Depends(member)):
    return {"contacts": agent.find_contacts(user_id, need)}


@app.get("/users/{user_id}/profile")
def user_profile(user_id: str, _: str = Depends(owner)):
    return {"facts": get_store().get_profile(user_id)}


@app.get("/deals/{user_id}/{deal_id}/image")
def deal_image(user_id: str, deal_id: str, _: str = Depends(owner)):
    """The deal's product photo, read from its store page the first time it's asked for.
    The UI calls this after showing the cards, so the reply isn't held up."""
    store = get_store()
    deal = store.get_deal(user_id, deal_id)
    if not deal:
        raise HTTPException(404, "No deal with that id.")
    if "image" not in deal:
        deal["image"] = product_image.find_image(deal.get("url"))
        store.set_deal_image(user_id, deal_id, deal["image"])
    return {"image": deal["image"]}


@app.get("/cart/{user_id}")
def get_cart(user_id: str, _: str = Depends(owner)):
    return checkout.cart_summary(get_store(), user_id)


@app.post("/cart/{user_id}/items")
def add_cart_item(user_id: str, req: CartItemRequest, _: str = Depends(owner)):
    return checkout.add_item(get_store(), user_id, req.deal_id)


@app.delete("/cart/{user_id}/items/{deal_id}")
def remove_cart_item(user_id: str, deal_id: str, _: str = Depends(owner)):
    get_store().remove_from_cart(user_id, [deal_id])
    return checkout.cart_summary(get_store(), user_id)


@app.get("/paypal/config")
def paypal_config():
    """Public settings the browser needs to load the PayPal JS SDK (the client id isn't secret)."""
    return {"client_id": config.PAYPAL_CLIENT_ID, "currency": config.PAYPAL_CURRENCY, "env": config.PAYPAL_ENV}


@app.post("/paypal/orders")
def create_paypal_order(req: CheckoutRequest, me: str = Depends(current_user)):
    same_user(req.user_id, me)
    return checkout.create_order(get_store(), get_paypal(), req.user_id)


@app.post("/paypal/orders/{order_id}/capture")
def capture_paypal_order(order_id: str, req: CheckoutRequest, me: str = Depends(current_user)):
    """The user approved the payment in PayPal; the payment agent pays the store and starts
    watching for the receipt."""
    same_user(req.user_id, me)
    return get_payment_agent().pay(get_paypal(), req.user_id, order_id)


@app.get("/purchases/{user_id}")
def list_purchases(user_id: str, _: str = Depends(member)):
    return {"purchases": get_store().list_purchases(user_id)}


@app.get("/google/status/{user_id}")
def google_status(user_id: str, _: str = Depends(member)):
    gmail = get_gmail()
    return {"configured": gmail.configured, "email": gmail.connected_email(user_id)}


@app.get("/google/connect", include_in_schema=False)
def google_connect(user_id: str, _: str = Depends(member)):
    _need_google_approval()
    account = get_store().get_account(user_id) or {}
    return RedirectResponse(get_gmail().auth_url(user_id, account.get("email")))


def _need_google_approval() -> None:
    if not config.GOOGLE_APP_VERIFIED:
        raise HTTPException(403, "Coming soon: this opens once Google approves Scout.")


@app.get("/auth/google", include_in_schema=False)
def google_sign_in():
    return RedirectResponse(get_gmail().sign_in_url())


def _finish_google_sign_in(request: Request, code: str, state: str) -> RedirectResponse:
    """Sign in the account with this Google email, creating it if it's new. A new account
    goes on to the "where do you shop?" step."""
    try:
        person = get_gmail().sign_in(code, state)
    except GmailError:
        logger.exception("Google sign-in failed")
        return RedirectResponse("/?signin=error")
    store = get_store()
    account = store.account_by_email(person["email"])
    new = account is None
    if new:
        account = store.create_account(person["email"], person["name"], None, None)
        store.remember(account["user_id"], "name", account["name"])
    response = RedirectResponse(f"/?signin={'new' if new else 'google'}")
    _start_session(response, request, account["user_id"])
    return response


def _finish_google_contacts(code: str, state: str) -> RedirectResponse:
    try:
        user_id, people = get_gmail().fetch_contacts(code, state)
    except GmailError:
        logger.exception("Google Contacts import failed")
        return RedirectResponse("/?contacts=error")
    contacts = [c for c in (contact_sources.from_google(p) for p in people) if c]
    get_store().save_pending_import(user_id, contacts)
    return RedirectResponse("/?contacts=google")


@app.get("/google/callback", include_in_schema=False)
def google_callback(request: Request, code: str | None = None, state: str | None = None, error: str | None = None):
    if state and get_gmail().contacts_user(state):
        return _finish_google_contacts(code, state) if code and not error else RedirectResponse("/?contacts=cancelled")
    if state and get_gmail().is_sign_in(state):
        return _finish_google_sign_in(request, code, state) if code and not error else RedirectResponse("/?signin=cancelled")
    if error or not (code and state):
        return RedirectResponse("/?gmail=cancelled")
    try:
        get_gmail().connect(code, state)
    except GmailError:
        logger.exception("Gmail connect failed")
        return RedirectResponse("/?gmail=error")
    return RedirectResponse("/?gmail=connected")


@app.delete("/google/{user_id}")
def google_disconnect(user_id: str, _: str = Depends(member)):
    get_gmail().disconnect(user_id)
    return {"email": None}


@app.get("/push/key")
def push_key():
    """The VAPID public key the browser subscribes with (it isn't secret)."""
    return {"public_key": config.VAPID_PUBLIC_KEY if push.configured() else None}


@app.post("/push/subscribe")
def push_subscribe(req: PushSubscribeRequest, me: str = Depends(current_user)):
    member(req.user_id, me)
    get_store().add_push_subscription(req.user_id, req.subscription.model_dump())
    return {"status": "subscribed"}
