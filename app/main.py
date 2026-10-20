"""FastAPI entry point. Run with: uvicorn app.main:app --reload"""

import base64
import binascii
import json
import logging
import queue
import threading
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app import checkout, config, events
from app.agents.contact_agent import ContactAgent
from app.agents.deal_agent import DealAgent
from app.agents.main_agent import MainAgent
from app.embeddings import get_embedder
from app.llm import LLM
from app.paypal import PayPal, PayPalError
from app.store import get_store

logger = logging.getLogger(__name__)

app = FastAPI(title="PayPal Shopping Agent")

# Conversation history per user, kept in process memory.
_sessions: dict[str, list] = {}

# Image formats Bedrock accepts, and its per-image size limit.
IMAGE_FORMATS = {"image/png": "png", "image/jpeg": "jpeg", "image/gif": "gif", "image/webp": "webp"}
MAX_IMAGE_BYTES = 3_750_000


@lru_cache(maxsize=1)
def get_llm() -> LLM:
    return LLM()


@lru_cache(maxsize=1)
def get_paypal() -> PayPal:
    return PayPal()


def get_contact_agent() -> ContactAgent:
    return ContactAgent(get_llm(), get_store(), get_embedder())


def get_main_agent(contacts: ContactAgent = Depends(get_contact_agent)) -> MainAgent:
    return MainAgent(get_llm(), get_store(), DealAgent(get_llm()), contacts)


@app.exception_handler(checkout.CheckoutError)
def checkout_error(request: Request, exc: checkout.CheckoutError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(PayPalError)
def paypal_error(request: Request, exc: PayPalError):
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


class ImageIn(BaseModel):
    media_type: str = Field(description="image/png, image/jpeg, image/gif or image/webp")
    data: str = Field(description="Base64-encoded image bytes")


class ChatRequest(BaseModel):
    user_id: str
    message: str
    location: str | None = Field(None, description="Only send if the user allowed location access")
    image: ImageIn | None = Field(None, description="Optional photo, e.g. of the space an appliance goes in")


class Contact(BaseModel):
    name: str
    phone: str | None = None
    email: str | None = None
    notes: str | None = Field(None, description="What the user knows about this contact")


class ImportRequest(BaseModel):
    user_id: str
    contacts: list[Contact]
    enrich_with_web: bool = True


class CartItemRequest(BaseModel):
    deal_id: str


class CheckoutRequest(BaseModel):
    user_id: str


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
    return FileResponse(Path(__file__).parent / "static" / "index.html")


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
    }


@app.post("/chat")
def chat(req: ChatRequest, agent: MainAgent = Depends(get_main_agent)):
    images = [_decode_image(req.image)] if req.image else None
    result = agent.chat(req.user_id, req.message, req.location, _sessions.get(req.user_id), images)
    _sessions[req.user_id] = result.pop("history")
    return result


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, agent: MainAgent = Depends(get_main_agent)):
    """Same as /chat, but streams progress as server-sent events while the agent works:
    note, step and step_done events, then one done (the /chat response) or error event."""
    images = [_decode_image(req.image)] if req.image else None
    updates: queue.Queue = queue.Queue()

    def run():
        with events.listen(updates.put):
            try:
                result = agent.chat(req.user_id, req.message, req.location, _sessions.get(req.user_id), images)
                _sessions[req.user_id] = result.pop("history")
                updates.put({"type": "done", **result})
            except Exception as e:
                logger.exception("chat failed")
                updates.put({"type": "error", "detail": f"{type(e).__name__}: {e}"})
            finally:
                updates.put(None)

    def stream():
        threading.Thread(target=run, daemon=True).start()
        while (event := updates.get()) is not None:
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/chat/{user_id}/reset")
def reset_chat(user_id: str):
    _sessions.pop(user_id, None)
    return {"status": "reset"}


@app.post("/contacts/import")
def import_contacts(req: ImportRequest, agent: ContactAgent = Depends(get_contact_agent)):
    contacts = [c.model_dump() for c in req.contacts]
    return {"imported": agent.import_contacts(req.user_id, contacts, req.enrich_with_web)}


@app.get("/contacts/{user_id}")
def list_contacts(user_id: str):
    return {"contacts": get_store().list_contacts(user_id)}


@app.get("/contacts/{user_id}/search")
def search_contacts(user_id: str, need: str, agent: ContactAgent = Depends(get_contact_agent)):
    return {"contacts": agent.find_contacts(user_id, need)}


@app.get("/users/{user_id}/profile")
def user_profile(user_id: str):
    return {"facts": get_store().get_profile(user_id)}


@app.get("/cart/{user_id}")
def get_cart(user_id: str):
    return checkout.cart_summary(get_store(), user_id)


@app.post("/cart/{user_id}/items")
def add_cart_item(user_id: str, req: CartItemRequest):
    return checkout.add_item(get_store(), user_id, req.deal_id)


@app.delete("/cart/{user_id}/items/{deal_id}")
def remove_cart_item(user_id: str, deal_id: str):
    get_store().remove_from_cart(user_id, [deal_id])
    return checkout.cart_summary(get_store(), user_id)


@app.get("/paypal/config")
def paypal_config():
    """Public settings the browser needs to load the PayPal JS SDK (the client id isn't secret)."""
    return {"client_id": config.PAYPAL_CLIENT_ID, "currency": config.PAYPAL_CURRENCY, "env": config.PAYPAL_ENV}


@app.post("/paypal/orders")
def create_paypal_order(req: CheckoutRequest):
    return checkout.create_order(get_store(), get_paypal(), req.user_id)


@app.post("/paypal/orders/{order_id}/capture")
def capture_paypal_order(order_id: str, req: CheckoutRequest):
    return checkout.capture_order(get_store(), get_paypal(), req.user_id, order_id)
