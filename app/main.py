"""FastAPI entry point. Run with: uvicorn app.main:app --reload"""

from functools import lru_cache

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from app import config
from app.agents.contact_agent import ContactAgent
from app.agents.deal_agent import DealAgent
from app.agents.main_agent import MainAgent
from app.embeddings import get_embedder
from app.llm import LLM
from app.store import get_store

app = FastAPI(title="PayPal Shopping Agent")

# Conversation history per user, kept in process memory.
_sessions: dict[str, list] = {}


@lru_cache(maxsize=1)
def get_llm() -> LLM:
    return LLM()


def get_contact_agent() -> ContactAgent:
    return ContactAgent(get_llm(), get_store(), get_embedder())


def get_main_agent(contacts: ContactAgent = Depends(get_contact_agent)) -> MainAgent:
    return MainAgent(get_llm(), get_store(), DealAgent(get_llm()), contacts)


class ChatRequest(BaseModel):
    user_id: str
    message: str
    location: str | None = Field(None, description="Only send if the user allowed location access")


class Contact(BaseModel):
    name: str
    phone: str | None = None
    email: str | None = None
    notes: str | None = Field(None, description="What the user knows about this contact")


class ImportRequest(BaseModel):
    user_id: str
    contacts: list[Contact]
    enrich_with_web: bool = True


@app.get("/health")
def health():
    return {
        "status": "ok",
        "storage": "mongodb" if config.MONGODB_URI else "in-memory",
        "web_search": bool(config.TAVILY_API_KEY),
        "embedding_model": config.EMBEDDING_MODEL,
    }


@app.post("/chat")
def chat(req: ChatRequest, agent: MainAgent = Depends(get_main_agent)):
    result = agent.chat(req.user_id, req.message, req.location, _sessions.get(req.user_id))
    _sessions[req.user_id] = result.pop("history")
    return result


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
