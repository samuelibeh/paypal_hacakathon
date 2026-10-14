"""MongoDB storage: user memory ("storage" in the diagram) and contact vectors."""

import uuid
from datetime import datetime, timezone
from functools import lru_cache

from app import config
from app.embeddings import cosine


class Store:
    def __init__(self, client=None):
        if client is None:
            if config.MONGODB_URI:
                from pymongo import MongoClient

                client = MongoClient(config.MONGODB_URI)
            else:
                import mongomock

                client = mongomock.MongoClient()
        db = client[config.MONGODB_DB]
        self.users = db["users"]
        self.contacts = db["contacts"]
        self.contacts.create_index("user_id")

    # --- user memory -----------------------------------------------------

    def remember(self, user_id: str, key: str, value: str) -> dict:
        self.users.update_one(
            {"user_id": user_id},
            {"$set": {f"facts.{key}": value, "updated_at": _now()}},
            upsert=True,
        )
        return self.get_profile(user_id)

    def get_profile(self, user_id: str) -> dict:
        doc = self.users.find_one({"user_id": user_id}, {"_id": 0}) or {}
        return doc.get("facts", {})

    # --- contacts ----------------------------------------------------------

    def add_contact(self, user_id: str, contact: dict, profile: str, embedding: list[float]) -> str:
        contact_id = contact.get("id") or uuid.uuid4().hex
        self.contacts.replace_one(
            {"user_id": user_id, "contact_id": contact_id},
            {
                "user_id": user_id,
                "contact_id": contact_id,
                "name": contact["name"],
                "phone": contact.get("phone"),
                "email": contact.get("email"),
                "notes": contact.get("notes"),
                "profile": profile,
                "embedding": embedding,
                "updated_at": _now(),
            },
            upsert=True,
        )
        return contact_id

    def list_contacts(self, user_id: str) -> list[dict]:
        return list(self.contacts.find({"user_id": user_id}, {"_id": 0, "embedding": 0}))

    def search_contacts(self, user_id: str, query_vec: list[float], limit: int, threshold: float) -> list[dict]:
        if config.MONGODB_VECTOR_INDEX:
            hits = self.contacts.aggregate([
                {"$vectorSearch": {
                    "index": config.MONGODB_VECTOR_INDEX,
                    "path": "embedding",
                    "queryVector": query_vec,
                    "filter": {"user_id": user_id},
                    "numCandidates": limit * 20,
                    "limit": limit,
                }},
                {"$project": {"_id": 0, "embedding": 0, "score": {"$meta": "vectorSearchScore"}}},
            ])
            # Atlas cosine score is (1 + cos) / 2; convert back so the threshold means the same thing
            hits = [{**h, "score": 2 * h["score"] - 1} for h in hits]
        else:
            hits = []
            for doc in self.contacts.find({"user_id": user_id}, {"_id": 0}):
                vec = doc.pop("embedding")
                hits.append({**doc, "score": cosine(query_vec, vec)})
            hits.sort(key=lambda h: h["score"], reverse=True)
        return [h for h in hits if h["score"] >= threshold][:limit]


def _now():
    return datetime.now(timezone.utc)


@lru_cache(maxsize=1)
def get_store() -> Store:
    return Store()
