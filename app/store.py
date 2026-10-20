"""MongoDB storage: user memory ("storage" in the diagram), contact vectors, the deals,
cart and PayPal orders used at checkout, and what the payment agent needs afterwards
(purchases awaiting a receipt, Gmail connections, push subscriptions)."""

import uuid
from datetime import datetime, timezone
from functools import lru_cache

from pymongo import ReturnDocument

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
        self.deals = db["deals"]
        self.deals.create_index([("user_id", 1), ("deal_id", 1)])
        self.carts = db["carts"]
        self.orders = db["orders"]
        self.orders.create_index("order_id", unique=True)
        self.purchases = db["purchases"]
        self.purchases.create_index("purchase_id", unique=True)
        self.purchases.create_index([("status", 1), ("next_check_ts", 1)])
        self.google_accounts = db["google_accounts"]
        self.google_accounts.create_index("user_id", unique=True)
        self.push_subscriptions = db["push_subscriptions"]
        self.push_subscriptions.create_index("endpoint", unique=True)

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

    # --- deals, cart and orders ----------------------------------------------

    def save_deals(self, user_id: str, deals: list[dict]) -> list[dict]:
        """Store deals shown to a user, giving each a short id the model and UI can refer to."""
        saved = [{**d, "deal_id": uuid.uuid4().hex[:8]} for d in deals]
        if saved:
            self.deals.insert_many([{**d, "user_id": user_id, "created_at": _now()} for d in saved])
        return saved

    def get_deal(self, user_id: str, deal_id: str) -> dict | None:
        return self.deals.find_one({"user_id": user_id, "deal_id": deal_id}, _DEAL_FIELDS)

    def add_to_cart(self, user_id: str, deal_id: str) -> None:
        self.carts.update_one({"user_id": user_id}, {"$addToSet": {"deal_ids": deal_id}}, upsert=True)

    def remove_from_cart(self, user_id: str, deal_ids: list[str]) -> None:
        self.carts.update_one({"user_id": user_id}, {"$pull": {"deal_ids": {"$in": deal_ids}}})

    def get_cart(self, user_id: str) -> list[dict]:
        ids = (self.carts.find_one({"user_id": user_id}) or {}).get("deal_ids", [])
        found = self.deals.find({"user_id": user_id, "deal_id": {"$in": ids}}, _DEAL_FIELDS)
        by_id = {d["deal_id"]: d for d in found}
        return [by_id[i] for i in ids if i in by_id]

    def save_order(
        self, user_id: str, order_id: str, deal_ids: list[str], total: str, currency: str,
        merchant: dict, items: list[dict],
    ) -> None:
        self.orders.insert_one({
            "order_id": order_id, "user_id": user_id, "deal_ids": deal_ids, "items": items,
            "total": total, "currency": currency, "merchant": merchant, "status": "CREATED",
            "created_at": _now(),
        })

    def get_order(self, order_id: str) -> dict | None:
        return self.orders.find_one({"order_id": order_id}, {"_id": 0})

    def update_order(self, order_id: str, **fields) -> None:
        self.orders.update_one({"order_id": order_id}, {"$set": {**fields, "updated_at": _now()}})

    # --- purchases the payment agent is following up -------------------------

    def create_purchase(self, purchase: dict) -> bool:
        """Insert unless it already exists; True if this call created it."""
        result = self.purchases.update_one(
            {"purchase_id": purchase["purchase_id"]}, {"$setOnInsert": purchase}, upsert=True,
        )
        return result.upserted_id is not None

    def get_purchase(self, purchase_id: str) -> dict | None:
        return self.purchases.find_one({"purchase_id": purchase_id}, {"_id": 0})

    def list_purchases(self, user_id: str) -> list[dict]:
        fields = {"_id": 0, "checked_message_ids": 0, "updated_at": 0}
        return list(self.purchases.find({"user_id": user_id}, fields).sort("paid_ts", -1))

    def claim_due_purchase(self, statuses: list[str], now: float, next_check_ts: float) -> dict | None:
        """Take one purchase that is due a check and push its next check back, so two
        workers never check the same purchase at once."""
        doc = self.purchases.find_one_and_update(
            {"status": {"$in": statuses}, "next_check_ts": {"$lte": now}},
            {"$set": {"next_check_ts": next_check_ts}},
            return_document=ReturnDocument.AFTER,
        )
        if doc:
            doc.pop("_id")  # not via projection: mongomock then returns None
        return doc

    def update_purchase(self, purchase_id: str, **fields) -> None:
        self.purchases.update_one({"purchase_id": purchase_id}, {"$set": {**fields, "updated_at": _now()}})

    def add_checked_messages(self, purchase_id: str, message_ids: list[str]) -> None:
        self.purchases.update_one(
            {"purchase_id": purchase_id}, {"$addToSet": {"checked_message_ids": {"$each": message_ids}}},
        )

    # --- Gmail connections and push subscriptions ----------------------------

    def save_google_account(self, user_id: str, email: str, refresh_token: str) -> None:
        self.google_accounts.replace_one(
            {"user_id": user_id},
            {"user_id": user_id, "email": email, "refresh_token": refresh_token, "connected_at": _now()},
            upsert=True,
        )

    def get_google_account(self, user_id: str) -> dict | None:
        return self.google_accounts.find_one({"user_id": user_id}, {"_id": 0})

    def delete_google_account(self, user_id: str) -> None:
        self.google_accounts.delete_one({"user_id": user_id})

    def add_push_subscription(self, user_id: str, subscription: dict) -> None:
        self.push_subscriptions.replace_one(
            {"endpoint": subscription["endpoint"]},
            {"user_id": user_id, "endpoint": subscription["endpoint"], "subscription": subscription, "created_at": _now()},
            upsert=True,
        )

    def list_push_subscriptions(self, user_id: str) -> list[dict]:
        return [d["subscription"] for d in self.push_subscriptions.find({"user_id": user_id})]

    def remove_push_subscription(self, endpoint: str) -> None:
        self.push_subscriptions.delete_one({"endpoint": endpoint})


_DEAL_FIELDS = {"_id": 0, "user_id": 0, "created_at": 0}


def _now():
    return datetime.now(timezone.utc)


@lru_cache(maxsize=1)
def get_store() -> Store:
    return Store()
