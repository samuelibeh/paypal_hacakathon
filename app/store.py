"""MongoDB storage: user memory ("storage" in the diagram), contact vectors, the deals,
cart and PayPal orders used at checkout, and what the payment agent needs afterwards
(purchases awaiting a receipt, Gmail connections, push subscriptions), and accounts."""

import json
import uuid
from datetime import datetime, timezone
from functools import lru_cache

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

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
        self.contacts.create_index([("user_id", 1), ("key", 1)])
        self.contacts.create_index([("status", 1), ("created_at", 1)])
        self.pending_imports = db["pending_imports"]
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
        self.chats = db["chats"]
        self.chats.create_index([("user_id", 1), ("active", 1)])
        self.accounts = db["accounts"]
        self.accounts.create_index("email", unique=True)
        self.accounts.create_index("user_id", unique=True)
        self.visits = db["visits"]
        self.visits.create_index([("device_id", 1), ("at", -1)])
        self.visits.create_index([("user_id", 1), ("at", -1)])

    # --- visits ------------------------------------------------------------

    def record_visit(self, device_id: str, user_id: str, signed_in: bool, device: str, browser: str,
                     user_agent: str, ip: str | None) -> None:
        """One opening of the app on one device, signed in or as a guest."""
        self.visits.insert_one({"device_id": device_id, "user_id": user_id, "signed_in": signed_in,
                                "device": device, "browser": browser, "user_agent": user_agent,
                                "ip": ip, "at": _now()})

    # --- accounts ----------------------------------------------------------

    def create_account(self, email: str, name: str, password_hash: str | None, location: str | None) -> dict | None:
        # password_hash is None for accounts made with "Continue with Google"
        """The new account, or None if the email is taken. The first account gets the
        "demo-user" id, so it keeps what was saved before accounts existed."""
        user_id = "demo-user" if self.accounts.count_documents({}) == 0 else uuid.uuid4().hex
        account = {"user_id": user_id, "email": email, "name": name, "password_hash": password_hash,
                   "location": location, "created_at": _now()}
        try:
            self.accounts.insert_one(account)
        except DuplicateKeyError:
            return None
        return _public(account)

    def account_by_email(self, email: str) -> dict | None:
        """Includes the password hash, for checking a sign-in."""
        return self.accounts.find_one({"email": email}, {"_id": 0})

    def get_account(self, user_id: str) -> dict | None:
        account = self.accounts.find_one({"user_id": user_id}, {"_id": 0})
        return _public(account) if account else None

    def update_account(self, user_id: str, **fields) -> dict | None:
        self.accounts.update_one({"user_id": user_id}, {"$set": fields})
        return self.get_account(user_id)

    def adopt_guest(self, guest_id: str, user_id: str) -> None:
        """Move what a guest did into the account they just signed up or signed in to: their
        chat (it becomes the current one), deals, cart, orders, purchases and remembered facts."""
        if self.chats.find_one({"user_id": guest_id, "active": True}):
            self.end_chat(user_id)
        for collection in (self.chats, self.deals, self.orders, self.purchases):
            collection.update_many({"user_id": guest_id}, {"$set": {"user_id": user_id}})
        cart = (self.carts.find_one({"user_id": guest_id}) or {}).get("deal_ids", [])
        if cart:
            self.carts.update_one({"user_id": user_id}, {"$addToSet": {"deal_ids": {"$each": cart}}}, upsert=True)
        self.carts.delete_one({"user_id": guest_id})
        known = self.get_profile(user_id)
        for key, value in self.get_profile(guest_id).items():
            if key not in known:  # what the account already knows wins, e.g. its name
                self.remember(user_id, key, value)
        self.users.delete_one({"user_id": guest_id})

    # --- chats ---------------------------------------------------------------
    # The current conversation per user. Turns are what the app shows; history is what the
    # model sees. Both are kept as JSON text, and photos are left out of the history.

    def chat_history(self, user_id: str) -> list | None:
        doc = self.chats.find_one({"user_id": user_id, "active": True}, {"history": 1})
        return json.loads(doc["history"]) if doc and doc.get("history") else None

    def save_chat_turn(self, user_id: str, message: str, had_photo: bool | str, reply: dict, history: list) -> bool:
        """had_photo is a small data-URL copy of the photo when the app sent one, else True/False.
        Returns True when this turn started a new chat."""
        turn = json.dumps({"message": message, "photo": had_photo, "reply": reply}, default=str)
        doc = self.chats.find_one_and_update(
            {"user_id": user_id, "active": True},
            {"$push": {"turns": turn},
             "$set": {"history": json.dumps(_without_photos(history), default=str), "updated_at": _now()},
             "$setOnInsert": {"chat_id": uuid.uuid4().hex, "started_at": _now()}},
            upsert=True, return_document=ReturnDocument.AFTER,
        )
        first = len(doc.get("turns", [])) == 1  # the chat may already exist, holding a hunt
        if first and not doc.get("title"):
            self.chats.update_one({"_id": doc["_id"]}, {"$set": {"title": _chat_title(message)}})
        return first

    # The current hunt (one search for one item) lives with the chat, so New chat starts clean.
    def get_hunt(self, user_id: str) -> dict | None:
        doc = self.chats.find_one({"user_id": user_id, "active": True}, {"hunt": 1})
        return json.loads(doc["hunt"]) if doc and doc.get("hunt") else None

    def save_hunt(self, user_id: str, hunt: dict) -> None:
        self.chats.update_one(
            {"user_id": user_id, "active": True},
            {"$set": {"hunt": json.dumps(hunt, default=str)},
             "$setOnInsert": {"chat_id": uuid.uuid4().hex, "started_at": _now()}},
            upsert=True,
        )

    def set_chat_title(self, user_id: str, title: str) -> None:
        self.chats.update_one({"user_id": user_id, "active": True}, {"$set": {"title": title}})

    def current_chat(self, user_id: str) -> list[dict]:
        doc = self.chats.find_one({"user_id": user_id, "active": True}, {"turns": 1})
        return [json.loads(t) for t in (doc or {}).get("turns", [])]

    def list_chats(self, user_id: str, limit: int = 30) -> list[dict]:
        """The user's chats, newest first: {chat_id, title, updated_at, current}."""
        docs = self.chats.find({"user_id": user_id}, {"chat_id": 1, "title": 1, "turns": {"$slice": 1},
                                                      "updated_at": 1, "active": 1}).sort([("updated_at", -1), ("_id", -1)]).limit(limit)
        return [{"chat_id": d["chat_id"], "updated_at": d.get("updated_at"), "current": bool(d.get("active")),
                 "title": d.get("title") or _chat_title(json.loads(d["turns"][0])["message"] if d.get("turns") else "")}
                for d in docs]

    def open_chat(self, user_id: str, chat_id: str) -> list[dict] | None:
        """Make an earlier chat the current one again. Returns its turns, or None if it isn't theirs."""
        if not self.chats.find_one({"user_id": user_id, "chat_id": chat_id}, {"_id": 1}):
            return None
        self.end_chat(user_id)
        self.chats.update_one({"user_id": user_id, "chat_id": chat_id}, {"$set": {"active": True}})
        return self.current_chat(user_id)

    def end_chat(self, user_id: str) -> None:
        """New chat: the current one is kept but no longer current."""
        self.chats.update_many({"user_id": user_id, "active": True}, {"$set": {"active": False, "ended_at": _now()}})

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

    # A contact's status: queued / looking_up / waiting (over the lookup cap) / check (asking
    # which of several people it is) / done (has a description) / nothing (looked up, nothing
    # useful) / notes (being described from the user's notes) / skipped (a single name).
    # A contact searched again keeps its description while queued, looked up or checked.

    def save_contact(self, user_id: str, contact: dict, key: str, source: str, status: str) -> tuple[str, bool]:
        """Add a contact, or update the one with the same key. Returns (contact_id, is_new)."""
        fields = {k: contact.get(k) for k in ("name", "phone", "email", "company", "title", "notes")}
        existing = self.contacts.find_one({"user_id": user_id, "key": key}, {"contact_id": 1})
        if existing:
            # keep what's known; only fill in details the earlier copy didn't have
            self.contacts.update_one({"_id": existing["_id"]},
                                     {"$set": {**{k: v for k, v in fields.items() if v}, "updated_at": _now()}})
            return existing["contact_id"], False
        contact_id = uuid.uuid4().hex
        self.contacts.insert_one({
            "user_id": user_id, "contact_id": contact_id, "key": key, "source": source, **fields,
            "status": status, "people": [], "profile": "", "profile_source": None, "links": [],
            "embedding": None, "attempts": 0, "created_at": _now(), "updated_at": _now(),
            # when it joined the lookup queue: the app's progress counts this run's lookups only
            "queued_at": _now() if status == "queued" else None,
        })
        return contact_id, True

    def get_contact(self, user_id: str, contact_id: str) -> dict | None:
        return self.contacts.find_one({"user_id": user_id, "contact_id": contact_id}, {"_id": 0, "embedding": 0})

    def update_contact(self, user_id: str, contact_id: str, **fields) -> None:
        self.contacts.update_one({"user_id": user_id, "contact_id": contact_id},
                                 {"$set": {**fields, "updated_at": _now()}})

    def contact_with_key(self, user_id: str, key: str, other_than: str | None = None) -> dict | None:
        """The saved contact with this key (same number, else email, else name), if any."""
        query = {"user_id": user_id, "key": key}
        if other_than:
            query["contact_id"] = {"$ne": other_than}
        return self.contacts.find_one(query, {"_id": 0, "embedding": 0})

    def delete_contact(self, user_id: str, contact_id: str) -> bool:
        return self.contacts.delete_one({"user_id": user_id, "contact_id": contact_id}).deleted_count == 1

    def set_contact_profile(self, user_id: str, contact_id: str, profile: str, embedding: list[float] | None,
                            profile_source: str | None = None, links: list[str] | None = None) -> bool:
        fields = {"profile": profile, "profile_source": profile_source, "embedding": embedding,
                  "status": "done" if profile else "skipped", "updated_at": _now()}
        if links is not None:
            fields["links"] = links
        result = self.contacts.update_one({"user_id": user_id, "contact_id": contact_id}, {"$set": fields})
        return result.matched_count == 1

    def claim_contact_lookup(self) -> dict | None:
        """The oldest queued contact, marked as being looked up so only one worker takes it."""
        doc = self.contacts.find_one_and_update(
            {"status": "queued"}, {"$set": {"status": "looking_up", "updated_at": _now()}},
            sort=[("created_at", 1)], return_document=ReturnDocument.AFTER,
        )
        if doc:  # dropped here rather than by projection, which mongomock mishandles here
            doc.pop("_id", None)
            doc.pop("embedding", None)
        return doc

    def requeue_interrupted_lookups(self) -> None:
        """Lookups cut off by a restart start again."""
        self.contacts.update_many({"status": "looking_up"}, {"$set": {"status": "queued"}})

    def queue_lookup(self, user_id: str, contact_id: str) -> None:
        """Look this contact up again; what it has stays until the lookup finds something."""
        self.update_contact(user_id, contact_id, status="queued", queued_at=_now(), attempts=0, people=[])

    def queue_waiting_contacts(self, user_id: str, limit: int) -> int:
        ids = [c["contact_id"] for c in self.contacts.find({"user_id": user_id, "status": "waiting"}, {"contact_id": 1})
               .sort("created_at", 1).limit(limit)]
        self.contacts.update_many({"user_id": user_id, "contact_id": {"$in": ids}},
                                  {"$set": {"status": "queued", "queued_at": _now()}})
        return len(ids)

    def contact_counts(self, user_id: str) -> dict:
        counts = {}
        for c in self.contacts.find({"user_id": user_id}, {"status": 1, "people": 1}):
            counts[c.get("status") or "done"] = counts.get(c.get("status") or "done", 0) + 1
        return counts

    def list_contacts(self, user_id: str) -> list[dict]:
        return list(self.contacts.find({"user_id": user_id}, {"_id": 0, "embedding": 0, "key": 0}).sort("name", 1))

    # Google Contacts come back through a redirect, so they wait here until the app shows them.
    def save_pending_import(self, user_id: str, contacts: list[dict]) -> None:
        self.pending_imports.replace_one({"user_id": user_id}, {"user_id": user_id, "contacts": contacts,
                                                                "created_at": _now()}, upsert=True)

    def take_pending_import(self, user_id: str) -> list[dict]:
        doc = self.pending_imports.find_one_and_delete({"user_id": user_id})
        return (doc or {}).get("contacts", [])

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
            # contacts with no profile have no embedding and are never matched
            for doc in self.contacts.find({"user_id": user_id, "embedding": {"$ne": None}}, {"_id": 0}):
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

    def set_deal_image(self, user_id: str, deal_id: str, image: str | None) -> None:
        """None is stored too, so a page without a photo isn't fetched again."""
        self.deals.update_one({"user_id": user_id, "deal_id": deal_id}, {"$set": {"image": image}})

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


def _chat_title(message: str) -> str:
    """A chat is named after its first message."""
    words = " ".join((message or "").split())
    return (words[:57] + "…") if len(words) > 58 else (words or "New chat")


def _without_photos(history: list) -> list:
    """Photos are large and only needed on the turn they were sent."""
    return [
        {**m, "content": [{"text": "[The user attached a photo here.]"} if "image" in b else b for b in m.get("content", [])]}
        for m in history
    ]


def _public(account: dict) -> dict:
    return {k: account.get(k) for k in ("user_id", "email", "name", "location")}


def _now():
    return datetime.now(timezone.utc)


@lru_cache(maxsize=1)
def get_store() -> Store:
    return Store()
