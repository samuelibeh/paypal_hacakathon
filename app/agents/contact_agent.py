"""Sub-agent: profiles imported contacts and finds ones who can help with a purchase.

On import, each contact gets a short profile paragraph (what they do, sell or know
about) written from the user's notes and, if enabled, a web search. The profile is
embedded and stored in MongoDB; searches embed the shopping need and return the
contacts whose profiles are closest by cosine similarity.
"""

from app import config
from app.embeddings import Embedder
from app.llm import LLM
from app.store import Store
from app.tools.web_search import WebSearchUnavailable, web_search

PROFILE_SYSTEM = """You write a 2-3 sentence profile of one of the user's contacts for a
shopping assistant. Focus only on what is useful for shopping help: their job, business,
products or services they sell, things they are known to have bought or know about.
Use only the notes and search results given. If search results are clearly about a
different person, ignore them. Do not include home addresses or other sensitive personal
details. If little is known, say so briefly."""


class ContactAgent:
    def __init__(self, llm: LLM, store: Store, embedder: Embedder, search=web_search):
        self.llm = llm
        self.store = store
        self.embedder = embedder
        self.search = search

    def import_contacts(self, user_id: str, contacts: list[dict], enrich_with_web: bool = True) -> list[dict]:
        imported = []
        for contact in contacts:
            profile = self._write_profile(contact, enrich_with_web)
            contact_id = self.store.add_contact(
                user_id, contact, profile, self.embedder.embed_passage(f"{contact['name']}. {profile}")
            )
            imported.append({"contact_id": contact_id, "name": contact["name"], "profile": profile})
        return imported

    def find_contacts(self, user_id: str, need: str, limit: int = 3) -> list[dict]:
        hits = self.store.search_contacts(
            user_id, self.embedder.embed_query(need), limit, config.CONTACT_MATCH_THRESHOLD
        )
        return [
            {k: h.get(k) for k in ("name", "phone", "email", "profile")} | {"score": round(h["score"], 3)}
            for h in hits
        ]

    def _write_profile(self, contact: dict, enrich_with_web: bool) -> str:
        facts = [f"Name: {contact['name']}"]
        if contact.get("notes"):
            facts.append(f"User's notes: {contact['notes']}")
        if contact.get("email"):
            facts.append(f"Email domain: {contact['email'].split('@')[-1]}")
        if enrich_with_web:
            query = " ".join(filter(None, [contact["name"], contact.get("notes")]))
            try:
                results = self.search(query, max_results=3)
                facts.append("Search results:\n" + "\n".join(f"- {r['title']}: {r['content']}" for r in results))
            except WebSearchUnavailable:
                pass
        return self.llm.complete(PROFILE_SYSTEM, "\n".join(facts), max_tokens=200)
