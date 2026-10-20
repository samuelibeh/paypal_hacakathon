"""Sub-agent: profiles imported contacts and finds ones who can help with a purchase.

On import, each contact gets a short profile paragraph (what they do, sell or know
about). If web enrichment is on, the model first turns the contact into a targeted
search query, we run that one Tavily search ourselves, and the results go into the
prompt that writes the profile. The profile is embedded and stored in MongoDB;
searches embed the shopping need and return the contacts whose profiles are closest
by cosine similarity.
"""

from app import config
from app.embeddings import Embedder
from app.llm import LLM
from app.store import Store
from app.tools.web_search import WebSearchUnavailable, web_search

QUERY_SYSTEM = """You turn one of the user's contacts into a single web search query that
would find public information about this specific person: their job, their business or
what they sell. Put their full name in double quotes, then add the most distinguishing
details from the facts (business name, trade, city, employer, or a company email domain).
Keep it under 10 words. If the facts give nothing beyond a name to tell them apart from
strangers with the same name, reply NONE. Reply with only the query."""

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
            facts = _facts(contact)
            query = self._search_query(facts) if enrich_with_web else None
            if query:
                facts += self._search_results(query)
            profile = self.llm.complete(PROFILE_SYSTEM, "\n".join(facts), max_tokens=200)
            contact_id = self.store.add_contact(
                user_id, contact, profile, self.embedder.embed_passage(f"{contact['name']}. {profile}")
            )
            imported.append({
                "contact_id": contact_id, "name": contact["name"], "profile": profile, "search_query": query,
            })
        return imported

    def find_contacts(self, user_id: str, need: str, limit: int = 3) -> list[dict]:
        hits = self.store.search_contacts(
            user_id, self.embedder.embed_query(need), limit, config.CONTACT_MATCH_THRESHOLD
        )
        return [
            {k: h.get(k) for k in ("name", "phone", "email", "profile")} | {"score": round(h["score"], 3)}
            for h in hits
        ]

    def _search_query(self, facts: list[str]) -> str | None:
        """Ask the model for one search query, or None if the contact isn't identifiable."""
        lines = self.llm.complete(QUERY_SYSTEM, "\n".join(facts), max_tokens=40).strip().strip("`").splitlines()
        query = lines[0].strip() if lines else ""
        return None if not query or query.upper() == "NONE" else query

    def _search_results(self, query: str) -> list[str]:
        try:
            results = self.search(query, max_results=3)
        except WebSearchUnavailable:
            return []
        if not results:
            return []
        return ["Search results:\n" + "\n".join(f"- {r['title']}: {r['content']}" for r in results)]


def _facts(contact: dict) -> list[str]:
    facts = [f"Name: {contact['name']}"]
    if contact.get("notes"):
        facts.append(f"User's notes: {contact['notes']}")
    if contact.get("email"):
        facts.append(f"Email domain: {contact['email'].split('@')[-1]}")
    return facts
