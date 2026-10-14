"""Main agent: talks to the user and delegates to the sub-agents.

Tools available to the main agent (matching the architecture sketch):
  - web_search            Tavily, for general questions
  - remember / get_profile  user storage
  - find_deals            deal-finder sub-agent (Tavily, location-aware)
  - find_contacts         contact sub-agent (vector search over profiled contacts)
"""

from app.agents.contact_agent import ContactAgent
from app.agents.deal_agent import DealAgent
from app.llm import LLM
from app.store import Store
from app.tools.web_search import WEB_SEARCH_SPEC, web_search

SYSTEM = """You are a PayPal shopping assistant. Users ask where to buy things.

For every shopping request:
1. Call get_profile to recall what you know about the user.
2. Call find_deals with the product (and the user's location if one is known).
3. Call find_contacts with a short description of the need, to see whether anyone in the
   user's contacts sells, repairs or recently bought that kind of item.
4. Answer with: the best deals (store, price, link) and any contacts who could help,
   saying briefly why each contact is relevant. Keep it short and practical.

When the user tells you something worth keeping (name, budget, preferences, city, sizes),
call remember. Never invent prices, stores or contacts; only use tool results."""


def _schema(props: dict, required: list[str]) -> dict:
    return {"json": {"type": "object", "properties": props, "required": required}}


class MainAgent:
    def __init__(self, llm: LLM, store: Store, deals: DealAgent, contacts: ContactAgent, search=web_search):
        self.llm = llm
        self.store = store
        self.deals = deals
        self.contacts = contacts
        self.search = search

    def chat(self, user_id: str, message: str, location: str | None = None, history: list | None = None) -> dict:
        text = message if not location else f"{message}\n\n(User shared location: {location})"
        messages = list(history or []) + [{"role": "user", "content": [{"text": text}]}]
        reply, trace = self.llm.run_tools(SYSTEM, messages, self._tools(user_id, location))
        return {"reply": reply, "tool_calls": trace, "history": messages}

    def _tools(self, user_id: str, location: str | None) -> dict:
        str_prop = {"type": "string"}
        return {
            "web_search": (WEB_SEARCH_SPEC, lambda a: self.search(a["query"])),
            "get_profile": (
                {"name": "get_profile", "description": "Get stored facts about the user.",
                 "inputSchema": _schema({}, [])},
                lambda a: self.store.get_profile(user_id),
            ),
            "remember": (
                {"name": "remember", "description": "Store a fact about the user, e.g. key='city'.",
                 "inputSchema": _schema({"key": str_prop, "value": str_prop}, ["key", "value"])},
                lambda a: self.store.remember(user_id, a["key"], a["value"]),
            ),
            "find_deals": (
                {"name": "find_deals",
                 "description": "Delegate to the deal-finder sub-agent: searches the web for discounted "
                                "prices on a product, near the location if given, otherwise online.",
                 "inputSchema": _schema({"product": str_prop, "location": str_prop}, ["product"])},
                lambda a: self.deals.find_deals(a["product"], a.get("location") or location),
            ),
            "find_contacts": (
                {"name": "find_contacts",
                 "description": "Delegate to the contact sub-agent: finds the user's contacts who sell, "
                                "know about or recently bought something related to the need.",
                 "inputSchema": _schema({"need": str_prop}, ["need"])},
                lambda a: self.contacts.find_contacts(user_id, a["need"]),
            ),
        }
