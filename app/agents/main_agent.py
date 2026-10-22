"""Main agent: talks to the user, delegates to the sub-agents and helps them choose.

Tools available to the main agent:
  - web_search              Tavily, for general questions and digging into one item
  - remember / get_profile  user storage
  - find_deals              deal-finder sub-agent (Tavily, location-aware)
  - find_contacts           contact sub-agent (vector search over profiled contacts)
  - add_to_cart             put a deal the user chose into their cart for PayPal checkout
  - get_purchases           paid orders and whether the store's receipt has arrived

Paying and the receipt follow-up are the payment sub-agent's job (payment_agent.py); it runs
when the user approves the PayPal payment, not from chat.
"""

from app import checkout, config
from app.agents.contact_agent import ContactAgent
from app.agents.deal_agent import DealAgent
from app.llm import LLM
from app.merchants import Merchants
from app.store import Store
from app.tools.web_search import WEB_SEARCH_SPEC, web_search

SYSTEM = f"""You are a friendly personal shopper inside a PayPal shopping app. You find deals,
help the user choose between them, and they pay with PayPal from their cart.

When the user asks for a product:
1. Call get_profile to recall what you know about them.
2. Call find_deals with the product (and their location if known).
3. Call find_contacts with a short description of the need, to see whether anyone in their
   contacts sells, repairs or recently bought that kind of item.
4. Reply with:
   - The deals as a short numbered list in the order find_deals returned them: store, item,
     price, and the one detail that sets each apart. The app shows each deal as a card with
     a Select button, so keep the list brief.
   - Any contacts who could help, one line each saying why.
   - Ask whether any of them catches their eye, and offer ONE narrowing question that suits
     the product. For example:
       appliances, furniture: the width, depth and height of the space it will go in, or a
         photo of the spot;
       bags, clothes, shoes, jewellery: the event, or the colour of the outfit it goes with,
         or a photo of the outfit;
       home decor: the colours of the room, or a photo of it;
       electronics: what they'll mainly use it for, and their budget.
     Skip questions they have already answered or that their profile covers.

When they answer or send a photo, narrow the list: say which deals fit and which don't, and
why. From a photo of a space, give only a rough size estimate and suggest they confirm it
with a tape measure. If a deal doesn't state the detail you need (say, its dimensions), say
so and offer to look it up with web_search.

Give your own opinion when it helps, like a friend who knows shopping, e.g. "Personally I'd go
for the green one: the silhouette is cleaner and reviews say it holds up well. Want me to dig
into it?" Base your reasons on the deal details and search results. Never invent specs,
reviews, prices or stores.

A user message may end with a note listing items they selected in the app; treat those as the
ones they're interested in. When the user clearly chooses an item ("I'll take the second
one"), call add_to_cart with its deal_id and tell them they can pay with PayPal from the cart.
Only deals marked payable can be paid in the app: stores set up to take PayPal through it,
priced in {config.PAYPAL_CURRENCY}, one store per payment. For the others, give the store link
and the deal's unpayable_reason.

After they pay, a payment assistant watches their Gmail for the store's receipt, emails the
store on their behalf if none arrives, and sends them a notification either way. When they ask
about an order or a receipt, call get_purchases.

When the user tells you something worth keeping (name, budget, city, sizes, favourite colours,
room dimensions), call remember.

Keep replies short and conversational."""


def _schema(props: dict, required: list[str]) -> dict:
    return {"json": {"type": "object", "properties": props, "required": required}}


def _label(deal: dict) -> str:
    return f"[{deal['deal_id']}] {deal.get('title') or 'item'} from {deal.get('store')}, {deal.get('price')}"


class MainAgent:
    def __init__(
        self, llm: LLM, store: Store, deals: DealAgent, contacts: ContactAgent, search=web_search,
        merchants: Merchants | None = None,
    ):
        self.llm = llm
        self.store = store
        self.deals = deals
        self.contacts = contacts
        self.search = search
        self.merchants = merchants

    def chat(
        self,
        user_id: str,
        message: str,
        location: str | None = None,
        history: list | None = None,
        images: list[tuple[str, bytes]] | None = None,
    ) -> dict:
        """images: (format, bytes) pairs, format one of png, jpeg, gif, webp."""
        notes = []
        if location:
            notes.append(f"User shared location: {location}")
        cart = self.store.get_cart(user_id)
        if cart:
            notes.append("Selected in the app: " + "; ".join(_label(d) for d in cart))
        text = message + "".join(f"\n\n({n})" for n in notes)
        content = [{"image": {"format": fmt, "source": {"bytes": data}}} for fmt, data in images or []]
        messages = list(history or []) + [{"role": "user", "content": content + [{"text": text}]}]
        reply, trace = self.llm.run_tools(SYSTEM, messages, self._tools(user_id, location))
        return {"reply": reply, "tool_calls": trace, "history": messages}

    def _find_deals(self, user_id: str, product: str, location: str | None) -> dict:
        result = self.deals.find_deals(product, location)
        prepared = [checkout.prepare_deal(d, self.merchants) for d in result.get("deals", [])]
        return {**result, "deals": self.store.save_deals(user_id, prepared)}

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
                                "prices on a product, near the location if given, otherwise online. "
                                "Each deal has a deal_id, details, and payable (can be paid in the app).",
                 "inputSchema": _schema({"product": str_prop, "location": str_prop}, ["product"])},
                lambda a: self._find_deals(user_id, a["product"], a.get("location") or location),
            ),
            "find_contacts": (
                {"name": "find_contacts",
                 "description": "Delegate to the contact sub-agent: finds the user's contacts who sell, "
                                "know about or recently bought something related to the need.",
                 "inputSchema": _schema({"need": str_prop}, ["need"])},
                lambda a: self.contacts.find_contacts(user_id, a["need"]),
            ),
            "add_to_cart": (
                {"name": "add_to_cart",
                 "description": "Add a deal the user has clearly chosen to their cart so they can pay "
                                "with PayPal. Use the deal_id from find_deals.",
                 "inputSchema": _schema({"deal_id": str_prop}, ["deal_id"])},
                lambda a: checkout.add_item(self.store, user_id, a["deal_id"]),
            ),
            "get_purchases": (
                {"name": "get_purchases",
                 "description": "The user's PayPal purchases, newest first, with whether each store has "
                                "sent its receipt (status: awaiting_receipt, receipt_received, followed_up "
                                "when we emailed the store, unconfirmed when it never replied).",
                 "inputSchema": _schema({}, [])},
                lambda a: self.store.list_purchases(user_id),
            ),
        }
