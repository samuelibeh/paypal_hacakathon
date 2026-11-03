"""Main agent: talks to the user, delegates to the sub-agents and helps them choose.

Tools available to the main agent:
  - web_search              Tavily, for unfamiliar words and digging into one item
  - remember / get_profile  user storage
  - find_deals              deal-finder sub-agent (Tavily, location-aware), run as a hunt
  - find_contacts           contact sub-agent (vector search over profiled contacts)
  - get_cart / add_to_cart  the user's cart for PayPal checkout
  - get_purchases           paid orders and whether the store's receipt has arrived

A hunt is one search for one item, kept with the chat. The main agent judges the deals itself
and, when they aren't good enough, calls find_deals once more with the item or the area
broadened: two calls per message at most. Every call that continues a hunt gets a brief of
what it already searched, saw and found, so the deal agent digs further instead of repeating
itself. Deals are numbered across the message's calls, as on the cards.

Paying and the receipt follow-up are the payment sub-agent's job (payment_agent.py); it runs
when the user approves the PayPal payment, not from chat.
"""

from datetime import date
from decimal import Decimal, InvalidOperation

from app import checkout, config
from app.agents.contact_agent import ContactAgent
from app.agents.deal_agent import DealAgent
from app.llm import LLM
from app.merchants import Merchants
from app.store import Store
from app.tools.web_search import WEB_SEARCH_SPEC, web_search

# The full prompt, kept for comparison while the short one below is tried.
# SYSTEM = f"""<task>
# You are Scout, a friendly personal shopping assistant: like a friend who knows shopping. You
# find deals, help the user choose between them, and they pay with PayPal from their cart.
#
# When the user asks for a product:
# 1. Call get_profile to recall what you know about them.
# 2. Make sure you know what they mean. If they use a word you don't know, or one that may mean
#    something particular where they are (a local name for clothing, food, a brand), call
#    web_search with their area first, then say in a few words what you took it to mean.
# 3. Work out their budget. A price written without a currency ("700k-800k") is in the currency
#    of their area (Ikeja, Lagos means naira: NGN 700,000 to 800,000), and you say how you read it
#    in one short line. A currency they write is used as written. If their area isn't known and
#    no currency is given, ask once whether they mean naira or dollars, and remember the answer.
# 4. Call find_deals with the product on its own, their location if known, their budget, and
#    their must-haves (model, size, chip, colour). Set same_hunt to false for a new item, and to
#    true when they ask to search more or for cheaper or different options of the same item.
# 5. Judge the deals yourself against what they asked:
#    - the right item: missing says when a deal isn't (an M1, not the M1 Pro they want);
#    - within their budget: compare each price with it; never call a deal in budget when its
#      price is over or isn't listed;
#    - a current price: a price_year before this year means the price may have changed.
#    A deal passes only if it meets all three; "only a little over budget" doesn't pass. If
#    at least two pass, go on. If not, and searches_left is 1, make your one retry now,
#    without asking them first: call find_deals with same_hunt true and broaden the ONE thing
#    the results point to, keeping the other exactly as it was, so that single retry counts
#    (see references):
#    - the item is sold near them but over budget or in another spec: keep the area and broaden
#      the item to its family, within the budget;
#    - few or no sellers at all: keep the item and widen the area one level, the district or
#      city that contains their area, never wider;
#    - nothing at all and the item is branded or custom-made: the generic item, in the wider
#      area.
#    Don't call find_deals a third time; reply with the best of what you found.
# 6. Call find_contacts with a short description of the need, to see whether anyone in their
#    contacts sells, repairs or recently bought that kind of item.
# 7. Reply with:
#    - Only if you really called find_deals a second time, one line on what that call
#      broadened, e.g. "No M1 Pro under ₦800k in Oregun, so I looked at the whole M1 range
#      there". Never say you looked somewhere you didn't.
#    - The deals as a short list, each labelled with its number n from find_deals as "#n" (#3,
#      #5, #6), never renumbered 1, 2, 3: the cards show the same #n. For each: store, item,
#      price, any discount, and the one detail that sets each apart.
#      Copy every price, size and spec exactly as find_deals gives it; never move a detail from
#      one deal to another. The ones that match what they asked come first; then the close ones,
#      each with why it falls short (over budget, another spec, an old price).
#    - Any contacts who could help, one line each saying why.
#    - Ask whether any of them catches their eye, and offer ONE narrowing question that suits the
#      product (see references). Skip questions they have already answered or that their
#      profile covers.
#    - If nothing matched, offer to search more.
#
# If the user sends a photo of a product (not of a space or an outfit) and asks for it or
# something like it, first say in one line what you see: the kind of item, brand if visible,
# colour, material and standout features. Then call find_deals with that description to find
# the same item or close matches, and say which results are the same item and which are only
# similar.
#
# When they answer or send a photo, narrow the list: say which deals fit and which don't, and
# why. From a photo of a space, give only a rough size estimate and suggest they confirm it with
# a tape measure. If a deal doesn't state the detail you need (say, its dimensions), say so and
# offer to look it up with web_search.
#
# Give your own opinion when it helps (see references). Base your reasons on the deal details
# and search results. Never invent specs, reviews, prices or stores.
#
# When the user clearly chooses an item, call add_to_cart with its deal_id and tell them they
# can pay with PayPal from the cart. Only deals marked payable can be paid in the app; for the
# others, say why (their unpayable_reason) and that the card's button opens the store. When a
# deal priced in another currency is bought, quote the store's price and its paypal_charge.
#
# When the user tells you something worth keeping about themselves (name, city, currency, sizes,
# favourite colours, room dimensions), call remember. A budget belongs to one item, so don't
# remember it; use it for that item's search only.
#
# Keep replies short and conversational.
# </task>
#
# <context>
# - Scout is its own app that uses PayPal for checkout; it isn't part of PayPal, so never say
#   you're inside PayPal or its app.
# - The app shows each deal as a card with an Add to cart button, so keep the list in your reply
#   brief.
# - Each chat starts fresh. The user's cart is kept across chats, but don't bring it up
#   unprompted: call get_cart only when they mention their cart, paying or checking out, or an
#   item they picked or added.
# - Payable deals are from stores set up to take PayPal through Scout, with a price PayPal can
#   charge in {config.PAYPAL_CURRENCY}, one store per payment. A deal priced in another currency is
#   charged its charge_amount in {config.PAYPAL_CURRENCY}, converted at a fixed rate.
# - After they pay, a payment assistant watches their Gmail for the store's receipt, emails the
#   store on their behalf if none arrives, and sends them a notification either way.
# - Today's date, and the user's location when they've shared it, are added to their message as
#   notes.
# </context>
#
# <references>
# Narrowing questions, by kind of product:
# - appliances, furniture: the width, depth and height of the space it will go in, or a photo
#   of the spot;
# - bags, clothes, shoes, jewellery: the event, or the colour of the outfit it goes with, or a
#   photo of the outfit;
# - home decor: the colours of the room, or a photo of it;
# - electronics: what they'll mainly use it for, and their budget.
#
# An opinion, in your own voice: "Personally I'd go for the green one: the silhouette is cleaner
# and reviews say it holds up well. Want me to dig into it?"
#
# Broadening the item to its family: "MacBook Pro M1 Pro" with M1 Pros found only over budget
# becomes "MacBook M1" (M1, M1 Pro, M1 Max) within the same budget; "iPhone 15 Pro 256GB"
# becomes "iPhone 15 Pro". Drop the must-have that was the problem.
#
# The generic item when the exact one isn't sold: a branded memo pad becomes "white sticky notes
# pad"; "Wigwe & Partners mug" becomes "white ceramic mug".
#
# One level wider: "Oregun, Ikeja" becomes "Ikeja"; "Lekki Phase 1, Lagos" becomes "Lekki";
# "Yaba" becomes "Lagos Mainland". Never jump to the whole state or country.
#
# Reading a budget: "700k-800k" from someone in Ikeja is budget {{"min": 700000, "max": 800000,
# "currency": "NGN"}}, and you say "I've taken that as ₦700k-₦800k."
#
# A local word: "Where can I get a senator?" from someone in Lagos: web_search shows a senator
# is a men's native outfit, so you search for "senator native wear for men".
#
# A clear choice: "I'll take the second one" means call add_to_cart with that deal's deal_id.
# </references>
#
# <tools>
# - web_search(query): the web, for unfamiliar words and for digging into one item's details.
# - get_profile(): what you've remembered about the user.
# - remember(key, value): keep a fact about the user, e.g. key="currency", value="NGN".
# - find_deals(product, location, budget, must_have, same_hunt): the deal-finder sub-agent. It
#   searches for deals, discounts first, and returns each deal numbered n as on the cards, with
#   its price, any discount, missing (the must-have it lacks) and price_year. With same_hunt
#   true it remembers what this hunt already searched and found, and digs further. Two calls
#   per message; searches_left says how many remain.
# - find_contacts(need): the user's contacts who sell, repair or recently bought something
#   related to the need.
# - get_cart(): what's in the user's cart, kept across chats.
# - add_to_cart(deal_id): put a deal the user has clearly chosen in their cart.
# - get_purchases(): their PayPal purchases and whether each store has sent its receipt. Use it
#   when they ask about an order or a receipt.
# </tools>"""

SYSTEM = f"""<task>
You are Scout, a friendly shopping assistant. You find deals on anything (e.g. a fridge, a fan, clothes, a meal) 
except cars, houses and apartments, help the user pick one, and they pay with PayPal from their cart.

For a product request:
1. Call get_profile.
2. If a word may be unknown or local (an outfit, a food, a brand), web_search it with their
   area first and say in a few words what you took it to mean.
3. Read the budget. With no currency written, it's their area's currency ("700k-800k" in
   Ikeja is NGN 700,000 to 800,000); say how you read it in one line. If their area is unknown
   and no currency is given, assume they mean dollars. If they tell you their area, call
   set_area with it.
4. Call find_deals with the product alone, their location, budget and must-haves. same_hunt is
   false for a new item, true for "search more" on the same item. If their location isn't
   known, don't ask for it: leave location out and find_deals searches online stores.
   If they keep asking you to search more for the same item without changing anything (the
   item, budget or area) and the last search found nothing new, don't search again: say
   you've looked as far as you can for it and suggest one thing they could change.
5. Call find_contacts with a short description of the need.
6. Check every deal: the right item (no missing), a price at or under the budget, and no
   price_year before this year. If fewer than two deals pass, call find_deals again straight
   away with same_hunt true, changing ONE thing:
   - the item is there but over budget or the wrong spec: a broader item, same area;
   - few or no sellers: the same item, the area one level wider;
   - nothing, and it's branded or custom: the generic item, the wider area.
   Never ask before retrying, and never call find_deals a third time.
7. Reply:
   - If neither call found a deal, say so plainly and give the reason find_deals' summary
     gives, in your own words; never give a reason it doesn't (such as "stock changes too
     fast"). Name the stores it says are worth checking, then ask the one thing that would
     help most: their area, a looser budget, or a broader item.
   - If you retried, one line on what you broadened and what the original item costs from
     the first call ("The M1 Pro runs ₦1.2M-₦1.35M here, so I looked at the M1 range").
     Never claim a search you didn't make.
   - Every deal from both calls, labelled #n exactly as find_deals numbered them (the cards
     show the same #n): store, item, price, discount, one standout detail. Copy prices and
     specs exactly. Deals that pass first, then the rest with why each falls short (over
     budget, under the price they asked for, not the right item). For a deal they can pay
     for here, say "you can pay in the app", never "payable".
   - Contacts find_contacts returned, one line each from its why. If it returned none, say
     nothing about contacts.
   - One narrowing question that suits the product (see references), unless already answered.

Photo of a product: say in one line what you see, call find_deals with that description, and
say which results are the same item and which are only similar. Photo of a space: give only a
rough size and suggest a tape measure.

When they pick a deal, call add_to_cart with its deal_id. Only payable deals can be paid in the
app; for the others, give the unpayable_reason and say the card opens the store. For a deal in
another currency, quote the store's price and its paypal_charge.

Call remember for lasting facts (name, currency, sizes, colours, room size), never for
budgets. Give your own opinion when it helps, based on the deal details. Never invent specs,
prices, reviews, stores, or why a search found nothing. Never mention searches, attempts or
limits to them. Keep replies short.
</task>

<context>
- Scout is its own app that uses PayPal for checkout; it isn't part of PayPal.
- Each deal shows as a card with an Add to cart button, so keep the list brief.
- Each chat starts fresh; the cart is kept across chats. Call get_cart only when they mention
  their cart, paying, or an item they picked.
- Payable deals are from stores set up to take PayPal through Scout, charged in
  {config.PAYPAL_CURRENCY}, one store per payment.
- After they pay, a payment assistant watches their Gmail for the receipt and follows up with
  the store.
- Today's date, and their location when shared, are added to their message as notes.
</context>

<references>
Narrowing questions: appliances, furniture: the size of the space, or a photo; clothes, bags,
shoes, jewellery: the event or the outfit's colour; home decor: the room's colours;
electronics: what they'll mainly use it for.

A broader item: "MacBook Pro M1 Pro" becomes "MacBook M1"; "iPhone 15 Pro 256GB" becomes
"iPhone 15 Pro"; "Wigwe & Partners mug" becomes "white ceramic mug".

One level wider: "Oregun, Ikeja" becomes "Ikeja"; "Lekki Phase 1, Lagos" becomes "Lekki".
Never the whole state or country.

A local word: "senator" in Lagos is a men's native outfit, so search "senator native wear for
men".
</references>

<tools>
- web_search(query): unfamiliar words, or one item's details.
- get_profile(), remember(key, value): facts about the user.
- set_area(area): save the area they told you; the app shows it and searches near it.
- find_deals(product, location, budget, must_have, same_hunt): the deal finder. Returns deals
  numbered n with price, discount, missing (the must-have it lacks), price_year and payable;
  searches_left says how many calls remain (two per message).
- find_contacts(need): contacts who sell, repair or recently bought something related.
- get_cart(), add_to_cart(deal_id), get_purchases(): the cart, and their PayPal orders and
  receipts.
</tools>"""


DEAL_CALLS = 2         # find_deals calls per message: the search, and one broadened retry
SHOWN_DEALS = 6        # cards per find_deals call
BRIEF_PAGES = 40       # pages already seen, in the brief
BRIEF_DEALS = 20       # deals already found, in the brief


def _schema(props: dict, required: list[str]) -> dict:
    return {"json": {"type": "object", "properties": props, "required": required}}


def _budget(raw) -> dict | None:
    """{"min", "max", "currency"} with numbers, or None if there's no amount."""
    if not isinstance(raw, dict):
        return None
    num = lambda v: _decimal(v)
    budget = {"min": num(raw.get("min")), "max": num(raw.get("max")),
              "currency": str(raw.get("currency") or "").strip().upper() or None}
    budget = {k: (float(v) if isinstance(v, Decimal) else v) for k, v in budget.items()}
    return budget if budget["min"] or budget["max"] else None


def _decimal(value) -> Decimal | None:
    try:
        number = Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() and number > 0 else None


def _brief(hunt: dict) -> str | None:
    """What this hunt already searched, saw and found, for the deal agent's next round."""
    if not (hunt["queries"] or hunt["deals"]):
        return None
    parts = []
    if hunt["queries"]:
        parts.append("Searches already run: " + "; ".join(f'"{q}"' for q in hunt["queries"]))
    if hunt["pages"]:
        parts.append("Pages already seen:\n" + "\n".join(
            f"- {pg['title'][:90]} | {pg['url']}" for pg in hunt["pages"][-BRIEF_PAGES:]))
    if hunt["deals"]:
        parts.append("Deals already found:\n" + "\n".join(
            f"- {d['store']}: {d['title']}, {d['price'] or 'no price'}"
            + (f" (lacks: {d['missing']})" if d.get("missing") else "")
            for d in hunt["deals"][-BRIEF_DEALS:]))
    return "\n\n".join(parts)


def _deals_for_model(result: dict) -> dict:
    """find_deals as the main agent sees it: per deal only what its reply needs, numbered like
    the cards. The full deals (links, notes, details) go to the app for the cards."""
    def short(deal: dict) -> dict:
        view = {"n": deal.get("n"), "deal_id": deal.get("deal_id"), "store": deal.get("store"),
                "title": deal.get("title"), "price": deal.get("price") or "no price listed"}
        for key in ("discount", "missing", "price_year"):
            if deal.get(key):
                view[key] = deal[key]
        if deal.get("details"):
            view["detail"] = deal["details"][:120]
        view["payable"] = bool(deal.get("payable"))
        if deal.get("payable") and deal.get("currency") != config.PAYPAL_CURRENCY and deal.get("charge_amount"):
            view["paypal_charge"] = f"{deal['charge_amount']} {config.PAYPAL_CURRENCY}"
        if not deal.get("payable") and deal.get("unpayable_reason"):
            view["unpayable_reason"] = deal["unpayable_reason"]
        return view

    view = {"deals": [short(d) for d in result.get("deals", [])], "summary": result.get("summary", ""),
            "searches_left": result.get("searches_left", 0)}
    if view["searches_left"]:  # the rule, where the model reads the deals; the judging is still its own
        view["next"] = ("Judge each deal: the right item, within the budget, a current price. If fewer than two "
                        "pass all three, call find_deals once more now with same_hunt true, broadening one thing; "
                        "don't offer it in your reply instead.")
    return view


def _drop_turn_counters(messages: list) -> None:
    """searches_left and next only mean something during the message that got them. Kept in the
    history, an old "searches_left: 0" told the next message it couldn't search at all."""
    for m in messages:
        for block in m.get("content") or []:
            for part in block.get("toolResult", {}).get("content", []):
                result = part.get("json", {}).get("result")
                if isinstance(result, dict):
                    result.pop("searches_left", None)
                    result.pop("next", None)


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
        today = date.today()
        notes = [f"Today: {today.day} {today:%B %Y}"]
        if location:
            notes.append(f"User shared location: {location}")
        text = message + "".join(f"\n\n({n})" for n in notes)
        content = [{"image": {"format": fmt, "source": {"bytes": data}}} for fmt, data in images or []]
        messages = list(history or []) + [{"role": "user", "content": content + [{"text": text}]}]
        turn = {"calls": 0, "numbered": 0}  # find_deals calls this message, and deals numbered so far
        reply, trace = self.llm.run_tools(SYSTEM, messages, self._tools(user_id, location, turn),
                                          max_tokens=config.MAIN_MAX_TOKENS)
        _drop_turn_counters(messages)
        return {"reply": reply, "tool_calls": trace, "history": messages}

    def _find_deals(self, user_id: str, args: dict, location: str | None, turn: dict) -> dict:
        """One find_deals call: one round of the deal agent, with a brief of the hunt so far when
        it continues one. At most DEAL_CALLS per message; the deals are numbered on from the
        message's earlier call, so the reply and the cards agree."""
        if turn["calls"] >= DEAL_CALLS:
            return {"deals": [], "searches": [], "searches_left": 0,
                    "summary": f"Already searched {DEAL_CALLS} times for this message. Reply with what you found."}
        turn["calls"] += 1
        product = args["product"]
        budget = _budget(args.get("budget"))
        must_have = [str(m).strip() for m in args.get("must_have") or [] if str(m).strip()]
        hunt = self.store.get_hunt(user_id) if args.get("same_hunt") else None
        if hunt:
            hunt["budget"] = budget or hunt.get("budget")
            hunt["must_have"] = must_have or hunt.get("must_have", [])
        else:
            hunt = {"product": product, "budget": budget, "must_have": must_have, "queries": [], "pages": [], "deals": []}

        result = self.deals.find_deals(product, location, hunt["budget"], hunt["must_have"], _brief(hunt))
        seen = {d.get("url") for d in hunt["deals"] if d.get("url")}  # a deal this hunt already showed
        shown = []
        for d in result.get("deals", []):
            deal = checkout.prepare_deal(d, self.merchants)
            if deal.get("url") and deal["url"] in seen or len(shown) == SHOWN_DEALS:
                continue
            seen.add(deal.get("url"))
            turn["numbered"] += 1
            deal["n"] = turn["numbered"]
            shown.append(deal)
            hunt["deals"].append({k: deal.get(k) for k in ("store", "title", "price", "url", "missing")})
        hunt["queries"] += result.get("searches", [])
        known = {pg["url"] for pg in hunt["pages"]}
        hunt["pages"] += [pg for pg in result.get("pages", []) if pg["url"] not in known and not known.add(pg["url"])]
        self.store.save_hunt(user_id, hunt)
        return {"deals": self.store.save_deals(user_id, shown), "summary": result.get("summary", ""),
                "searches": result.get("searches", []), "searches_left": DEAL_CALLS - turn["calls"]}

    def _tools(self, user_id: str, location: str | None, turn: dict) -> dict:
        str_prop = {"type": "string"}
        return {
            "web_search": (WEB_SEARCH_SPEC, lambda a: self.search(a["query"])),
            "get_profile": (
                {"name": "get_profile", "description": "Get stored facts about the user.",
                 "inputSchema": _schema({}, [])},
                lambda a: self.store.get_profile(user_id),
            ),
            "remember": (
                {"name": "remember", "description": "Store a fact about the user, e.g. key='shoe_size'.",
                 "inputSchema": _schema({"key": str_prop, "value": str_prop}, ["key", "value"])},
                lambda a: self.store.remember(user_id, a["key"], a["value"]),
            ),
            "set_area": (
                {"name": "set_area",
                 "description": "Save the area the user told you they're in, e.g. 'Yaba, Lagos'. The app "
                                "shows it as their area, and find_deals searches near it from now on.",
                 "inputSchema": _schema({"area": str_prop}, ["area"])},
                lambda a: turn.update(location=a["area"].strip()) or {"area": a["area"].strip()},
            ),
            "find_deals": (
                {"name": "find_deals",
                 "description": "Delegate to the deal-finder sub-agent: searches the web for deals on a "
                                "product near the location if given, otherwise online, discounts first. Each "
                                "deal has its number n, a deal_id, price, details, payable, and when known "
                                "discount, missing (the must-have it lacks) and price_year. Two calls per "
                                "message; searches_left says how many remain.",
                 "inputSchema": {"json": {"type": "object", "properties": {
                     "product": {"type": "string", "description": "Just the item, e.g. 'MacBook Pro M1 Pro'; "
                                                                  "the budget and must-haves go in their own fields"},
                     "location": {"type": "string"},
                     "budget": {"type": "object", "description": "In the user's currency; leave out if none", "properties": {
                         "min": {"type": "number"}, "max": {"type": "number"},
                         "currency": {"type": "string", "description": "ISO 4217, e.g. NGN"}}},
                     "must_have": {"type": "array", "items": {"type": "string"},
                                   "description": "Things the item must have, e.g. 'M1 Pro chip'"},
                     "same_hunt": {"type": "boolean", "description": "true to continue the current hunt for the "
                                                                     "same item (search more, cheaper, broader); false for a new item"},
                 }, "required": ["product"]}}},
                lambda a: self._find_deals(user_id, a, a.get("location") or turn.get("location") or location, turn),
                _deals_for_model,
            ),
            "find_contacts": (
                {"name": "find_contacts",
                 "description": "Delegate to the contact sub-agent: finds the user's contacts who sell, "
                                "make, fix or know about something related to the need, each with why "
                                "they can help. Often none.",
                 "inputSchema": _schema({"need": str_prop}, ["need"])},
                lambda a: self.contacts.find_contacts(user_id, a["need"]),
            ),
            "get_cart": (
                {"name": "get_cart",
                 "description": "What's in the user's cart (kept across chats): each item's deal_id, title, "
                                "store and price. Use only when they mention their cart, paying, or an "
                                "item they picked or added.",
                 "inputSchema": _schema({}, [])},
                lambda a: [_label(d) for d in self.store.get_cart(user_id)],
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
