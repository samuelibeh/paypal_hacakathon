"""Cart and PayPal checkout.

Prices always come from the deals we stored when the agent found them, never from the
client, so the browser can't change what gets charged. The PayPal order names the store's
own PayPal account as the payee, so paying sends the money to the store. A cart holds items
from one store at a time, since each payment goes to one payee.
"""

from decimal import Decimal, InvalidOperation

from app import config
from app.merchants import Merchants, registry

ZERO_DECIMAL_CURRENCIES = {"HUF", "JPY", "TWD"}  # PayPal rejects decimals for these


class CheckoutError(ValueError):
    """A cart or checkout problem the user can act on; the message is shown to them."""


def money(value, currency: str) -> str | None:
    """Format a positive amount the way PayPal expects ("499.00"), or None if it isn't one."""
    try:
        amount = Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None
    if not amount.is_finite() or amount <= 0:
        return None
    step = Decimal("1") if currency in ZERO_DECIMAL_CURRENCIES else Decimal("0.01")
    return str(amount.quantize(step))


def prepare_deal(deal: dict, merchants: Merchants | None = None) -> dict:
    """Normalise the deal agent's price fields and mark whether the deal can be paid in the app."""
    merchant = (merchants or registry()).for_url(deal.get("url"))
    currency = str(deal.get("currency") or "").strip().upper() or None
    amount = money(deal.get("amount"), currency or "")
    if amount is None:
        reason = "it has no price we can charge"
    elif not merchant:
        reason = "the app can only pay stores that are set up to take PayPal through it"
    elif currency != config.PAYPAL_CURRENCY:
        reason = f"checkout takes {config.PAYPAL_CURRENCY} prices only"
    else:
        reason = None
    return {
        **deal,
        "amount": amount,
        "currency": currency,
        "merchant_id": merchant.id if merchant else None,
        "payable": reason is None,
        "unpayable_reason": reason,
    }


def cart_summary(store, user_id: str) -> dict:
    items = store.get_cart(user_id)
    total = sum((Decimal(i["amount"]) for i in items), Decimal(0))
    return {
        "items": items,
        "total": money(total, config.PAYPAL_CURRENCY) or "0.00",
        "currency": config.PAYPAL_CURRENCY,
    }


def add_item(store, user_id: str, deal_id: str) -> dict:
    deal = store.get_deal(user_id, deal_id)
    if not deal:
        raise CheckoutError(f"No deal with id {deal_id}.")
    title = deal.get("title") or "This deal"
    if not deal.get("payable"):
        reason = deal.get("unpayable_reason") or f"checkout takes {config.PAYPAL_CURRENCY} prices only"
        raise CheckoutError(f"{title} can't be paid in the app ({reason}). It can be bought on the store's site: {deal.get('url')}")
    other = next((i for i in store.get_cart(user_id) if i.get("merchant_id") != deal["merchant_id"]), None)
    if other:
        raise CheckoutError(
            f"Your cart has items from {other.get('store')}. Each PayPal payment goes to one store, "
            f"so pay for those or remove them before adding {title}."
        )
    store.add_to_cart(user_id, deal_id)
    return cart_summary(store, user_id)


def create_order(store, paypal, user_id: str, merchants: Merchants | None = None) -> dict:
    cart = cart_summary(store, user_id)
    if not cart["items"]:
        raise CheckoutError("Your cart is empty.")
    merchant_ids = {i.get("merchant_id") for i in cart["items"]}
    merchant = (merchants or registry()).get(merchant_ids.pop()) if len(merchant_ids) == 1 else None
    if not merchant:
        raise CheckoutError("Your cart has items the app can't pay for together. Remove them and add them again.")
    items = [{"name": i.get("title") or "Item", "amount": i["amount"]} for i in cart["items"]]
    order = paypal.create_order(items, cart["currency"], cart["total"], payee_email=merchant.paypal_email)
    store.save_order(
        user_id, order["id"], [i["deal_id"] for i in cart["items"]], cart["total"], cart["currency"],
        merchant=merchant.to_dict(), items=[{"title": i["name"], "amount": i["amount"]} for i in items],
    )
    return {"id": order["id"], "total": cart["total"], "currency": cart["currency"], "store": merchant.name}


def capture_order(store, paypal, user_id: str, order_id: str) -> dict:
    order = store.get_order(order_id)
    if not order or order["user_id"] != user_id:
        raise CheckoutError("Order not found.")
    result = paypal.capture_order(order_id)
    status = result.get("status")
    payments = ((result.get("purchase_units") or [{}])[0]).get("payments") or {}
    capture_id = ((payments.get("captures") or [{}])[0]).get("id")
    store.update_order(order_id, status=status, capture_id=capture_id)
    if status == "COMPLETED":
        store.remove_from_cart(user_id, order["deal_ids"])
    return {"status": status, "capture_id": capture_id, "total": order["total"], "currency": order["currency"]}
