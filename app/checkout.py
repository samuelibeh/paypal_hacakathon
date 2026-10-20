"""Cart and PayPal checkout.

Prices always come from the deals we stored when the agent found them, never from the
client, so the browser can't change what gets charged.
"""

from decimal import Decimal, InvalidOperation

from app import config

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


def prepare_deal(deal: dict) -> dict:
    """Normalise the deal agent's price fields and mark whether the deal can be paid in the app."""
    currency = str(deal.get("currency") or "").strip().upper() or None
    amount = money(deal.get("amount"), currency or "")
    return {
        **deal,
        "amount": amount,
        "currency": currency,
        "payable": amount is not None and currency == config.PAYPAL_CURRENCY,
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
    if not deal.get("payable"):
        raise CheckoutError(
            f"{deal.get('title') or 'This deal'} can't be paid in the app (checkout takes "
            f"{config.PAYPAL_CURRENCY} prices only). It can be bought on the store's site: {deal.get('url')}"
        )
    store.add_to_cart(user_id, deal_id)
    return cart_summary(store, user_id)


def create_order(store, paypal, user_id: str) -> dict:
    cart = cart_summary(store, user_id)
    if not cart["items"]:
        raise CheckoutError("Your cart is empty.")
    items = [
        {"name": f"{i.get('title') or 'Item'} ({i.get('store') or 'store'})", "amount": i["amount"]}
        for i in cart["items"]
    ]
    order = paypal.create_order(items, cart["currency"], cart["total"])
    store.save_order(user_id, order["id"], [i["deal_id"] for i in cart["items"]], cart["total"], cart["currency"])
    return {"id": order["id"], "total": cart["total"], "currency": cart["currency"]}


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
