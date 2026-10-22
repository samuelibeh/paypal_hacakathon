import json

import httpx
import pytest

from app import checkout
from app.merchants import Merchant, Merchants
from app.paypal import PayPal, PayPalError
from tests.fakes import fake_store

SHOP = Merchant("shop", "Shop", ("shop.com",), "payee@shop.com", "help@shop.com", ("shop.com",))
MART = Merchant("mart", "Mart", ("mart.com",), "payee@mart.com", "help@mart.com", ("mart.com",))
MERCHANTS = Merchants([SHOP, MART])


class FakePayPalServer:
    """Answers the three PayPal endpoints we use and records what was sent."""

    def __init__(self, capture_status="COMPLETED"):
        self.capture_status = capture_status
        self.orders = []
        self.captures = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/oauth2/token":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        assert request.headers["authorization"] == "Bearer tok"
        if path == "/v2/checkout/orders":
            self.orders.append(json.loads(request.content))
            return httpx.Response(201, json={"id": "ORDER1", "status": "CREATED"})
        if path.endswith("/capture"):
            self.captures.append(request.headers["paypal-request-id"])
            return httpx.Response(201, json={
                "id": "ORDER1", "status": self.capture_status,
                "purchase_units": [{"payments": {"captures": [{"id": "CAP1"}]}}],
            })
        return httpx.Response(404, json={"message": "not found"})

    def client(self) -> PayPal:
        http = httpx.Client(base_url="https://paypal.test", transport=httpx.MockTransport(self.handler))
        return PayPal("id", "secret", http=http)


def _deal(**fields):
    return checkout.prepare_deal(fields, MERCHANTS)


def _stock(store, user_id="u1"):
    deals = [
        _deal(store="Shop", title="Fridge", price="$499.99", amount=499.99, currency="usd", url="https://www.shop.com/fridge"),
        _deal(store="Shop", title="Ice maker", price="$1,000", amount="1,000", currency="USD", url="https://shop.com/ice"),
        _deal(store="Shop", title="Fridge", price="₦450,000", amount=450000, currency="NGN", url="https://shop.com/ng"),
        _deal(store="Mart", title="Freezer", price="$300", amount=300, currency="USD", url="https://mart.com/f"),
        _deal(store="Elsewhere", title="Fridge", price="$450", amount=450, currency="USD", url="https://elsewhere.com/f"),
    ]
    return store.save_deals(user_id, deals)


def test_prepare_deal_normalises_price_and_flags_payable():
    shop_url = "https://shop.com/x"
    assert _deal(amount="1,299.5", currency="usd", url=shop_url) == {
        "amount": "1299.50", "currency": "USD", "url": shop_url,
        "merchant_id": "shop", "payable": True, "unpayable_reason": None,
    }
    assert "USD prices only" in _deal(amount=450000, currency="NGN", url=shop_url)["unpayable_reason"]
    assert "no price" in _deal(amount=None, currency="USD", url=shop_url)["unpayable_reason"]
    assert _deal(amount="-5", currency="USD", url=shop_url)["amount"] is None
    unknown = _deal(amount=10, currency="USD", url="https://notshop.com/x")
    assert unknown["payable"] is False and "set up to take PayPal" in unknown["unpayable_reason"]
    assert _deal(amount=10, currency="USD", url="https://evilshop.com/x")["merchant_id"] is None


def test_cart_only_accepts_payable_deals_of_this_user():
    store = fake_store()
    fridge, _, naira_fridge, _, elsewhere = _stock(store)

    cart = checkout.add_item(store, "u1", fridge["deal_id"])
    assert [i["title"] for i in cart["items"]] == ["Fridge"] and cart["total"] == "499.99"

    with pytest.raises(checkout.CheckoutError, match="USD prices only"):
        checkout.add_item(store, "u1", naira_fridge["deal_id"])
    with pytest.raises(checkout.CheckoutError, match="set up to take PayPal"):
        checkout.add_item(store, "u1", elsewhere["deal_id"])
    with pytest.raises(checkout.CheckoutError, match="No deal"):
        checkout.add_item(store, "someone-else", fridge["deal_id"])


def test_cart_holds_one_store_at_a_time():
    store = fake_store()
    fridge, _, _, freezer, _ = _stock(store)
    checkout.add_item(store, "u1", fridge["deal_id"])

    with pytest.raises(checkout.CheckoutError, match="Your cart has items from Shop"):
        checkout.add_item(store, "u1", freezer["deal_id"])

    store.remove_from_cart("u1", [fridge["deal_id"]])
    assert checkout.add_item(store, "u1", freezer["deal_id"])["total"] == "300.00"


def test_order_uses_stored_prices_and_capture_empties_cart():
    store, server = fake_store(), FakePayPalServer()
    paypal = server.client()
    fridge, ice_maker, *_ = _stock(store)
    checkout.add_item(store, "u1", fridge["deal_id"])
    checkout.add_item(store, "u1", ice_maker["deal_id"])

    order = checkout.create_order(store, paypal, "u1", MERCHANTS)

    assert order == {"id": "ORDER1", "total": "1499.99", "currency": "USD", "store": "Shop"}
    unit = server.orders[0]["purchase_units"][0]
    assert unit["payee"] == {"email_address": "payee@shop.com"}  # the money goes to the store
    assert unit["amount"]["value"] == "1499.99"
    assert [i["unit_amount"]["value"] for i in unit["items"]] == ["499.99", "1000.00"]

    with pytest.raises(checkout.CheckoutError, match="Order not found"):
        checkout.capture_order(store, paypal, "intruder", "ORDER1")

    result = checkout.capture_order(store, paypal, "u1", "ORDER1")

    assert result == {"status": "COMPLETED", "capture_id": "CAP1", "total": "1499.99", "currency": "USD"}
    assert server.captures == ["capture-ORDER1"]
    assert store.get_cart("u1") == []
    assert store.get_order("ORDER1")["status"] == "COMPLETED"


def test_failed_capture_keeps_the_cart():
    store, server = fake_store(), FakePayPalServer(capture_status="PENDING")
    paypal = server.client()
    fridge, *_ = _stock(store)
    checkout.add_item(store, "u1", fridge["deal_id"])
    checkout.create_order(store, paypal, "u1", MERCHANTS)

    assert checkout.capture_order(store, paypal, "u1", "ORDER1")["status"] == "PENDING"
    assert len(store.get_cart("u1")) == 1


def test_empty_cart_and_missing_credentials_are_reported():
    with pytest.raises(checkout.CheckoutError, match="empty"):
        checkout.create_order(fake_store(), FakePayPalServer().client(), "u1", MERCHANTS)
    with pytest.raises(PayPalError) as err:
        PayPal(None, None)
    assert err.value.status_code == 503
