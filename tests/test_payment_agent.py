import json

from app import checkout
from app.agents.payment_agent import PaymentAgent
from app.gmail import GmailNotConnected
from app.llm import LLM
from app.push import Notifier
from tests.fakes import FakeBedrock, fake_store, text_reply
from tests.test_checkout import MERCHANTS, FakePayPalServer, _stock

DAY = 24 * 3600


class FakeGmail:
    def __init__(self, connected=True):
        self.connected = connected
        self.inbox, self.queries, self.sent = [], [], []

    def connected_email(self, user_id):
        return "sam@gmail.com" if self.connected else None

    def search(self, user_id, query, limit=10):
        if not self.connected:
            raise GmailNotConnected("Gmail isn't connected for this user.")
        self.queries.append(query)
        return list(self.inbox)

    def send(self, user_id, to, subject, body):
        if not self.connected:
            raise GmailNotConnected("Gmail isn't connected for this user.")
        self.sent.append({"to": to, "subject": subject, "body": body})
        return "sent-1"


def email(id, sender, subject, text=""):
    return {"id": id, "from": sender, "subject": subject, "date": "Tue, 20 Oct 2026", "text": text}


class Setup:
    """A user who has approved a $499.99 PayPal payment to Shop, and a payment agent."""

    def __init__(self, connected=True):
        self.now = 1_800_000_000.0
        self.store = fake_store()
        self.bedrock = FakeBedrock()
        self.gmail = FakeGmail(connected)
        self.pushed = []
        self.store.add_push_subscription("u1", {"endpoint": "https://push.test/1", "keys": {"p256dh": "k", "auth": "a"}})
        notifier = Notifier(self.store, send=lambda sub, data: self.pushed.append(json.loads(data)), enabled=True)
        self.agent = PaymentAgent(LLM(self.bedrock, "m"), self.store, self.gmail, notifier,
                                  wait_hours=24, poll_seconds=300, clock=lambda: self.now)
        self.paypal = FakePayPalServer().client()
        fridge, *_ = _stock(self.store)
        checkout.add_item(self.store, "u1", fridge["deal_id"])
        checkout.create_order(self.store, self.paypal, "u1", MERCHANTS)

    def pay(self):
        return self.agent.pay(self.paypal, "u1", "ORDER1")

    def later(self, seconds=300):
        self.now += seconds
        return self.agent.check_due()

    def purchase(self):
        return self.store.get_purchase("ORDER1")


def test_paying_tells_the_user_once_and_starts_watching_for_the_receipt():
    s = Setup()

    result = s.pay()
    s.pay()  # a retried capture

    assert result["status"] == "COMPLETED" and result["store"] == "Shop" and result["watching_receipt"]
    assert s.purchase()["status"] == "awaiting_receipt"
    assert s.purchase()["merchant"]["support_email"] == "help@shop.com"
    assert [p["title"] for p in s.pushed] == ["Payment sent to Shop"]
    assert "$499.99" in s.pushed[0]["body"] and "sam@gmail.com" in s.pushed[0]["body"]


def test_a_matching_store_email_completes_the_purchase():
    s = Setup()
    s.pay()
    s.gmail.inbox = [
        email("m1", "deals@shop.com", "Big sale this weekend"),
        email("m2", "orders@shop.com", "Your order is confirmed", "Fridge $499.99, PayPal CAP1"),
    ]
    s.bedrock.responses.append(text_reply('{"receipt": 2, "reason": "Same item and amount."}'))

    assert s.later() == 1

    assert s.purchase()["status"] == "receipt_received"
    assert s.purchase()["receipt"]["subject"] == "Your order is confirmed"
    assert s.pushed[-1] == {"title": "Payment complete", "body": "Shop sent your receipt for $499.99: “Your order is confirmed”.", "url": "/"}
    assert "from:shop.com" in s.gmail.queries[0] and "after:" in s.gmail.queries[0]
    judged = s.bedrock.requests[0]["messages"][0]["content"][0]["text"]
    assert "PayPal transaction ID: CAP1" in judged and "Big sale" in judged


def test_no_receipt_by_the_deadline_emails_the_store_as_the_user_then_keeps_watching():
    s = Setup()
    s.store.remember("u1", "name", "Sam")
    s.pay()
    s.gmail.inbox = [email("m1", "deals@shop.com", "Big sale this weekend")]
    s.bedrock.responses.append(text_reply('{"receipt": null, "reason": "Only an advert."}'))

    s.later()
    s.later()  # nothing new: the advert isn't judged again
    assert len(s.bedrock.requests) == 1 and s.purchase()["status"] == "awaiting_receipt"

    s.bedrock.responses.append(text_reply("Hello Shop team, I paid $499.99 with PayPal... Thanks, Sam"))
    s.later(DAY)

    assert s.gmail.sent == [{
        "to": "help@shop.com",
        "subject": "Order confirmation for my PayPal payment of $499.99",
        "body": "Hello Shop team, I paid $499.99 with PayPal... Thanks, Sam",
    }]
    asked = s.bedrock.requests[1]
    assert "own voice (first person)" in asked["system"][0]["text"]
    assert "Shopper's name: Sam" in asked["messages"][0]["content"][0]["text"]
    assert s.purchase()["status"] == "followed_up"
    assert s.pushed[-1]["title"] == "No receipt from Shop yet" and "help@shop.com" in s.pushed[-1]["body"]

    s.gmail.inbox.append(email("m2", "help@shop.com", "Re: Order confirmation", "We have your order."))
    s.bedrock.responses.append(text_reply('{"receipt": 1, "reason": "The store confirms the order."}'))
    s.later()

    assert s.purchase()["status"] == "receipt_received"
    assert s.pushed[-1]["title"] == "Payment complete"


def test_the_user_is_told_to_contact_the_store_when_the_follow_up_gets_no_reply():
    s = Setup()
    s.pay()
    s.bedrock.responses.append(text_reply("Hello Shop team..."))
    s.later(DAY)
    assert s.purchase()["status"] == "followed_up"

    s.later(DAY)

    assert s.purchase()["status"] == "unconfirmed"
    assert s.pushed[-1]["title"] == "Still no receipt from Shop"
    assert "help@shop.com" in s.pushed[-1]["body"] and "CAP1" in s.pushed[-1]["body"]
    assert s.later(DAY) == 0  # it stops checking


def test_without_gmail_the_user_is_told_and_nothing_is_sent():
    s = Setup(connected=False)
    s.pay()
    assert "Connect Gmail" in s.pushed[0]["body"]

    s.later(300)
    assert s.bedrock.requests == []

    s.bedrock.responses.append(text_reply("Hello Shop team..."))
    s.later(DAY)

    assert s.gmail.sent == [] and s.purchase()["status"] == "unconfirmed"
    assert s.pushed[-1]["title"] == "No receipt from Shop"
    assert "Gmail isn't connected" in s.pushed[-1]["body"] and "help@shop.com" in s.pushed[-1]["body"]


def test_follow_up_uses_a_template_when_claude_is_unavailable():
    s = Setup()
    s.pay()
    s.later(DAY)  # no scripted reply, so the Claude call fails

    body = s.gmail.sent[0]["body"]
    assert body.startswith("Hello Shop team,") and "- Fridge ($499.99)" in body and "CAP1" in body


def test_expired_push_subscriptions_are_dropped():
    store = fake_store()
    store.add_push_subscription("u1", {"endpoint": "https://push.test/gone", "keys": {}})
    store.add_push_subscription("u1", {"endpoint": "https://push.test/ok", "keys": {}})

    class Gone(Exception):
        response = type("R", (), {"status_code": 410})()

    def send(sub, data):
        if sub["endpoint"].endswith("gone"):
            raise Gone()

    assert Notifier(store, send=send, enabled=True).notify("u1", "t", "b") == 1
    assert [s["endpoint"] for s in store.list_push_subscriptions("u1")] == ["https://push.test/ok"]
