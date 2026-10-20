"""Minimal PayPal Orders v2 client (sandbox unless PAYPAL_ENV=live)."""

import time
import uuid

import httpx

from app import config


class PayPalError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


class PayPal:
    def __init__(
        self,
        client_id: str | None = config.PAYPAL_CLIENT_ID,
        secret: str | None = config.PAYPAL_CLIENT_SECRET,
        base_url: str = config.PAYPAL_BASE_URL,
        http: httpx.Client | None = None,
    ):
        if not (client_id and secret):
            raise PayPalError("PayPal isn't configured: set PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET.", 503)
        self.auth = (client_id, secret)
        self.http = http or httpx.Client(base_url=base_url, timeout=30)
        self._token, self._token_expiry = None, 0.0

    def create_order(self, items: list[dict], currency: str, total: str) -> dict:
        """items: [{"name", "amount"}], amounts already formatted as PayPal strings."""
        money = lambda value: {"currency_code": currency, "value": value}
        return self._post("/v2/checkout/orders", {
            "intent": "CAPTURE",
            "purchase_units": [{
                "amount": {**money(total), "breakdown": {"item_total": money(total)}},
                "items": [
                    {"name": i["name"][:127], "quantity": "1", "unit_amount": money(i["amount"])} for i in items
                ],
            }],
        }, request_id=uuid.uuid4().hex)

    def capture_order(self, order_id: str) -> dict:
        # Same request id on retries, so a repeated capture can't charge twice.
        return self._post(f"/v2/checkout/orders/{order_id}/capture", None, request_id=f"capture-{order_id}")

    def _access_token(self) -> str:
        if self._token and time.time() < self._token_expiry:
            return self._token
        data = self._send("POST", "/v1/oauth2/token", auth=self.auth, data={"grant_type": "client_credentials"})
        self._token = data["access_token"]
        self._token_expiry = time.time() + data.get("expires_in", 0) - 60
        return self._token

    def _post(self, path: str, body: dict | None, request_id: str) -> dict:
        headers = {
            "Authorization": f"Bearer {self._access_token()}",
            "Content-Type": "application/json",
            "PayPal-Request-Id": request_id,
        }
        return self._send("POST", path, json=body, headers=headers)

    def _send(self, method: str, path: str, **kwargs) -> dict:
        try:
            resp = self.http.request(method, path, **kwargs)
        except httpx.HTTPError as e:
            raise PayPalError(f"Couldn't reach PayPal: {e}") from e
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if resp.status_code >= 400:
            detail = data.get("message") or data.get("error_description") or resp.text[:200]
            issue = (data.get("details") or [{}])[0].get("issue")
            raise PayPalError(f"PayPal error {resp.status_code}: {detail}" + (f" ({issue})" if issue else ""))
        return data
