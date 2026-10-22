"""Stores the app can pay directly with PayPal.

A deal can be paid in the app only when its link is on one of these stores' domains. The
PayPal order then names that store's PayPal account as the payee, so the money goes to the
store, not to us. Each entry also says which domains the store's receipts come from and
where the payment agent should write when no receipt arrives.

The list is a JSON file (MERCHANTS_FILE, see merchants.example.json).
"""

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from app import config


@dataclass(frozen=True)
class Merchant:
    id: str
    name: str
    domains: tuple[str, ...]
    paypal_email: str
    support_email: str
    receipt_senders: tuple[str, ...]

    def owns(self, url: str | None) -> bool:
        host = (urlparse(url or "").hostname or "").lower()
        return any(host == d or host.endswith("." + d) for d in self.domains)

    def to_dict(self) -> dict:
        return asdict(self)


class Merchants:
    def __init__(self, merchants: list[Merchant]):
        self._by_id = {m.id: m for m in merchants}

    @classmethod
    def from_file(cls, path: str) -> "Merchants":
        file = Path(path)
        return cls([_parse(e) for e in json.loads(file.read_text())] if file.exists() else [])

    def get(self, merchant_id: str | None) -> Merchant | None:
        return self._by_id.get(merchant_id)

    def for_url(self, url: str | None) -> Merchant | None:
        return next((m for m in self._by_id.values() if m.owns(url)), None)


def _parse(entry: dict) -> Merchant:
    domains = tuple(d.lower() for d in entry["domains"])
    return Merchant(
        id=entry["id"],
        name=entry["name"],
        domains=domains,
        paypal_email=entry["paypal_email"],
        support_email=entry["support_email"],
        receipt_senders=tuple(s.lower() for s in entry.get("receipt_senders") or domains),
    )


@lru_cache(maxsize=1)
def registry() -> Merchants:
    return Merchants.from_file(config.MERCHANTS_FILE)
