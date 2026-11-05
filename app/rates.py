"""Exchange rates for checkout: what one unit of a currency costs in PAYPAL_CURRENCY.

A background thread (see main.py) refreshes them once a day from a free service that needs no
key. The last good set is kept in the database, so a restart or an outage carries on with it.
EXCHANGE_RATES in .env only fills in currencies neither of those has.
"""

import logging
from decimal import Decimal, InvalidOperation

import httpx

from app import config

logger = logging.getLogger(__name__)

RATES_URL = "https://open.er-api.com/v6/latest/{base}"
REFRESH_SECONDS = 24 * 3600
RETRY_SECONDS = 3600  # after a failed refresh

_live: dict[str, Decimal] = {}


def current() -> dict[str, Decimal]:
    return {**config.EXCHANGE_RATES, **_live}


def refresh(store) -> int:
    """Fetch today's rates and keep them. Raises if the service can't give a usable set, which
    leaves the rates in use as they are. Returns how many currencies came back."""
    resp = httpx.get(RATES_URL.format(base=config.PAYPAL_CURRENCY), timeout=20)
    resp.raise_for_status()
    fetched = _per_unit(resp.json())
    store.save_rates(config.PAYPAL_CURRENCY, {code: str(rate) for code, rate in fetched.items()})
    _live.clear()
    _live.update(fetched)
    return len(fetched)


def load_saved(store) -> None:
    """Start from the last rates fetched, so checkout works before the first refresh ends."""
    saved = store.load_rates(config.PAYPAL_CURRENCY)
    if saved:
        _live.update({code: Decimal(rate) for code, rate in saved.items()})


def _per_unit(data: dict) -> dict[str, Decimal]:
    """The service says how much of each currency one PAYPAL_CURRENCY buys ("1 USD = 63.5 PHP");
    checkout needs the opposite ("1 PHP = 0.0157 USD")."""
    if data.get("result") != "success" or data.get("base_code") != config.PAYPAL_CURRENCY:
        raise ValueError(f"unusable rates: {data.get('result')}, {data.get('base_code')}")
    rates = {}
    for code, per_base in (data.get("rates") or {}).items():
        try:
            value = Decimal(str(per_base))
        except InvalidOperation:
            continue
        if value.is_finite() and value > 0:
            rates[code.upper()] = 1 / value
    if not rates:
        raise ValueError("no rates came back")
    return rates
