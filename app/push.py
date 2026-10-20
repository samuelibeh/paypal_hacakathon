"""Web Push notifications to every browser where the user turned notifications on."""

import json
import logging

from app import config

logger = logging.getLogger(__name__)


def _webpush(subscription: dict, data: str) -> None:
    from pywebpush import webpush

    webpush(
        subscription, data,
        vapid_private_key=config.VAPID_PRIVATE_KEY,
        vapid_claims={"sub": config.VAPID_SUBJECT},
        ttl=24 * 3600,  # keep it queued for a phone that's offline
    )


class Notifier:
    def __init__(self, store, send=None, enabled: bool | None = None):
        self.store = store
        self.send = send or _webpush
        self.enabled = configured() if enabled is None else enabled

    def notify(self, user_id: str, title: str, body: str) -> int:
        """Push to all the user's subscribed browsers. Returns how many accepted it."""
        if not self.enabled:
            return 0
        data = json.dumps({"title": title, "body": body, "url": "/"})
        delivered = 0
        for subscription in self.store.list_push_subscriptions(user_id):
            try:
                self.send(subscription, data)
                delivered += 1
            except Exception as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status in (404, 410):  # the browser unsubscribed
                    self.store.remove_push_subscription(subscription["endpoint"])
                else:
                    logger.warning("push to %s failed: %s", user_id, e)
        return delivered


def configured() -> bool:
    return bool(config.VAPID_PUBLIC_KEY and config.VAPID_PRIVATE_KEY and config.VAPID_SUBJECT)
