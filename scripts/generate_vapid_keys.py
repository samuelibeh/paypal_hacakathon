"""Print a new VAPID key pair for Web Push, as lines to paste into .env.

    python -m scripts.generate_vapid_keys
"""

import base64

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def main():
    key = ec.generate_private_key(ec.SECP256R1())
    private = key.private_numbers().private_value.to_bytes(32, "big")
    public = key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    print(f"VAPID_PUBLIC_KEY={b64url(public)}")
    print(f"VAPID_PRIVATE_KEY={b64url(private)}")
    print("VAPID_SUBJECT=mailto:you@example.com")


if __name__ == "__main__":
    main()
