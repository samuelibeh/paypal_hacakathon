"""Settings loaded from environment / .env."""

import os
import secrets
from decimal import Decimal

from dotenv import load_dotenv

load_dotenv()

# LLM (Claude on AWS Bedrock; boto3 picks up AWS_BEARER_TOKEN_BEDROCK automatically)
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "arn:aws:bedrock:us-east-1:022677638266:inference-profile/"
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
)
# The main agent (it judges the deals and decides on a retry) can run a different model from
# the sub-agents, e.g. global.anthropic.claude-haiku-5-5. Unset: the same model as the rest.
MAIN_MODEL_ID = os.getenv("MAIN_MODEL_ID") or BEDROCK_MODEL_ID
# Newer models think before answering, and thinking counts toward this limit.
MAIN_MAX_TOKENS = int(os.getenv("MAIN_MAX_TOKENS", "4096"))

# Web search (Tavily)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# MongoDB. Leave MONGODB_URI unset to run with an in-memory store.
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "paypal_shopping_agent")
# Name of an Atlas Vector Search index on contacts.embedding. When unset,
# similarity is computed in the app (fine for a personal contact list).
MONGODB_VECTOR_INDEX = os.getenv("MONGODB_VECTOR_INDEX")

# Embeddings: open-source, 512-token context, 384 dimensions
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = 384

CONTACT_MATCH_THRESHOLD = float(os.getenv("CONTACT_MATCH_THRESHOLD", "0.5"))

# PayPal checkout (REST app credentials from https://developer.paypal.com/dashboard/applications)
PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID")
PAYPAL_CLIENT_SECRET = os.getenv("PAYPAL_CLIENT_SECRET")
PAYPAL_ENV = os.getenv("PAYPAL_ENV", "sandbox")
PAYPAL_BASE_URL = "https://api-m.paypal.com" if PAYPAL_ENV == "live" else "https://api-m.sandbox.paypal.com"
# Deals priced in this currency can be paid in the app; others link out to the store.
PAYPAL_CURRENCY = os.getenv("PAYPAL_CURRENCY", "USD").upper()
# Prices in other currencies are charged in PAYPAL_CURRENCY at these fixed rates, e.g.
# "NGN:0.00065" means 1 NGN = 0.00065 USD. A currency with no rate links out to the store.
EXCHANGE_RATES = {
    code.strip().upper(): Decimal(rate)
    for code, rate in (pair.split(":") for pair in os.getenv("EXCHANGE_RATES", "").split(",") if ":" in pair)
}
# Stores the app pays directly (their PayPal account is the payee). See merchants.example.json.
MERCHANTS_FILE = os.getenv("MERCHANTS_FILE", "merchants.json")

# Gmail, for finding the store's receipt and emailing the store when none arrives.
# OAuth client (type "Web application") from https://console.cloud.google.com/apis/credentials
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/google/callback")
# Gmail and Google Contacts need Google to verify the app before the public can use them.
# Until it does, the app shows them as "opens soon" and the server refuses them. Sign-in
# with Google only needs name and email, so it works either way. Set to 1 once approved.
GOOGLE_APP_VERIFIED = os.getenv("GOOGLE_APP_VERIFIED", "").lower() in ("1", "true", "yes")
# Signs the OAuth state. If unset, a random one is used, so a restart cancels connects in progress.
APP_SECRET = os.getenv("APP_SECRET") or secrets.token_hex(32)

# Web Push (generate keys with: python -m scripts.generate_vapid_keys)
VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY")
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY")
VAPID_SUBJECT = os.getenv("VAPID_SUBJECT")  # mailto:you@example.com

# How long the payment agent waits for a receipt before emailing the store, and how often it looks.
RECEIPT_WAIT_HOURS = float(os.getenv("RECEIPT_WAIT_HOURS", "24"))
RECEIPT_POLL_SECONDS = int(os.getenv("RECEIPT_POLL_SECONDS", "300"))

MAX_AGENT_TURNS = 8
