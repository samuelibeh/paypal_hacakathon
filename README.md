# PayPal Shopping Agent

A user asks "where can I buy a fridge?". A main Claude agent passes the question to
two sub-agents and returns the best deals it found, plus any of the user's contacts who
could help: someone who sells fridges, repairs them, or just bought one.

```
                         ┌──────────────┐
  user ──"where can I…"──▶   Main AI    │◀──▶ storage (user facts, MongoDB)
       ◀── answer ───────│ (Claude)     │◀──▶ Tavily web search
                         └──┬────────┬──┘
                find_deals  │        │  find_contacts
                            ▼        ▼
              ┌──────────────────┐  ┌───────────────────────────┐
              │ Deal sub-agent   │  │ Contact sub-agent         │
              │ Tavily search    │  │ bge-small embeddings      │
              │ location-aware   │  │ MongoDB vector search     │
              └──────────────────┘  └───────────────────────────┘
```

- **Main agent** (`app/agents/main_agent.py`): runs a Claude tool-use loop. Its tools are
  `get_profile` / `remember` (user storage), `web_search`, `find_deals`, `find_contacts` and
  `add_to_cart`. After showing deals it asks one narrowing question that suits the product
  (the size of the space for an appliance, the event or outfit colour for a bag), accepts a
  photo, gives its own pick with reasons, and adds the item the user chooses to their cart.
- **Checkout** (`app/checkout.py`, `app/paypal.py`, `app/merchants.py`): deals are stored with
  an id when they're shown. The user selects them into a cart, in the UI or by telling the
  agent, and pays with PayPal (Orders v2 API and JS SDK buttons). The server builds the order
  from the stored prices, never from the browser. The order names the store's own PayPal
  account as the payee, so the money goes to the store. A deal can be paid in the app only if
  its link is on a store listed in `MERCHANTS_FILE` and it's priced in `PAYPAL_CURRENCY`; the
  rest link out to the store. A cart holds one store's items at a time.
- **Payment sub-agent** (`app/agents/payment_agent.py`): when the user approves the payment in
  the PayPal window, it captures it (the payment to the store) and pushes "Payment sent". A
  background thread then checks the user's Gmail every `RECEIPT_POLL_SECONDS` for mail from
  the store, and Claude decides whether any of it confirms this purchase. If one does, the
  user gets "Payment complete". If nothing arrives within `RECEIPT_WAIT_HOURS`, the agent
  emails the store's support address from the user's Gmail, in the user's own voice, asking
  them to confirm the order, and pushes "No receipt yet, I emailed them". If that brings no
  reply within another wait, it tells the user to contact the store, with the PayPal
  transaction ID. The main agent can report all this with `get_purchases`.
- **Deal sub-agent** (`app/agents/deal_agent.py`): a Claude agent with Tavily search. It looks for
  discounted prices near the user if they shared their location, and on major online stores
  (Amazon, Walmart, Temu, eBay…) if not. It returns structured deals.
- **Contact sub-agent** (`app/agents/contact_agent.py`): on import, if web enrichment is on,
  Claude turns each contact into one targeted search query (`"Morris Adeyemi" Ikeja appliance shop`),
  or `NONE` when a bare name can't be told apart from strangers. We run that Tavily search
  ourselves and pass the results to Claude, which writes a 2–3 sentence shopping-relevant
  profile. The profile is embedded and stored in MongoDB. At query time it embeds the need
  ("fridge") and returns the contacts whose profiles are most similar by cosine similarity.

| Piece | Choice |
|---|---|
| API | FastAPI (`app/main.py`) |
| LLM | Claude Haiku 4.5 on AWS Bedrock (Converse API, `app/llm.py`) |
| Embeddings | `BAAI/bge-small-en-v1.5`: open source (MIT), 512-token context, 384 dims, runs locally |
| Vector DB | MongoDB: Atlas `$vectorSearch` if an index is configured, otherwise cosine in-app |
| Web search | Tavily |

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in AWS_BEARER_TOKEN_BEDROCK, TAVILY_API_KEY, MONGODB_URI
uvicorn app.main:app --reload
python -m scripts.demo        # imports 3 sample contacts and asks for a fridge
```

If `MONGODB_URI` is unset, the app uses an in-memory store, which is lost on restart. If
`TAVILY_API_KEY` is unset, web search tools return an error and the agent tells the user.

### PayPal sandbox

1. At https://developer.paypal.com/dashboard/applications/sandbox, create an app and copy
   its client id and secret into `PAYPAL_CLIENT_ID` / `PAYPAL_CLIENT_SECRET`.
2. Under Testing Tools > Sandbox Accounts, use the personal (buyer) account to log in at
   the PayPal popup. No real money moves in sandbox.

### Stores the app can pay

Copy `merchants.example.json` to `merchants.json` and list each store: the domains its deal
links are on, its PayPal account (`paypal_email`, the payee), the domains its receipts come
from, and its support email. For the sandbox, create a business account under Testing Tools
> Sandbox Accounts and use its email as `paypal_email`.

Paying another merchant's account works in the sandbox. Before going live, check with PayPal
that your app may take payments for other merchants (PayPal Commerce Platform), and only list
stores that have agreed to it.

### Gmail (receipt checks)

1. In Google Cloud, enable the Gmail API, set up the OAuth consent screen (testing mode is
   fine for a demo; add your Google account as a test user), and create an OAuth client of
   type "Web application" with `http://localhost:8000/google/callback` as a redirect URI.
2. Put its id and secret in `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET`, and a long random
   `APP_SECRET`.
3. In the web UI, open Purchases > Connect Gmail. The app asks for read access (to find the
   receipt) and send access (for the follow-up email).

### Push notifications

```bash
python -m scripts.generate_vapid_keys   # paste the three lines into .env
```

Then open Purchases > Turn on notifications in the browser. On an iPhone, add the page to the
home screen first (iOS 16.4 or later); browsers only allow push on `localhost` or HTTPS.

### MongoDB Atlas vector search

```bash
python -m scripts.create_vector_index   # creates "contact_embeddings" on contacts.embedding
# then in .env:
MONGODB_VECTOR_INDEX=contact_embeddings
```

## API

| Method | Path | Body / params |
|---|---|---|
| POST | `/chat` | `{user_id, message, location?, image?: {media_type, data}}`: send `location` only if the user allowed it; `data` is base64 |
| POST | `/chat/{user_id}/reset` | clears conversation history |
| POST | `/contacts/import` | `{user_id, contacts: [{name, phone?, email?, notes?}], enrich_with_web}` |
| GET | `/contacts/{user_id}` | list profiled contacts |
| GET | `/contacts/{user_id}/search?need=fridge` | vector search only |
| GET | `/users/{user_id}/profile` | stored user facts |
| GET | `/cart/{user_id}` | selected deals and total |
| POST | `/cart/{user_id}/items` | `{deal_id}` |
| DELETE | `/cart/{user_id}/items/{deal_id}` | |
| GET | `/paypal/config` | public client id and currency for the JS SDK |
| POST | `/paypal/orders` | `{user_id}`: creates a PayPal order from the cart |
| POST | `/paypal/orders/{order_id}/capture` | `{user_id}`: the payment agent pays the store and starts watching for the receipt |
| GET | `/purchases/{user_id}` | paid orders and their receipt status |
| GET | `/google/connect?user_id=` | starts the Gmail sign-in |
| GET | `/google/status/{user_id}` | whether Gmail is set up and connected |
| DELETE | `/google/{user_id}` | disconnects Gmail |
| GET | `/push/key` | VAPID public key for the browser |
| POST | `/push/subscribe` | `{user_id, subscription}` from `PushManager.subscribe()` |
| GET | `/health` | which backends are configured |

Web UI: http://localhost:8000/ (chat, contact import and search, stored profile; `app/static/index.html`)
Interactive docs: http://localhost:8000/docs

## Tests

```bash
pytest
```

The tests use a scripted fake Bedrock client, a bag-of-words embedder, and mongomock, so they
need no network, keys, or model download.

## Privacy notes

Importing contacts and searching the web about them should be opt-in. The UI should ask
first, and `enrich_with_web` defaults on only for the API. Profiles are told to keep to
shopping-relevant facts (job, business, products) and to leave out addresses or other
sensitive details. Contacts are always scoped to the `user_id` that imported them.

`user_id` comes from the browser and nothing checks it, so anyone who knows a user id can act
as that user. That's fine for a demo, but add real sign-in before connecting real Gmail
accounts: Gmail refresh tokens are stored in MongoDB (`google_accounts`) and should then be
encrypted too. The payment agent only reads mail from the store's own domains, and only sends
the one follow-up email for a purchase the user paid for.

`claude_api.py` is a standalone helper for the direct Anthropic API and for Bedrock, from the
first iteration of this repo.
