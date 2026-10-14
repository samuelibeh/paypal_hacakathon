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
  `get_profile` / `remember` (user storage), `web_search`, `find_deals`, and `find_contacts`.
- **Deal sub-agent** (`app/agents/deal_agent.py`): a Claude agent with Tavily search. It looks for
  discounted prices near the user if they shared their location, and on major online stores
  (Amazon, Walmart, Temu, eBay…) if not. It returns structured deals.
- **Contact sub-agent** (`app/agents/contact_agent.py`): on import, it writes a 2–3 sentence
  shopping-relevant profile of each contact from the user's notes, plus a web search if
  enabled, then embeds the profile and stores it in MongoDB. At query time it embeds the
  need ("fridge") and returns the contacts whose profiles are most similar by cosine similarity.

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

### MongoDB Atlas vector search

```bash
python -m scripts.create_vector_index   # creates "contact_embeddings" on contacts.embedding
# then in .env:
MONGODB_VECTOR_INDEX=contact_embeddings
```

## API

| Method | Path | Body / params |
|---|---|---|
| POST | `/chat` | `{user_id, message, location?}`: send `location` only if the user allowed it |
| POST | `/chat/{user_id}/reset` | clears conversation history |
| POST | `/contacts/import` | `{user_id, contacts: [{name, phone?, email?, notes?}], enrich_with_web}` |
| GET | `/contacts/{user_id}` | list profiled contacts |
| GET | `/contacts/{user_id}/search?need=fridge` | vector search only |
| GET | `/users/{user_id}/profile` | stored user facts |
| GET | `/health` | which backends are configured |

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

`claude_api.py` is a standalone helper for the direct Anthropic API and for Bedrock, from the
first iteration of this repo.
