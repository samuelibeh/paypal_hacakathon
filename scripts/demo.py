"""End-to-end demo against the running API: import contacts, then ask for a fridge.

Usage: uvicorn app.main:app &  then  python -m scripts.demo
"""

import json

import httpx

BASE = "http://localhost:8000"
USER = "demo-user"

CONTACTS = [
    {"name": "Morris Adeyemi", "phone": "+234 800 000 0001", "notes": "runs an appliance shop in Ikeja, sells fridges and ACs"},
    {"name": "Bob Okafor", "phone": "+234 800 000 0002", "notes": "bought a new Samsung fridge last month"},
    {"name": "Jessica Eze", "phone": "+234 800 000 0003", "notes": "wedding photographer"},
]

with httpx.Client(base_url=BASE, timeout=300) as http:
    print(http.get("/health").json())
    print("\nImporting contacts...")
    for c in http.post("/contacts/import", json={"user_id": USER, "contacts": CONTACTS, "enrich_with_web": False}).json()["imported"]:
        print(f"- {c['name']}: {c['profile']}")

    print("\nAsking the agent...")
    out = http.post("/chat", json={"user_id": USER, "message": "My name is Sam. Where can I buy a fridge?", "location": "Lagos, Nigeria"}).json()
    for call in out["tool_calls"]:
        print(f"[tool] {call['tool']}({json.dumps(call['input'])})")
    print("\n" + out["reply"])
