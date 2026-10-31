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
    account = {"name": "Sam", "email": "demo@example.com", "password": "demo-password"}
    signed_in = http.post("/auth/signin", json=account)
    USER = (signed_in if signed_in.is_success else http.post("/auth/signup", json=account)).json()["user_id"]
    print("\nImporting contacts (descriptions are written in the background)...")
    print(http.post("/contacts/import", json={"user_id": USER, "contacts": CONTACTS}).json())

    print("\nAsking the agent...")
    out = http.post("/chat", json={"user_id": USER, "message": "My name is Sam. Where can I buy a fridge?", "location": "Lagos, Nigeria"}).json()
    for call in out["tool_calls"]:
        print(f"[tool] {call['tool']}({json.dumps(call['input'])})")
    print("\n" + out["reply"])
