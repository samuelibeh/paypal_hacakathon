"""Test doubles: scripted Bedrock client and a bag-of-words embedder."""

import copy
import hashlib

import mongomock
import numpy as np
from fastapi.testclient import TestClient

from app import auth
from app.store import Store


class FakeBedrock:
    """Returns queued Converse responses and records every request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def converse(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return self.responses.pop(0)


def text_reply(text):
    return {"output": {"message": {"role": "assistant", "content": [{"text": text}]}}, "stopReason": "end_turn"}


def tool_call(name, args, tool_id="t1"):
    return {
        "output": {"message": {"role": "assistant", "content": [
            {"toolUse": {"toolUseId": tool_id, "name": name, "input": args}}
        ]}},
        "stopReason": "tool_use",
    }


class FakeEmbedder:
    def _vec(self, text):
        v = np.zeros(64)
        for word in text.lower().replace(".", " ").split():
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    embed_passage = embed_query = _vec


def fake_store():
    return Store(client=mongomock.MongoClient())


def signed_in(app, user_id: str) -> TestClient:
    """A test client with a valid sign-in cookie for user_id."""
    client = TestClient(app)
    client.cookies.set(auth.COOKIE, auth.session_token(user_id))
    return client
