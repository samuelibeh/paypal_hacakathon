"""Local open-source sentence embeddings (bge-small-en-v1.5, 512-token context)."""

from functools import lru_cache

import numpy as np

from app import config

# bge models expect this prefix on short search queries, not on stored passages.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 512

    def embed_passage(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(QUERY_PREFIX + text, normalize_embeddings=True).tolist()


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()


def cosine(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else 0.0
