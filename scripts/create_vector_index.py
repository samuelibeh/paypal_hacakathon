"""Create the Atlas Vector Search index on contacts.embedding.

Usage: MONGODB_URI=mongodb+srv://... python -m scripts.create_vector_index
Then set MONGODB_VECTOR_INDEX=contact_embeddings in .env.
"""

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

from app import config

INDEX_NAME = "contact_embeddings"

if __name__ == "__main__":
    contacts = MongoClient(config.MONGODB_URI)[config.MONGODB_DB]["contacts"]
    contacts.create_search_index(SearchIndexModel(
        name=INDEX_NAME,
        type="vectorSearch",
        definition={"fields": [
            {"type": "vector", "path": "embedding", "numDimensions": config.EMBEDDING_DIM, "similarity": "cosine"},
            {"type": "filter", "path": "user_id"},
        ]},
    ))
    print(f"Created index {INDEX_NAME!r}. Set MONGODB_VECTOR_INDEX={INDEX_NAME} in .env")
