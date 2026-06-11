"""Qdrant vector store wrapper.

Local file mode by default (zero accounts). Set QDRANT_URL to switch to
Qdrant Cloud (free tier) without changing any code — that is the "cloud" build.
"""
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import config


def get_client():
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY or None)
    return QdrantClient(path=config.QDRANT_PATH)


class Store:
    def __init__(self):
        self.client = get_client()
        self.name = config.COLLECTION

    def ensure(self):
        names = [c.name for c in self.client.get_collections().collections]
        if self.name not in names:
            self.client.create_collection(
                self.name,
                vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
            )

    def upsert(self, items):
        """items: list of {text, source, vector}."""
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=it["vector"],
                payload={"text": it["text"], "source": it["source"]},
            )
            for it in items
        ]
        self.client.upsert(self.name, points=points)

    def search(self, vector, top_k):
        hits = self.client.query_points(
            self.name, query=vector, limit=top_k, with_payload=True
        ).points
        return [
            {
                "text": h.payload.get("text", ""),
                "source": h.payload.get("source", ""),
                "score": h.score,
            }
            for h in hits
        ]
