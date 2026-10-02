import os
import json
from typing import List, Optional
import redis

class FeatureStoreClient:
    """Low-latency interface to cache user profiles and candidate embeddings in Redis."""
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.host = os.getenv("REDIS_HOST", host)
        self.port = int(os.getenv("REDIS_PORT", port))
        self.client = redis.Redis(host=self.host, port=self.port, db=db, decode_responses=True)

    def ping(self) -> bool:
        try:
            return self.client.ping()
        except redis.ConnectionError:
            return False

    def cache_user_context(self, customer_id: str, context: dict, ttl_seconds: int = 86400):
        self.client.setex(f"user:{customer_id}", ttl_seconds, json.dumps(context))

    def get_user_context(self, customer_id: str) -> Optional[dict]:
        raw = self.client.get(f"user:{customer_id}")
        return json.loads(raw) if raw else None

    def cache_item_vector(self, article_id: str, vector: List[float]):
        self.client.set(f"item_vec:{article_id}", json.dumps(vector))

    def get_item_vector(self, article_id: str) -> Optional[List[float]]:
        raw = self.client.get(f"item_vec:{article_id}")
        return json.loads(raw) if raw else None

if __name__ == "__main__":
    client = FeatureStoreClient()
    print(f"Redis Feature Store Client initialized: {client.host}:{client.port}")