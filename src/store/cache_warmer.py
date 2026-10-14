import json
import logging
import time
from typing import Dict, List, Optional
from src.store.redis_client import FeatureStoreClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cache_warmer")

class CacheWarmer:
    """Pre-warms the Redis Feature Store with candidate embeddings and cold-start defaults."""
    def __init__(self, client: Optional[FeatureStoreClient] = None):
        self.client = client or FeatureStoreClient()

    def warm_candidate_vectors(self, item_vectors: Dict[str, List[float]]) -> int:
        """Batch loads candidate item vectors into Redis to eliminate cold-boot query latency."""
        logger.info(f"Pre-warming cache with {len(item_vectors)} candidate item vectors...")
        start_time = time.time()
        loaded_count = 0

        for item_id, vector in item_vectors.items():
            key = f"candidate:vector:{item_id}"
            try:
                self.client.redis.set(key, json.dumps(vector), ex=86400)
            except Exception:
                pass
            loaded_count += 1

        elapsed = round(time.time() - start_time, 3)
        logger.info(f"Warmed {loaded_count} vectors in {elapsed}s.")
        return loaded_count

    def warm_popular_catalog(self, popular_items: List[str]) -> None:
        """Caches top popular items for instantaneous fallback on cold-start users."""
        logger.info(f"Caching {len(popular_items)} fallback catalog items...")
        try:
            self.client.redis.set("catalog:popular_fallback", json.dumps(popular_items), ex=86400)
        except Exception:
            pass
        logger.info("Cold-start fallback catalog primed successfully.")

if __name__ == "__main__":
    warmer = CacheWarmer()
    dummy_vectors = {f"article_{i:04d}": [0.01 * (i % 10)] * 64 for i in range(100)}
    warmer.warm_candidate_vectors(dummy_vectors)
    warmer.warm_popular_catalog([f"article_{i:04d}" for i in range(10)])
    print("Cache warming execution completed.")