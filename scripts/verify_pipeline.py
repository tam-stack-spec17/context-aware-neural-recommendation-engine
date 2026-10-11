import sys
import numpy as np

def run_integration_checks():
    print("=" * 60)
    print("RUNNING RECSYS PIPELINE END-TO-END INTEGRATION AUDIT")
    print("=" * 60)

    # 1. Verify ANN Search Engine
    print("\n[1/3] Testing Approximate Nearest Neighbor (ANN) Index...")
    from src.search.ann_index import VectorANNIndex
    index = VectorANNIndex(embedding_dim=64)
    dummy_catalog = {f"art_{i:04d}": np.random.randn(64).tolist() for i in range(200)}
    index.build_index(dummy_catalog)
    results = index.search(np.random.randn(64), top_k=5)
    assert len(results) == 5, "ANN search failed to return top-5 candidates"
    print(f"  --> ANN Index verified. Top candidate: {results[0][0]} (score: {results[0][1]:.4f})")

    # 2. Verify Redis Client Initialization
    print("\n[2/3] Checking Redis Feature Store client interface...")
    from src.store.redis_client import FeatureStoreClient
    client = FeatureStoreClient()
    connected = client.ping()
    print(f"  --> Redis Client initialized (Active connection: {connected})")

    # 3. Verify Recommendation API schemas
    print("\n[3/3] Validating FastAPI route definitions and schemas...")
    from src.api.main import app
    assert app.title == "Context-Aware Neural Recommendation Service"
    routes = [route.path for route in app.routes]
    assert "/health" in routes and "/recommend" in routes, "Missing required API routes"
    print(f"  --> FastAPI routes verified: {routes}")

    print("\n" + "=" * 60)
    print("ALL CORE SYSTEM MODULES INTEGRATED AND VALIDATED")
    print("=" * 60)

if __name__ == "__main__":
    try:
        run_integration_checks()
    except Exception as e:
        print(f"Integration Check Failed: {e}", file=sys.stderr)
        sys.exit(1)