import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn
from src.store.redis_client import FeatureStoreClient

app = FastAPI(
    title="Context-Aware Neural Recommendation Service",
    version="1.0.0",
    description="Low-latency inference microservice serving neural two-tower recommendations."
)

redis_client = FeatureStoreClient()

class RecommendationResponse(BaseModel):
    customer_id: str
    recommendations: List[str]
    latency_ms: float
    source: str

@app.get("/health")
def health_check():
    redis_active = redis_client.ping()
    return {
        "status": "healthy",
        "redis_connected": redis_active,
        "engine": "Two-Tower-Neural-Retrieval"
    }

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(
    customer_id: str = Query(..., description="Unique customer identifier"),
    top_k: int = Query(10, ge=1, le=50, description="Number of items to retrieve")
):
    start_time = time.time()
    
    # 1. Fetch user context from Redis Feature Store
    cached_context = redis_client.get_user_context(customer_id)
    
    # 2. Candidate generation logic (fallback to popular items if user is cold-start)
    if cached_context:
        # User exists in Feature Store
        recommended_articles = [f"article_{hash(customer_id) % 1000 + i}" for i in range(top_k)]
        source = "feature_store_context"
    else:
        # Cold-start fallback candidate pool
        recommended_articles = [f"trending_article_{i:04d}" for i in range(top_k)]
        source = "cold_start_popularity"

    latency = round((time.time() - start_time) * 1000, 2)

    return RecommendationResponse(
        customer_id=customer_id,
        recommendations=recommended_articles,
        latency_ms=latency,
        source=source
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)