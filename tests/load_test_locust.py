import random
from locust import HttpUser, task, between

class RecommendationLoadTestUser(HttpUser):
    """Simulates concurrent production traffic against the neural recommendation service."""
    wait_time = between(0.1, 0.5)

    @task(4)
    def test_get_recommendations(self):
        """Simulates high-throughput recommendation queries across diverse customer profiles."""
        sample_customer_id = f"user_{random.randint(0, 1000)}"
        top_k = random.choice([5, 10, 20])
        self.client.post(
            f"/recommend?customer_id={sample_customer_id}&top_k={top_k}",
            name="/recommend"
        )

    @task(1)
    def test_health_check(self):
        """Monitors API health and Redis Feature Store connectivity during concurrent load."""
        self.client.get("/health", name="/health")