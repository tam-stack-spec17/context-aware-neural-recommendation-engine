import time
import numpy as np
from typing import Dict, List

class InferenceTelemetryTracker:
    """Monitors real-time inference latency quantiles (P50/P95/P99) and Redis cache hit rates."""
    def __init__(self, sla_threshold_ms: float = 50.0):
        self.sla_threshold_ms = sla_threshold_ms
        self.latencies_ms: List[float] = []
        self.cache_hits: int = 0
        self.cold_start_fallbacks: int = 0
        self.sla_violations: int = 0

    def record_request(self, latency_ms: float, source: str) -> None:
        """Logs an individual recommendation request outcome."""
        self.latencies_ms.append(latency_ms)
        if source == "feature_store_context":
            self.cache_hits += 1
        else:
            self.cold_start_fallbacks += 1

        if latency_ms > self.sla_threshold_ms:
            self.sla_violations += 1

    def generate_sla_report(self) -> Dict[str, float]:
        """Computes production telemetry metrics for operational health audits."""
        total_requests = len(self.latencies_ms)
        if total_requests == 0:
            return {"total_requests": 0, "status": "no_traffic"}

        arr = np.array(self.latencies_ms, dtype=np.float32)
        return {
            "total_requests": total_requests,
            "p50_latency_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_latency_ms": round(float(np.percentile(arr, 99)), 2),
            "cache_hit_ratio": round(self.cache_hits / total_requests, 4),
            "sla_compliance_rate": round(1.0 - (self.sla_violations / total_requests), 4),
        }

if __name__ == "__main__":
    tracker = InferenceTelemetryTracker(sla_threshold_ms=50.0)
    rng = np.random.default_rng(seed=42)
    for _ in range(250):
        sim_latency = float(rng.normal(loc=14.5, scale=4.2))
        sim_source = "feature_store_context" if rng.random() < 0.88 else "cold_start_popularity"
        tracker.record_request(max(1.5, sim_latency), sim_source)

    report = tracker.generate_sla_report()
    print("Production Telemetry Audit Summary:", report)