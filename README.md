# Context-Aware Neural Recommendation Engine

A production-grade **Two-Tower Neural Retrieval System** built with **TensorFlow Recommenders (TFRS)**, **PySpark**, **Redis Feature Store**, **Approximate Nearest Neighbor (ANN)** vector indexing, **FastAPI**, and **Apache Airflow**.

---

## 1. System Architecture

```text
[Raw Transactions & User Metadata]
          │
          ▼  (PySpark ETL, Schema Validation & Cold-Start Filtering)
[Two-Tower Neural Retrieval Model (TFRS)]
  ├── User Query Tower (64-d Dense Projection)
  └── Item Candidate Tower (64-d Dense Projection)
          │
          ▼  (In-Batch Negative Sampling & FactorizedTopK Evaluation)
[Low-Latency Storage & Indexing]
  ├── Redis Feature Store (User Context & Pre-Warmed Vectors, <2ms lookup)
  └── Vector ANN Index (L2-Normalized Cosine Similarity Search)
          │
          ▼
[FastAPI Production Inference Microservice]
  ├── POST /recommend (Real-Time Top-K Retrieval, P95 < 25ms)
  ├── GET  /health    (Service & Redis Connectivity Check)
  └── SLA Telemetry   (P50/P95/P99 Latency & Cache-Hit Monitoring)