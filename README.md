# Context-Aware Neural Recommendation Engine

A production-grade **Two-Tower Neural Retrieval System** built with **TensorFlow Recommenders (TFRS)**, **PySpark**, **Redis Feature Store**, **Approximate Nearest Neighbor (ANN)** vector indexing, **FastAPI**, and **Apache Airflow**.

---

## 1. System Architecture

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

---

## 2. Evaluation & Latency Benchmarks

### Model Retrieval Metrics (`src/training/train_eval.py`)
| Metric | Score | Description |
| :--- | :--- | :--- |
| **Recall@10** | `0.4280` | Ground-truth interaction retrieved within top 10 candidates |
| **Recall@50** | `0.7810` | Ground-truth interaction retrieved within top 50 candidates |
| **NDCG@10** | `0.3640` | Position-aware ranking quality across top 10 predictions |

### Serving & Concurrency SLA (`tests/load_test_locust.py` & `src/monitoring/telemetry.py`)
| Metric | Value | Target SLA |
| :--- | :--- | :--- |
| **P50 Latency** | `14.2 ms` | `< 20.0 ms` |
| **P95 Latency** | `21.8 ms` | `< 35.0 ms` |
| **P99 Latency** | `29.4 ms` | `< 50.0 ms` |
| **Redis Cache Hit Ratio** | `88.4%` | `> 80.0%` |

---

## 3. Repository Structure

    ├── .github/workflows/ci.yml       # Automated GitHub Actions syntax & build CI
    ├── dags/model_retrain_dag.py      # Apache Airflow weekly retraining & cache sync DAG
    ├── docs/architecture.md           # Detailed system design & SLA specifications
    ├── scripts/verify_pipeline.py     # Automated end-to-end integration test suite
    ├── src/
    │   ├── api/main.py                # FastAPI recommendation & health endpoints
    │   ├── etl/                       # PySpark data ingestion, cleaning & quality checks
    │   ├── models/two_tower.py        # TFRS User Query & Item Candidate neural towers
    │   ├── monitoring/telemetry.py    # Real-time P50/P95/P99 SLA & cache telemetry
    │   ├── search/ann_index.py        # Cosine-similarity ANN vector search engine
    │   ├── serving/export_model.py    # TF serving signatures & vector batch export
    │   ├── store/redis_client.py      # Low-latency Redis feature store client
    │   ├── store/cache_warmer.py      # Startup batch vector & catalog pre-warmer
    │   └── training/train_eval.py     # Model training loop & Top-K evaluation pipeline
    ├── tests/load_test_locust.py      # Locust concurrency & stress testing suite
    ├── Dockerfile                     # Multi-stage production container image
    └── docker-compose.prod.yml        # Multi-container orchestration (FastAPI + Redis)

---

## 4. Quickstart & Verification Commands

1. **Run End-to-End Integration Audit:**
   `python scripts/verify_pipeline.py`

2. **Launch Production Stack via Docker Compose:**
   `docker-compose -f docker-compose.prod.yml up --build -d`

3. **Start FastAPI Inference Server Locally:**
   `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload`