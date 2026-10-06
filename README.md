# Context-Aware Neural Recommendation Engine (Deep Learning) 

[![CI](https://github.com/tam-stack-spec17/context-aware-neural-recommendation-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/tam-stack-spec17/context-aware-neural-recommendation-engine/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10-blue)
![Status](https://img.shields.io/badge/status-mid--project%20review-orange)

A deep-learning recommendation system built around a **two-tower neural retrieval model** (TensorFlow Recommenders), with **PySpark** for data processing and a **Redis** feature store for low-latency lookups.

> **Project stage: mid-project review.** Several components are working in isolation, but the end-to-end pipeline is **not yet connected** and the model has only been run on **synthetic data**. This README states clearly what is implemented, what is partial, and what is planned.

---

## Table of Contents

1. [Project Status](#1-project-status)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Repository Structure](#3-repository-structure)
4. [Component Details](#4-component-details)
5. [Tech Stack](#5-tech-stack)
6. [Getting Started](#6-getting-started)
7. [Continuous Integration](#7-continuous-integration)
8. [Team and Contributions](#8-team-and-contributions)
9. [Known Issues and Limitations](#9-known-issues-and-limitations)
10. [Roadmap](#10-roadmap)

---

## 1. Project Status

| Status | Meaning |
|---|---|
| ✅ Implemented | Code exists in the repository and is complete for its stated purpose |
| 🟡 Partial | Code exists but is incomplete, not integrated, or has caveats |
| ⬜ Planned | No implementation in the repository yet |

| Stage | Component | Status | Notes |
|---|---|---|---|
| Infrastructure | Reusable Spark session builder (`src/utils/spark.py`) | ✅ Implemented | Smoke-tested locally; not yet called by other modules |
| Data ingestion | Loading raw data into Spark | ⬜ Planned | No ingestion code; `data/raw/` and `data/processed/` contain only `.gitkeep` |
| Cleaning | Null-row removal (`clean_transactions`) | ✅ Implemented | Uses `DataFrame.dropna()`; not yet wired to an ingestion step |
| Cleaning | Cold-start user identification | 🟡 Partial | Logic exists only in a stray file, `clean_transactions(2).py`; not in the main module (see [Known Issues](#9-known-issues-and-limitations)) |
| Feature / context processing | Context feature engineering | ⬜ Planned | No code. The current model uses IDs only (see below) |
| Modeling | Two-tower retrieval model (`src/models/two_tower.py`) | 🟡 Partial | Working architecture, but ID-embedding only; no context features yet |
| Training / evaluation | Train and evaluate loop (`src/training/train_eval.py`) | 🟡 Partial | Runs on synthetic data only; metric output needs correction (see [Known Issues](#9-known-issues-and-limitations)) |
| Feature store | Redis client (`src/store/redis_client.py`) | ✅ Implemented | Needs a running Redis instance; not yet called by other modules |
| Serving | Query-tower export and candidate vector export (`src/serving/export_model.py`) | 🟡 Partial | Helper class and function defined; not invoked by any pipeline step |
| Serving | REST API | ⬜ Planned | `fastapi` and `uvicorn` are listed in `requirements.txt`, but no API code exists |
| DevOps | Redis via Docker Compose | ✅ Implemented | `docker-compose.yml` runs Redis only |
| DevOps | Application Dockerfile | ⬜ Planned | None in the repository |
| DevOps | CI | 🟡 Partial | Installs dependencies and runs a syntax check only |
| Quality | Automated tests | ⬜ Planned | No test suite |
| Data | Dataset selection and documentation | ⬜ Planned | No real dataset is included or documented |

---

## 2. Pipeline Overview

The intended flow, with the current state of each stage:

```
 Data ingestion ──► Cleaning / preprocessing ──► Feature & context processing ──► Two-tower model ──► Serving
   (planned)         (null removal done;          (planned)                       (ID-only, trained     (export helpers +
                      cold-start partial)                                          on synthetic data)    Redis client;
                                                                                                         API planned)
```

At serving time, the intended design (reflected in `export_model.py` and `redis_client.py`) is:

1. Candidate (item) embeddings are computed offline by the candidate tower and exported.
2. Item vectors and user context are cached in Redis.
3. The query tower turns a `customer_id` into an embedding that is matched against candidate vectors.

Steps 1 and 2 have supporting code; the end-to-end wiring and the API are not built yet.

---

## 3. Repository Structure

```
context-aware-neural-recommendation-engine/
├── .github/workflows/ci.yml          # CI: install dependencies + syntax check
├── data/
│   ├── raw/.gitkeep                  # placeholder (raw data is git-ignored)
│   └── processed/.gitkeep            # placeholder (processed data is git-ignored)
├── src/
│   ├── etl/
│   │   ├── clean_transactions.py             # null-row removal (main module)
│   │   ├── clean_transactions(2).py          # contains cold-start logic (stray copy)
│   │   ├── clean_transactions (1).py         # duplicate of the main module (stray copy)
│   │   └── clean_transactions_py.ipynb       # Colab scratch notebook (early stub, outdated)
│   ├── models/two_tower.py           # TFRS two-tower retrieval model
│   ├── store/redis_client.py         # Redis feature store client
│   ├── serving/export_model.py       # query-tower serving signature + candidate vector export
│   ├── training/train_eval.py        # training and evaluation on synthetic data
│   └── utils/spark.py                # reusable SparkSession builder
├── docker-compose.yml                # Redis service
├── requirements.txt
└── README.md
```

---

## 4. Component Details

### 4.1 Spark Session Builder: `src/utils/spark.py` ✅

Provides `get_spark_session()`, a reusable PySpark session factory.

| Parameter | Default | Purpose |
|---|---|---|
| `app_name` | `"RecommendationSystem"` | Spark application name |
| `driver_memory` | `"2g"` | Sets `spark.driver.memory` |
| `executor_memory` | `"2g"` | Sets `spark.executor.memory` |

It also enables Apache Arrow (`spark.sql.execution.arrow.pyspark.enabled=true`) for faster Spark/pandas conversion, and uses `getOrCreate()` so an existing session is reused.

```python
from src.utils.spark import get_spark_session

spark = get_spark_session(app_name="RecSys-ETL", driver_memory="4g", executor_memory="4g")
```

**Verification:** a local smoke test confirmed the function returns a session with the requested app name, Arrow enabled, and that `clean_transactions` runs on a DataFrame created from it. Other modules do not yet import this builder.

### 4.2 Data Cleaning: `src/etl/` 🟡

- **`clean_transactions(df)`** ✅ — removes any row containing a null value (`df.dropna()`) and returns the cleaned PySpark DataFrame.
- **`identify_cold_start_users(df, user_column, threshold=5)`** 🟡 — groups by the user column and returns users with fewer than `threshold` (default 5) transactions. This function exists **only** in `clean_transactions(2).py`. That filename contains parentheses and a space, so it cannot be imported as a Python module, and the function is not in `clean_transactions.py`.

### 4.3 Two-Tower Model: `src/models/two_tower.py` 🟡

Built with TensorFlow Recommenders:

- **`UserQueryTower`** — `StringLookup` → `Embedding` → `Dense(128, relu)` → `Dense(embedding_dim)`. Input: `customer_id`.
- **`ItemCandidateTower`** — the same structure. Input: `article_id`.
- **`TwoTowerRecommendationEngine`** — a `tfrs.Model` with a `tfrs.tasks.Retrieval` task (in-batch negatives) and `FactorizedTopK` metrics over the candidate set.

> **Current limitation:** despite the project name, the towers consume only user and item **identifiers**. No contextual features (time, device, location, demographics, item attributes, etc.) are used yet, even though some docstrings mention demographics and item context. Adding context features is the main modeling task still to be done.

### 4.4 Training and Evaluation: `src/training/train_eval.py` 🟡

- Generates **synthetic** interactions (50 users, 100 items, 2,000 samples), splits 1,600 / 400 for train / test, trains for 3 epochs with Adam (`learning_rate=0.01`, `embedding_dim=32`), then evaluates.
- It is a **pipeline check**, not a benchmark. No real dataset is used, so no results from this script should be presented as model performance. See the metric caveat in [Known Issues](#9-known-issues-and-limitations).

### 4.5 Redis Feature Store: `src/store/redis_client.py` ✅

`FeatureStoreClient` wraps `redis-py`. Host and port come from the `REDIS_HOST` / `REDIS_PORT` environment variables, falling back to `localhost:6379`.

| Method | Description |
|---|---|
| `ping()` | Returns `True` if Redis is reachable |
| `cache_user_context(customer_id, context, ttl_seconds=86400)` | Stores a JSON user context under `user:<id>` with a TTL |
| `get_user_context(customer_id)` | Reads the user context back |
| `cache_item_vector(article_id, vector)` | Stores an item embedding under `item_vec:<id>` |
| `get_item_vector(article_id)` | Reads an item embedding back |

### 4.6 Serving Export: `src/serving/export_model.py` 🟡

- **`QueryServingModule`** — a `tf.Module` wrapping the query tower with a `tf.function` input signature (`customer_id`: string vector).
- **`export_candidate_embeddings(candidate_tower, article_ids, export_path)`** — computes item vectors and writes them to a JSON file.

Neither is called from the training script yet, and no model is saved to disk.

### 4.7 Infrastructure ✅

`docker-compose.yml` starts a `redis:7-alpine` container (`recsys_redis`) on port 6379 with a persistent volume. There is currently no Dockerfile for the application itself.

---

## 5. Tech Stack

From `requirements.txt`:

| Area | Library |
|---|---|
| Data processing | `pyspark>=3.5.0`, `pandas>=2.0.0`, `pyarrow>=14.0.0`, `numpy>=1.24.0` |
| Modeling | `tensorflow>=2.15.0`, `tensorflow-recommenders>=0.7.3` |
| Feature store | `redis>=5.0.0` (client), Redis 7 (Docker) |
| API *(declared, not yet used)* | `fastapi>=0.110.0`, `uvicorn>=0.28.0` |

PySpark also requires a Java runtime (JDK) on the machine.

---

## 6. Getting Started

**Prerequisites:** Python 3.10 (the version used in CI), Java (for PySpark), Docker (optional, for Redis).

```bash
# 1. Clone
git clone https://github.com/tam-stack-spec17/context-aware-neural-recommendation-engine.git
cd context-aware-neural-recommendation-engine

# 2. Create an environment and install dependencies
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. (Optional) Start Redis
docker compose up -d

# 4. Run the synthetic-data training/evaluation pipeline check
python -m src.training.train_eval

# 5. (Optional) Check the Redis client configuration
python -m src.store.redis_client
```

Data directories `data/raw/` and `data/processed/` are git-ignored (except `.gitkeep`); no dataset is bundled with the repository.

---

## 7. Continuous Integration

`.github/workflows/ci.yml` runs on every push to `main`: it sets up Python 3.10, installs `requirements.txt`, and runs `python -m compileall src/`. This catches syntax errors only. There are no unit tests, linting, or training runs in CI.

---

## 8. Team and Contributions

| Member | Role | Contribution |
|---|---|---|
| **Chinmaysk08** | Team Lead | Reusable PySpark session builder (`src/utils/spark.py`) |

Other contributions below are taken from the Git commit history and should be confirmed by the team before the review:

| Contributor (Git author) | Commits touch |
|---|---|
| `tam-stack-spec17` | Repository setup, Docker Compose, CI, Redis client, two-tower model, training/evaluation, serving export |
| Shivarathri Renuka | Transaction cleaning module, cold-start user identification |

---

## 9. Known Issues and Limitations

These are documented here for transparency; they have **not** been fixed as part of this README update.

1. **Evaluation script prints hardcoded fallback values.** In `train_eval.py`, each metric is read with `.get(key, default)` using defaults of `0.428` (Recall@10), `0.781` (Recall@50), and `0.364` (NDCG@10). If a key is missing from the results, the printed "benchmark" is a constant rather than a computed value. The `factorized_top_k/top_10_ndcg` key is not a standard TFRS `FactorizedTopK` output, so NDCG@10 will likely always print the placeholder. **These numbers must not be reported as results.** Recall@K is available through TFRS's `top_K_categorical_accuracy` keys.
2. **Cold-start logic is not in the main module.** `identify_cold_start_users` lives only in `src/etl/clean_transactions(2).py`, which cannot be imported.
3. **Stray files in `src/etl/`.** `clean_transactions (1).py` and `clean_transactions(2).py` look like accidental upload copies, and `clean_transactions_py.ipynb` is an outdated Colab scratch notebook that contains an empty function stub. These should be merged or removed.
4. **"Context-aware" is not yet realized.** The model only uses `customer_id` and `article_id`.
5. **Spark settings on an existing session.** Memory options only take effect when a new JVM/session starts. Calling `get_spark_session()` again in the same process re-applies the supplied configuration values to the existing session without restarting it, so the memory values may not reflect actual allocation.
6. **Modules are isolated.** The Spark builder, ETL, Redis client, and export helpers are not yet called from a common pipeline entry point.
7. **No tests**, and CI does not exercise runtime behavior.
8. **`fastapi` / `uvicorn` are dependencies without code.**

---

## 10. Roadmap

Planned work, in suggested order (none of this is implemented yet):

1. Select and document a real dataset; add an ingestion step using `get_spark_session()`.
2. Merge cold-start identification into `clean_transactions.py` and remove the stray files.
3. Build feature/context processing (e.g., user and item attributes, temporal context).
4. Extend the two towers to consume context features.
5. Train on real data; fix metric reporting (remove hardcoded fallbacks, add a real NDCG computation) and record genuine results.
6. Save the trained model, export candidate vectors, and load them into Redis.
7. Implement a FastAPI recommendation endpoint backed by the query tower and Redis.
8. Add unit tests, extend CI beyond syntax checks, and add an application Dockerfile.
9. Define cold-start recommendation handling for the identified users.
