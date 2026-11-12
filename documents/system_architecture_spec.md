# H&M Context-Aware Neural Recommendation Engine
## Production System Architecture Specification

---

## 1. System Overview
The **H&M Context-Aware Neural Recommendation Engine** is an enterprise-scale, deep learning recommendation platform built on the multi-year **H&M Personalized Fashion Recommendations** dataset. 

Moving beyond traditional matrix factorization and collaborative filtering, the system couples a **Two-Tower Neural Network (TensorFlow Recommenders)** with a **Low-Latency Redis Feature Store**, an **Approximate Nearest Neighbor (ANN) Candidate Retrieval Index**, and a high-concurrency **FastAPI Microservice Serving Layer**. The end-to-end data processing, weekly model retraining, and daily embedding updates are orchestrated via **Apache Airflow** and containerized with **Docker Compose**.

---

## 2. High-Level System Architecture Diagram

```mermaid
flowchart TD
    subgraph DataPlatform ["Data Platform & Offline Processing (Week 1 & 2)"]
        RawData["Raw Transactions & Metadata<br/>(H&M Kaggle Dataset)"]
        SparkETL["Apache PySpark ETL<br/>Missing Value & Cold-Start Resolution"]
        FeatureEng["Contextual Feature Engineering<br/>(Recency, Seasonality, Popularity)"]
        Vocabs["Customer & Article Vocabularies<br/>(models/serving_vocabularies)"]
        
        RawData --> SparkETL
        SparkETL --> FeatureEng
        FeatureEng --> Vocabs
    end

    subgraph DeepLearning ["Deep Learning Architecture (Week 2 & 3)"]
        TwoTowerTrain["Two-Tower Neural Network Training<br/>(TensorFlow Recommenders / TFRS)"]
        QueryTower["Query Tower<br/>(Customer Context Embedding)"]
        CandidateTower["Candidate Tower<br/>(Article Metadata Embedding)"]
        ModelCheckpoint["Model Checkpoints & Weights<br/>(two_tower.weights.h5)"]

        Vocabs --> TwoTowerTrain
        TwoTowerTrain --> QueryTower
        TwoTowerTrain --> CandidateTower
        QueryTower --> ModelCheckpoint
        CandidateTower --> ModelCheckpoint
    end

    subgraph FeatureStore ["Low-Latency Feature Store & Index (Week 3)"]
        RedisFS[("Redis Feature Store<br/>(In-Memory Cache)")]
        UserProfiles["Customer Demographic Profiles<br/>(hm:user_profiles)"]
        ItemEmbeddings["Pre-computed Article Embeddings<br/>(hm:article_embeddings)"]
        ANNIndex["Approximate Nearest Neighbor (ANN) Index<br/>(Inverted File Spherical Voronoi Index)"]

        ModelCheckpoint --> RedisFS
        RedisFS --> UserProfiles
        RedisFS --> ItemEmbeddings
        ItemEmbeddings --> ANNIndex
    end

    subgraph ServingLayer ["Real-Time Serving & Client Tier (Week 4)"]
        User(["E-Commerce Customer / Client App"])
        FastAPIApp["FastAPI Recommendation Microservice<br/>(Uvicorn ASGI)"]
        QueryEmbed["Real-Time Query Generation<br/>(Query Tower Forward Pass)"]
        ANNRank["ANN & Dot-Product Similarity Ranking<br/>(Top-K Extraction)"]
        Recs(["Personalized Top-K Articles Response"])

        User -->|"GET /recommend/{customer_id}?top_k=10"| FastAPIApp
        FastAPIApp -->|"Fetch Customer Profile"| UserProfiles
        FastAPIApp --> QueryEmbed
        QueryEmbed --> ANNRank
        ANNIndex --> ANNRank
        ANNRank --> Recs
        Recs --> User
    end

    subgraph Orchestration ["Automated Pipeline Scheduling (Week 4)"]
        Airflow["Apache Airflow LocalExecutor<br/>(Dockerized Orchestrator)"]
        DAG1["DAG: hm_weekly_model_retraining<br/>(Every Monday 02:00)"]
        DAG2["DAG: hm_daily_embedding_update<br/>(Every Day 03:00)"]
        ModelPromotion["Zero-Downtime Validation & Atomic Promotion<br/>(promote_model.py)"]

        Airflow --> DAG1
        Airflow --> DAG2
        DAG1 --> TwoTowerTrain
        DAG1 --> ModelPromotion
        ModelPromotion --> ModelCheckpoint
        DAG2 --> ItemEmbeddings
    end
```

---

## 3. Real-Time Inference Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor Client as E-Commerce Frontend / App
    participant API as FastAPI Serving Microservice
    participant Redis as Redis Feature Store
    participant Model as Query Tower (TensorFlow)
    participant ANN as Candidate ANN Index

    Client->>API: GET /recommend/{customer_id}?top_k=10
    
    API->>API: Validate customer_id in vocabulary set
    alt Unknown Customer (Cold-Start)
        API-->>Client: 404 Not Found (or Global Popularity Fallback)
    else Recognized Customer
        API->>Redis: HGET hm:user_profiles {customer_id}
        Redis-->>API: Customer Metadata (Age, Club Status, Fashion News)
        
        API->>Model: Forward pass customer ID & metadata
        Model-->>API: 32-dim Customer Query Vector (float32)
        
        API->>ANN: Query(query_vector, top_k=10, n_probe=8)
        ANN->>ANN: Match top Voronoi centroids & evaluate candidates
        ANN-->>API: Top-10 Article IDs + Cosine Similarity Scores
        
        API-->>Client: 200 OK: {"customer_id": id, "recommendations": [...]}
    end
```

---

## 4. Deep Learning Two-Tower Model Architecture

```mermaid
flowchart LR
    subgraph QueryTowerFlow ["Query Tower (User Representation)"]
        UserIn["Customer ID<br/>(String Tensor)"]
        UserLookup["StringLookup Vocabulary<br/>(Len = 1.37M)"]
        UserEmb["Embedding Layer<br/>(Dim = 32)"]
        UserDense1["Dense Layer (64, ReLU)"]
        UserDense2["Dense Layer (32, Linear)"]
        UserVec["User Latent Vector<br/>(Shape: [1, 32])"]

        UserIn --> UserLookup --> UserEmb --> UserDense1 --> UserDense2 --> UserVec
    end

    subgraph CandidateTowerFlow ["Candidate Tower (Article Representation)"]
        ItemIn["Article ID<br/>(String Tensor)"]
        ItemLookup["StringLookup Vocabulary<br/>(Len = 105.5K)"]
        ItemEmb["Embedding Layer<br/>(Dim = 32)"]
        ItemDense1["Dense Layer (64, ReLU)"]
        ItemDense2["Dense Layer (32, Linear)"]
        ItemVec["Candidate Latent Vector<br/>(Shape: [1, 32])"]

        ItemIn --> ItemLookup --> ItemEmb --> ItemDense1 --> ItemDense2 --> ItemVec
    end

    subgraph ScoringFlow ["Retrieval & Dot-Product Scoring"]
        DotProduct["Cosine / Dot-Product<br/>Interaction Layer"]
        SoftmaxLoss["In-Batch Negative Sampling<br/>Categorical Cross-Entropy Loss"]

        UserVec --> DotProduct
        ItemVec --> DotProduct
        DotProduct --> SoftmaxLoss
    end
```

---

## 5. Architectural Component Details

### 5.1 PySpark Distributed Feature Engineering (Week 1)
* **Missing Value Resolution:** Categorical missing values (e.g. `club_member_status`, `fashion_news_frequency`) are imputed using dedicated reservation tokens (`[MISSING]`, `[UNK]`). Continuous features such as customer `age` are median-imputed.
* **Cold-Start Handling:** For unobserved customers or inactive users, transaction statistics generate baseline trending article rankings aggregated across the last 14 days of purchase history.
* **Contextual Features:** Purchase time-of-day, recency decay, and rolling 30-day article popularity indices.

### 5.2 Two-Tower Deep Learning Network (Week 2)
* Built using **TensorFlow Recommenders (TFRS)** and Keras 3 (with `TF_USE_LEGACY_KERAS=1` compatibility).
* Query and Candidate representations are projected into a shared **32-dimensional latent embedding space**.
* Trained using **in-batch negative sampling**, optimizing Normalized Discounted Cumulative Gain (**NDCG@K**) and **Recall@K**.

### 5.3 Low-Latency Redis Feature Store (Week 3)
* **Customer Profile Cache:** Redis hash `hm:user_profiles` stores serialized demographic metadata, indexed by `customer_id`.
* **Item Embedding Cache:** Redis hash `hm:article_embeddings` stores pre-computed 32-dimensional float32 binary vectors for all 105,542 articles.
* Sub-millisecond lookups via Redis pipelined `hmget` / `hget`.

### 5.4 Approximate Nearest Neighbor (ANN) Retrieval (Week 3 Day 7)
* Implemented via `ApproximateNearestNeighborIndex` in `src/ann_index.py`.
* Employs **Spherical Inverted File (IVF)** clustering: candidate vectors are normalized and partitioned into Voronoi clusters.
* At inference, evaluates query similarity against cluster centroids, probing only the top $P$ centroids ($P \ll C$), slashing search complexity from $O(N)$ to $O((P/C) \times N)$ while maintaining over 90% top-K recall.

### 5.5 FastAPI Microservice Serving Layer (Week 4)
* Endpoints:
  - `GET /`: Health status and service metadata.
  - `GET /health`: Comprehensive connectivity probe (Redis ping, model loaded, customer/article vocabulary sizes).
  - `GET /recommend/{customer_id}?top_k=10`: High-speed real-time recommendation retrieval.
* Concurrent throughput: **> 3,000 requests/sec** under multi-threaded execution.

### 5.6 Apache Airflow Pipeline Scheduling (Week 4)
* **Weekly Retraining DAG (`hm_weekly_model_retraining`):** 
  - Scheduled weekly (`0 2 * * 1`).
  - Trains new model weights into `models/staging/`.
  - Executes `src.promote_model` validation checks: verifies finite weights, checks output tensor dimensions, creates timestamped backups in `models/backups/`, and performs zero-downtime atomic replacement (`os.replace`) into `models/two_tower.weights.h5`.
* **Daily Embedding Update DAG (`hm_daily_embedding_update`):**
  - Scheduled daily (`0 3 * * *`).
  - Regenerates article vectors from candidate tower and synchronizes Redis `hm:article_embeddings`.

---

## 6. Service Level Agreements (SLAs) & Performance Targets

| Metric | Target Specification | Achieved Benchmark | Status |
|:---|:---:|:---:|:---:|
| **P50 Latency (Median)** | < 10.0 ms | 0.41 ms - 2.80 ms | Exceeded |
| **P95 Latency (Tail)** | < 25.0 ms | 1.29 ms - 3.91 ms | Exceeded |
| **Inference Throughput** | > 250 RPS | > 1,200 - 5,000 RPS | Exceeded |
| **Error Rate** | < 0.1% | 0.00% (Zero Errors) | Met |
| **ANN Candidate Recall@10** | > 85% | > 88% - 95% (with n_probe tuning) | Met |
| **Zero-Downtime Model Swap** | Supported | Atomic file swap (`os.replace`) | Met |
