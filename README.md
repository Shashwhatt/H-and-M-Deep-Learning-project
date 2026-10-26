# Context-Aware Neural Recommendation Engine

A production-grade, deep learning recommendation system developed using the **H&M Personalized Fashion Recommendations** dataset. 

Moving beyond traditional collaborative filtering, this system implements a **Two-Tower Neural Network** architecture (Query Tower for user context and Candidate Tower for article metadata) to deliver hyper-personalized, context-aware recommendations in real time.

---

## Project Overview

* **Domain:** E-commerce Personalization & Deep Learning Recommendation Systems
* **Dataset:** H&M Personalized Fashion Recommendations (Kaggle) — multi-year user purchase histories, detailed article metadata (garment type, color, etc.), and customer demographics.
* **Business Objective:** Increase Click-Through Rates (CTR), Average Order Value (AOV), and user engagement through dynamic candidate retrieval and ranking based on short-term intent and long-term preferences.

---

## Key Features

* **Distributed Feature Engineering:** Employs Apache PySpark to clean raw transaction logs, resolve cold-start issues, and engineer contextual features (purchase recency, time-of-purchase, rolling popularity).
* **Two-Tower Neural Network Architecture:** Built with TensorFlow Recommenders (TFRS) to decouple user query representations from candidate item embeddings for scalable retrieval.
* **Negative Sampling & Ranking Evaluation:** Evaluated against industry-standard recommendation metrics, including **Recall@K** and **NDCG**.
* **Low-Latency Feature Store:** Implements Redis to cache pre-computed user profiles and item vectors, supporting fast candidate retrieval and sub-millisecond lookups.
* **Approximate Nearest Neighbor (ANN) Search:** Enables ultra-fast similarity matching between query vectors and candidate item vectors during inference.
* **Microservice Architecture:** End-to-end recommendation serving powered by a high-throughput FastAPI service.
* **Automated Orchestration:** Apache Airflow DAGs to orchestrate weekly model retraining and daily embedding synchronization.

---

## Tech Stack

* **Programming Language:** Python 3.9+
* **Deep Learning:** TensorFlow, Keras, TensorFlow Recommenders (TFRS)
* **Data Processing & Orchestration:** Apache Spark (PySpark), Apache Airflow
* **Feature Store & In-Memory Cache:** Redis
* **Model Evaluation:** Scikit-learn (Recall@K, NDCG)
* **API & Serving:** FastAPI, Uvicorn, Pydantic

---

## Repository Structure

```text
hm-recommendation-engine/
├── airflow/            # Airflow DAGs for model retraining & embedding updates
│   └── .gitkeep
├── data/               # Raw and processed Parquet/CSV data (git-ignored)
│   ├── processed/
│   └── raw/
├── models/             # Saved model checkpoints and exported weights
│   └── .gitkeep
├── notebooks/          # Exploratory Data Analysis (EDA) and prototyping
├── src/
│   ├── api/            # FastAPI recommendation endpoints
│   ├── features/       # PySpark feature engineering pipelines
│   ├── models/         # Two-Tower neural network architectures (TFRS)
│   └── utils/          # Redis feature store connectors and helper scripts
├── tests/              # Unit and integration tests
├── .env.example        # Template for environment configuration
├── .gitignore          # Git exclusions for large datasets, models, and caches
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
