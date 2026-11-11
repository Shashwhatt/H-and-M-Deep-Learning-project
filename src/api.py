
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
from pathlib import Path

import tensorflow as tf
from fastapi import FastAPI, HTTPException, Query

from src.two_tower_model import HMTwoTowerModel


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
VOCAB_DIR = BASE_DIR / "models" / "serving_vocabularies"
MODEL_PATH = BASE_DIR / "models" / "two_tower.weights.h5"

CUSTOMER_VOCAB_PATH = VOCAB_DIR / "customer_id_vocabulary.json"
ARTICLE_VOCAB_PATH = VOCAB_DIR / "article_id_vocabulary.json"

EMBEDDING_DIM = 32
BATCH_SIZE = 4096

app = FastAPI(
    title="H&M Recommendation API",
    description="Context-Aware Neural Recommendation Engine",
    version="1.0.0"
)


# Load vocabularies
print("Loading vocabularies...")

with open(CUSTOMER_VOCAB_PATH, "r", encoding="utf-8") as file:
    customer_ids = json.load(file)

with open(ARTICLE_VOCAB_PATH, "r", encoding="utf-8") as file:
    article_ids = json.load(file)

customer_id_set = set(customer_ids)

print("Customers:", len(customer_ids))
print("Articles:", len(article_ids))


# Initialize and load the trained model
print("Loading trained Two-Tower model...")

model = HMTwoTowerModel(
    customer_ids,
    article_ids,
    embedding_dim=EMBEDDING_DIM
)

sample = {
    "customer_id": tf.constant([customer_ids[0]]),
    "article_id": tf.constant([article_ids[0]])
}

model(sample)
model.load_weights(str(MODEL_PATH))

print("Trained model loaded successfully!")


# Precompute article embeddings
print("Generating article embeddings...")

candidate_embeddings = []

for start in range(0, len(article_ids), BATCH_SIZE):
    batch_ids = article_ids[start:start + BATCH_SIZE]

    embeddings = model.candidate_tower({
        "article_id": tf.constant(batch_ids)
    })

    candidate_embeddings.append(embeddings)

candidate_embeddings = tf.concat(candidate_embeddings, axis=0)

print("Article embeddings ready:", candidate_embeddings.shape)


@app.get("/")
def home():
    return {
        "message": "H&M Recommendation API is running",
        "model": "Two-Tower Neural Recommendation Engine"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "customers": len(customer_ids),
        "articles": len(article_ids)
    }


@app.get("/recommend/{customer_id}")
def recommend(
    customer_id: str,
    top_k: int = Query(default=10, ge=1, le=50)
):
    if customer_id not in customer_id_set:
        raise HTTPException(
            status_code=404,
            detail="Customer ID not found in model vocabulary"
        )

    query_embedding = model.query_tower({
        "customer_id": tf.constant([customer_id])
    })

    scores = tf.linalg.matmul(
        query_embedding,
        candidate_embeddings,
        transpose_b=True
    )

    actual_k = min(top_k, len(article_ids))

    top_results = tf.math.top_k(
        scores[0],
        k=actual_k
    )

    indices = top_results.indices.numpy().tolist()
    values = top_results.values.numpy().tolist()

    recommendations = []

    for index, score in zip(indices, values):
        recommendations.append({
            "article_id": article_ids[index],
            "score": round(float(score), 6)
        })

    return {
        "customer_id": customer_id,
        "recommendations": recommendations
    }
