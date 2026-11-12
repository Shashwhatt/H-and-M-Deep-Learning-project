import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
from pathlib import Path

import numpy as np
import redis
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Query

from src.two_tower_model import HMTwoTowerModel, load_serving_weights


BASE_DIR = Path(__file__).resolve().parent.parent

VOCAB_DIR = BASE_DIR / "models" / "serving_vocabularies"
MODEL_PATH = BASE_DIR / "models" / "two_tower.weights.h5"

CUSTOMER_VOCAB_PATH = VOCAB_DIR / "customer_id_vocabulary.json"
ARTICLE_VOCAB_PATH = VOCAB_DIR / "article_id_vocabulary.json"

REDIS_HOST = "localhost"
REDIS_PORT = 6379

CUSTOMER_HASH = "hm:user_profiles"
ARTICLE_HASH = "hm:article_embeddings"

EMBEDDING_DIM = 32
REDIS_BATCH_SIZE = 2000

app = FastAPI(
    title="H&M Recommendation API",
    description="Redis-powered Two-Tower Recommendation Engine",
    version="2.0.0"
)


print("Connecting to Redis...")

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=False,
    socket_connect_timeout=10,
    socket_timeout=30
)

redis_client.ping()
print("Redis connected successfully!")


print("Loading model vocabularies...")

with open(CUSTOMER_VOCAB_PATH, "r", encoding="utf-8") as file:
    customer_ids = json.load(file)

with open(ARTICLE_VOCAB_PATH, "r", encoding="utf-8") as file:
    article_ids = json.load(file)

customer_id_set = set(customer_ids)

print("Customers:", len(customer_ids))
print("Articles:", len(article_ids))


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
load_serving_weights(model, MODEL_PATH)

print("Trained model loaded successfully!")


def load_article_embeddings():
    print("Loading article embeddings from Redis...")

    if not redis_client.exists(ARTICLE_HASH):
        raise RuntimeError(
            "Article embeddings are missing from Redis. "
            "Run src.redis_feature_store first."
        )

    all_embeddings = []

    for start in range(0, len(article_ids), REDIS_BATCH_SIZE):
        batch_ids = article_ids[start:start + REDIS_BATCH_SIZE]

        values = redis_client.hmget(
            ARTICLE_HASH,
            batch_ids
        )

        for article_id, value in zip(batch_ids, values):
            if value is None:
                raise RuntimeError(
                    f"Missing Redis embedding for article: {article_id}"
                )

            embedding = np.frombuffer(
                value,
                dtype=np.float32
            )

            if embedding.size != EMBEDDING_DIM:
                raise RuntimeError(
                    f"Invalid embedding dimension for {article_id}"
                )

            all_embeddings.append(embedding)

        print(
            f"Loaded article embeddings: "
            f"{min(start + len(batch_ids), len(article_ids)):,}"
        )

    embeddings = np.stack(all_embeddings)

    print("Redis article embeddings ready:", embeddings.shape)

    return embeddings


candidate_embeddings = load_article_embeddings()


@app.get("/")
def home():
    return {
        "message": "H&M Recommendation API is running",
        "model": "Two-Tower Neural Recommendation Engine",
        "feature_store": "Redis"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "redis_connected": bool(redis_client.ping()),
        "customers": len(customer_ids),
        "articles": len(article_ids),
        "article_embeddings_shape": list(candidate_embeddings.shape)
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

    # Fetch the customer profile from Redis.
    profile_data = redis_client.hget(
        CUSTOMER_HASH,
        customer_id
    )

    if profile_data is None:
        raise HTTPException(
            status_code=404,
            detail="Customer profile not found in Redis"
        )

    try:
        customer_profile = json.loads(profile_data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(
            status_code=500,
            detail="Invalid customer profile stored in Redis"
        )

    # Generate the customer query embedding.
    query_embedding = model.query_tower({
        "customer_id": tf.constant([customer_id])
    }).numpy()[0]

    # Rank Redis article embeddings using dot-product similarity.
    scores = np.dot(
        candidate_embeddings,
        query_embedding
    )

    actual_k = min(top_k, len(article_ids))

    top_indices = np.argpartition(
        scores,
        -actual_k
    )[-actual_k:]

    top_indices = top_indices[
        np.argsort(scores[top_indices])[::-1]
    ]

    recommendations = []

    for index in top_indices:
        recommendations.append({
            "article_id": article_ids[int(index)],
            "score": round(float(scores[index]), 6)
        })

    return {
        "customer_id": customer_id,
        "profile_available": customer_profile is not None,
        "recommendations": recommendations
    }
