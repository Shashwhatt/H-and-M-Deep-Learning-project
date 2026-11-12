import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
from pathlib import Path

import numpy as np
import pandas as pd
import redis
import tensorflow as tf

from src.two_tower_model import HMTwoTowerModel, load_serving_weights


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "two_tower.weights.h5"
VOCAB_DIR = BASE_DIR / "models" / "serving_vocabularies"

CUSTOMER_CSV = BASE_DIR / "data" / "processed" / "customers.csv"

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))

CUSTOMER_HASH = "hm:user_profiles"
ARTICLE_HASH = "hm:article_embeddings"

EMBEDDING_DIM = 32
EMBEDDING_BATCH_SIZE = 2048
CUSTOMER_CHUNK_SIZE = 5000
REDIS_BATCH_SIZE = 500


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def connect_redis():
    client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=False,
        socket_connect_timeout=10,
        socket_timeout=30,
    )
    client.ping()
    print("Connected to Redis successfully!")
    return client


def store_customer_profiles(client):
    print("\nLoading customer profiles into Redis...")

    total = 0

    for chunk in pd.read_csv(
        CUSTOMER_CSV,
        usecols=[
            "customer_id",
            "age",
            "club_member_status",
            "fashion_news_frequency",
        ],
        dtype={"customer_id": "string"},
        chunksize=CUSTOMER_CHUNK_SIZE,
    ):
        chunk["customer_id"] = (
            chunk["customer_id"].astype("string").str.strip()
        )

        chunk = chunk.dropna(subset=["customer_id"])
        chunk = chunk[chunk["customer_id"] != ""]

        mapping = {}

        for row in chunk.itertuples(index=False):
            customer_id = str(row.customer_id)

            profile = {
                "age": None if pd.isna(row.age) else str(row.age),
                "club_member_status": (
                    None
                    if pd.isna(row.club_member_status)
                    else str(row.club_member_status).strip()
                ),
                "fashion_news_frequency": (
                    None
                    if pd.isna(row.fashion_news_frequency)
                    else str(row.fashion_news_frequency).strip()
                ),
            }

            mapping[customer_id] = json.dumps(profile).encode("utf-8")

        if mapping:
            client.hset(CUSTOMER_HASH, mapping=mapping)

        total += len(mapping)
        print(f"Customer profiles stored: {total:,}")

    print(f"Finished storing {total:,} customer profiles.")


def store_article_embeddings(client):
    print("\nLoading model vocabularies...")

    customer_ids = load_json(
        VOCAB_DIR / "customer_id_vocabulary.json"
    )
    article_ids = load_json(
        VOCAB_DIR / "article_id_vocabulary.json"
    )

    customer_list = customer_ids
    article_list = article_ids

    print("Customers:", len(customer_list))
    print("Articles:", len(article_list))

    print("\nLoading trained Two-Tower model...")

    model = HMTwoTowerModel(
        customer_list,
        article_list,
        embedding_dim=EMBEDDING_DIM,
    )

    sample = {
        "customer_id": tf.constant([customer_list[0]]),
        "article_id": tf.constant([article_list[0]]),
    }

    model(sample)
    load_serving_weights(model, MODEL_PATH)

    print("Trained model loaded successfully!")
    print("\nGenerating and storing article embeddings...")

    total = 0

    for start in range(0, len(article_list), EMBEDDING_BATCH_SIZE):
        batch_ids = article_list[
            start:start + EMBEDDING_BATCH_SIZE
        ]

        embeddings = model.candidate_tower(
            {"article_id": tf.constant(batch_ids)}
        ).numpy().astype(np.float32)

        for batch_start in range(
            0, len(batch_ids), REDIS_BATCH_SIZE
        ):
            end = batch_start + REDIS_BATCH_SIZE

            batch_mapping = {
                batch_ids[index]: embeddings[index].tobytes()
                for index in range(batch_start, min(end, len(batch_ids)))
            }

            if batch_mapping:
                client.hset(ARTICLE_HASH, mapping=batch_mapping)

            total += len(batch_mapping)

        print(f"Article embeddings stored: {total:,}")

    print(f"Finished storing {total:,} article embeddings.")


def main():
    client = connect_redis()

    if client.exists(CUSTOMER_HASH):
        print("Customer profiles already exist in Redis. Skipping.")
    else:
        store_customer_profiles(client)
    store_article_embeddings(client)

    print("\nRedis feature-store population completed!")
    print("Customer hash:", CUSTOMER_HASH)
    print("Article hash:", ARTICLE_HASH)


if __name__ == "__main__":
    main()


