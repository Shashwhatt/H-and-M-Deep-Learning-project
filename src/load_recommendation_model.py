
import json
from pathlib import Path

import tensorflow as tf

from src.models.two_tower_neural_network import HMTTwoTowerModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "two_tower"

EMBEDDING_DIM = 32


def load_recommendation_model():
    """Load the trained context-aware Two-Tower model."""

    # Load saved vocabularies.
    with (MODEL_DIR / "customer_vocabulary.json").open(
        encoding="utf-8"
    ) as file:
        customer_ids = json.load(file)

    with (MODEL_DIR / "article_vocabulary.json").open(
        encoding="utf-8"
    ) as file:
        article_ids = json.load(file)

    with (MODEL_DIR / "club_member_status_vocabulary.json").open(
        encoding="utf-8"
    ) as file:
        club_status_vocabulary = json.load(file)

    with (MODEL_DIR / "fashion_news_frequency_vocabulary.json").open(
        encoding="utf-8"
    ) as file:
        fashion_frequency_vocabulary = json.load(file)

    if not customer_ids or not article_ids:
        raise ValueError("Model vocabularies are empty.")

    if not club_status_vocabulary or not fashion_frequency_vocabulary:
        raise ValueError("Customer feature vocabularies are empty.")

    # Reconstruct the candidate dataset.
    candidates_ds = tf.data.Dataset.from_tensor_slices(article_ids)

    # Reconstruct the context-aware model architecture.
    model = HMTTwoTowerModel(
        unique_customer_ids=customer_ids,
        unique_article_ids=article_ids,
        parsed_candidates_ds=candidates_ds,
        club_status_vocab_size=len(club_status_vocabulary),
        fashion_frequency_vocab_size=len(fashion_frequency_vocabulary),
        embedding_dim=EMBEDDING_DIM,
    )

    # Build the query tower with all required customer features.
    model.query_model({
        "customer_id": tf.constant([customer_ids[0]]),
        "age": tf.constant([31.0], dtype=tf.float32),
        "age_missing": tf.constant([0], dtype=tf.int64),
        "club_member_status_id": tf.constant([0], dtype=tf.int64),
        "fashion_news_frequency_id": tf.constant([0], dtype=tf.int64),
    })

    # Build the candidate tower.
    model.candidate_model(tf.constant([article_ids[0]]))

    # Load trained weights.
    weights_path = MODEL_DIR / "two_tower.weights.h5"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}"
        )

    model.load_weights(weights_path)

    # Generate embeddings for all available articles.
    candidate_embeddings = model.candidate_model(
        tf.constant(article_ids)
    )

    print("Context-aware recommendation model loaded successfully!")
    print(f"Customers: {len(customer_ids)}")
    print(f"Articles: {len(article_ids)}")
    print(f"Club status categories: {len(club_status_vocabulary)}")
    print(
        "Fashion frequency categories: "
        f"{len(fashion_frequency_vocabulary)}"
    )
    print(f"Candidate embedding shape: {candidate_embeddings.shape}")

    return model, customer_ids, article_ids, candidate_embeddings
