
import json
from pathlib import Path

import tensorflow as tf

from src.models.two_tower_neural_network import HMTTwoTowerModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "two_tower"

EMBEDDING_DIM = 32


def load_recommendation_model():
    """Load the trained Two-Tower model and its vocabularies."""

    # Load saved vocabularies
    with open(MODEL_DIR / "customer_vocabulary.json", encoding="utf-8") as file:
        customer_ids = json.load(file)

    with open(MODEL_DIR / "article_vocabulary.json", encoding="utf-8") as file:
        article_ids = json.load(file)

    if not customer_ids or not article_ids:
        raise ValueError("Model vocabularies are empty.")

    # Reconstruct the candidate dataset
    candidates_ds = tf.data.Dataset.from_tensor_slices(article_ids)

    # Reconstruct the original model architecture
    model = HMTTwoTowerModel(
        unique_customer_ids=customer_ids,
        unique_article_ids=article_ids,
        parsed_candidates_ds=candidates_ds,
        embedding_dim=EMBEDDING_DIM,
    )

    # Build both towers before loading weights
    model.query_model({
        "customer_id": tf.constant([customer_ids[0]])
    })

    model.candidate_model(
        tf.constant([article_ids[0]])
    )

    # Load trained weights
    weights_path = MODEL_DIR / "two_tower.weights.h5"

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    model.load_weights(weights_path)

    # Generate embeddings for all available articles
    candidate_embeddings = model.candidate_model(
        tf.constant(article_ids)
    )

    print("Trained recommendation model loaded successfully!")
    print(f"Customers: {len(customer_ids)}")
    print(f"Articles: {len(article_ids)}")
    print(f"Candidate embedding shape: {candidate_embeddings.shape}")

    return model, customer_ids, article_ids, candidate_embeddings
