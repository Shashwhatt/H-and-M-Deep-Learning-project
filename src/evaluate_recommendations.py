
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.models.two_tower_neural_network import (
    HMQueryTower,
    HMCandidateTower,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "train_positive_interactions.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "validation_positive_interactions.csv"
)


def recall_at_k(actual, recommended, k):
    """Calculate Recall@K for one customer."""

    relevant = set(actual)
    predicted = set(recommended[:k])

    if not relevant:
        return 0.0

    return len(relevant & predicted) / len(relevant)


def ndcg_at_k(actual, recommended, k):
    """Calculate NDCG@K for one customer."""

    relevant = set(actual)

    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, article in enumerate(recommended[:k])
        if article in relevant
    )

    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1.0 / np.log2(rank + 2)
        for rank in range(ideal_hits)
    )

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommendations(
    model,
    train_df,
    validation_df,
    candidate_ids
):
    """Evaluate Recall@K and NDCG@K."""

    results = []
        # Get articles already purchased by each customer
    purchased_articles = (
        train_df.groupby("customer_id")["article_id"]
        .apply(set)
        .to_dict()
    )

    for customer_id, group in validation_df.groupby("customer_id"):

        actual = set(group["article_id"])

        # Exclude articles already purchased during training
        previously_purchased = purchased_articles.get(
            customer_id, set()
        )

        # Evaluate only articles not already purchased
        actual = actual - previously_purchased

        available_candidates = [
            article_id
            for article_id in candidate_ids
            if article_id not in previously_purchased
        ]

        # Skip customers with no remaining validation targets
        if not actual or not available_candidates:
            continue

        customer_input = tf.constant([customer_id])

        user_embedding = model.query_model(
            {"customer_id": customer_input}
        )

        candidate_tensor = tf.constant(available_candidates)

        item_embeddings = model.candidate_model(candidate_tensor)

        scores = tf.linalg.matmul(
            user_embedding,
            item_embeddings,
            transpose_b=True,
        )

        scores = scores.numpy()[0]

        top_indices = np.argsort(scores)[::-1][:100]

        recommendations = [
            available_candidates[index]
            for index in top_indices
        ]

        results.append({
            "customer_id": customer_id,
            "Recall@10": recall_at_k(actual, recommendations, 10),
            "Recall@50": recall_at_k(actual, recommendations, 50),
            "Recall@100": recall_at_k(actual, recommendations, 100),
            "NDCG@10": ndcg_at_k(actual, recommendations, 10),
            "NDCG@50": ndcg_at_k(actual, recommendations, 50),
            "NDCG@100": ndcg_at_k(actual, recommendations, 100),
        })

    return pd.DataFrame(results)


def main():
    """Load the trained model and evaluate recommendations."""

    from src.train_two_tower import (
        load_interactions,
        prepare_datasets,
        EMBEDDING_DIM,
        MODEL_DIR,
    )
    from src.models.two_tower_neural_network import HMTTwoTowerModel

    print("===== H&M RECOMMENDATION EVALUATION =====")

    # Load the same data used during training
    train_df, validation_df = load_interactions()

    (
        train_ds,
        validation_ds,
        candidates_ds,
        unique_customer_ids,
        unique_article_ids,
    ) = prepare_datasets(train_df, validation_df)

    # Reconstruct the original model
    model = HMTTwoTowerModel(
        unique_customer_ids=unique_customer_ids,
        unique_article_ids=unique_article_ids,
        parsed_candidates_ds=candidates_ds,
        embedding_dim=EMBEDDING_DIM,
    )

    # Build model layers before loading saved weights
    model.query_model({
        "customer_id": tf.constant([unique_customer_ids[0]])
    })

    model.candidate_model(
        tf.constant([unique_article_ids[0]])
    )

    # Load saved weights
    weights_path = MODEL_DIR / "two_tower.weights.h5"
    model.load_weights(weights_path)

    print("\nTrained model weights loaded successfully!")

    # Keep only validation articles represented in training
    known_articles = set(train_df["article_id"])
    known_validation = validation_df[
        validation_df["article_id"].isin(known_articles)
    ].copy()

    unseen_count = len(validation_df) - len(known_validation)

    print(f"Validation interactions: {len(validation_df)}")
    print(f"Known-article interactions: {len(known_validation)}")
    print(f"Unseen-article interactions excluded: {unseen_count}")


    # Evaluate recommendations
    results = evaluate_recommendations(
        model,
        train_df,
        known_validation,
        sorted(train_df["article_id"].unique()),
    )

    print("\n===== EVALUATION RESULTS =====")
    print(results.drop(columns=["customer_id"]).mean().to_string())

    # Save evaluation results
    output_path = MODEL_DIR / "evaluation_results.csv"
    results.to_csv(output_path, index=False)

    print(f"\nEvaluation results saved to: {output_path}")



if __name__ == "__main__":
    main()
