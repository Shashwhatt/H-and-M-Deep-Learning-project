
import json
import os

# Set before importing TensorFlow.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf

from src.models.two_tower_neural_network import HMTTwoTowerModel
from src.prepare_user_features import prepare_user_features
from src.user_feature_vocabularies import USER_CATEGORICAL_FEATURES

from src.train_two_tower import (
    load_interactions,
    MODEL_DIR,
    EMBEDDING_DIM,
)


def recall_at_k(actual, recommended, k):
    """Calculate Recall@K."""
    relevant = set(actual)
    predicted = set(recommended[:k])

    if not relevant:
        return 0.0

    return len(relevant & predicted) / len(relevant)


def ndcg_at_k(actual, recommended, k):
    """Calculate NDCG@K."""
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


def load_model_artifacts():
    """Load preprocessing artifacts saved during training."""

    vocabularies = {}

    for feature in USER_CATEGORICAL_FEATURES:
        path = MODEL_DIR / f"{feature}_vocabulary.json"

        with path.open("r", encoding="utf-8") as file:
            vocabularies[feature] = json.load(file)

    age_path = MODEL_DIR / "age_median.json"

    with age_path.open("r", encoding="utf-8") as file:
        age_config = json.load(file)

    if "age_median" not in age_config:
        raise ValueError("Invalid saved age configuration.")

    return vocabularies, age_config["age_median"]


def build_query_inputs(customer, customer_id=None):
    """Build all five inputs required by the context-aware query tower."""

    if customer_id is None:
        customer_id = customer["customer_id"]

    return {
        "customer_id": tf.constant(
            [str(customer_id)], dtype=tf.string
        ),

        "age": tf.constant(
            [float(customer["age"])], dtype=tf.float32
        ),

        "age_missing": tf.constant(
            [int(customer["age_missing"])], dtype=tf.int64
        ),

        "club_member_status_id": tf.constant(
            [int(customer["club_member_status_id"])],
            dtype=tf.int64
        ),

        "fashion_news_frequency_id": tf.constant(
            [int(customer["fashion_news_frequency_id"])],
            dtype=tf.int64
        ),
    }


def evaluate_recommendations(
    model,
    train_df,
    validation_df,
    customer_features,
    candidate_ids,
):
    """Evaluate Recall@K and NDCG@K."""

    results = []

    purchased_articles = (
        train_df.groupby("customer_id")["article_id"]
        .apply(set)
        .to_dict()
    )

    customer_feature_map = customer_features.set_index("customer_id")

    candidate_ids_array = np.array(candidate_ids, dtype=str)

    print("\nCalculating candidate embeddings...")

    candidate_embeddings = model.candidate_model(
        tf.constant(candidate_ids, dtype=tf.string)
    ).numpy()

    skipped_customers = 0

    print("Evaluating customers...")

    for customer_id, group in validation_df.groupby("customer_id"):

        if customer_id not in customer_feature_map.index:
            skipped_customers += 1
            continue

        actual = set(group["article_id"])

        previously_purchased = purchased_articles.get(
            customer_id, set()
        )

        actual = actual - previously_purchased

        if not actual:
            skipped_customers += 1
            continue

        eligible_mask = ~np.isin(
            candidate_ids_array,
            list(previously_purchased),
        )

        eligible_ids = candidate_ids_array[eligible_mask]

        eligible_embeddings = candidate_embeddings[eligible_mask]

        if len(eligible_ids) == 0:
            skipped_customers += 1
            continue

        customer = customer_feature_map.loc[customer_id]

        user_embedding = model.query_model(
            build_query_inputs(customer, customer_id)
        ).numpy()

        scores = np.matmul(
            user_embedding,
            eligible_embeddings.T,
        )[0]

        top_indices = np.argsort(scores)[::-1][:100]

        recommendations = eligible_ids[top_indices].tolist()

        results.append({
            "customer_id": customer_id,
            "Recall@10": recall_at_k(actual, recommendations, 10),
            "Recall@50": recall_at_k(actual, recommendations, 50),
            "Recall@100": recall_at_k(actual, recommendations, 100),
            "NDCG@10": ndcg_at_k(actual, recommendations, 10),
            "NDCG@50": ndcg_at_k(actual, recommendations, 50),
            "NDCG@100": ndcg_at_k(actual, recommendations, 100),
        })

    print(f"\nEvaluated customers: {len(results):,}")
    print(f"Skipped customers: {skipped_customers:,}")

    return pd.DataFrame(results)


def main():
    """Load and evaluate the trained context-aware model."""

    print("===== H&M CONTEXT-AWARE RECOMMENDATION EVALUATION =====")

    # Step 1: Load interaction data and customer metadata.

    train_df, validation_df, customers = load_interactions()

    # Step 2: Load saved preprocessing artifacts.

    vocabularies, age_median = load_model_artifacts()

    customer_features, _ = prepare_user_features(
        customers,
        vocabularies=vocabularies,
        age_median=age_median,
    )

    print("\nCustomer features prepared successfully.")

    # Step 3: Verify customer metadata coverage.

    metadata_ids = set(customer_features["customer_id"])

    for name, dataframe in [
        ("Training", train_df),
        ("Validation", validation_df),
    ]:

        missing = set(dataframe["customer_id"]) - metadata_ids

        if missing:
            raise ValueError(
                f"{name} data contains {len(missing)} "
                "customers without metadata."
            )

    # Step 4: Reconstruct model architecture.

    unique_customer_ids = sorted(
        set(train_df["customer_id"])
        | set(validation_df["customer_id"])
    )

    unique_article_ids = sorted(
        set(train_df["article_id"])
        | set(validation_df["article_id"])
    )

    candidates_ds = tf.data.Dataset.from_tensor_slices(
        unique_article_ids
    )

    club_vocab_size = len(
        vocabularies["club_member_status"]
    )

    fashion_vocab_size = len(
        vocabularies["fashion_news_frequency"]
    )

    model = HMTTwoTowerModel(
        unique_customer_ids=unique_customer_ids,
        unique_article_ids=unique_article_ids,
        parsed_candidates_ds=candidates_ds,
        club_status_vocab_size=club_vocab_size,
        fashion_frequency_vocab_size=fashion_vocab_size,
        embedding_dim=EMBEDDING_DIM,
    )

    # Build model layers before loading weights.

    model.query_model(
        build_query_inputs(customer_features.iloc[0])
    )

    model.candidate_model(
        tf.constant(
            [unique_article_ids[0]],
            dtype=tf.string,
        )
    )

    # Step 5: Load trained model weights.

    weights_path = MODEL_DIR / "two_tower.weights.h5"

    model.load_weights(weights_path)

    print("\nContext-aware model weights loaded successfully!")

    # Step 6: Prepare evaluation candidates.

    known_articles = set(train_df["article_id"])

    known_validation = validation_df[
        validation_df["article_id"].isin(known_articles)
    ].copy()

    unseen_count = len(validation_df) - len(known_validation)

    print(f"\nValidation interactions: {len(validation_df):,}")
    print(f"Known-article interactions: {len(known_validation):,}")
    print(f"Unseen-article interactions excluded: {unseen_count:,}")

    candidate_ids = sorted(known_articles)

    if not candidate_ids:
        raise ValueError("No training candidate articles found.")

    # Step 7: Evaluate recommendations.

    results = evaluate_recommendations(
        model=model,
        train_df=train_df,
        validation_df=known_validation,
        customer_features=customer_features,
        candidate_ids=candidate_ids,
    )

    if results.empty:
        raise ValueError(
            "No customers could be evaluated. "
            "Check validation data and customer metadata."
        )

    # Step 8: Display aggregate evaluation metrics.

    metric_columns = [
        "Recall@10",
        "Recall@50",
        "Recall@100",
        "NDCG@10",
        "NDCG@50",
        "NDCG@100",
    ]

    print("\n===== CONTEXT-AWARE EVALUATION RESULTS =====")

    print(
        results[metric_columns].mean().to_string()
    )

    # Step 9: Save evaluation results.

    output_path = MODEL_DIR / "evaluation_results.csv"

    results.to_csv(output_path, index=False)

    print(f"\nEvaluation results saved to: {output_path}")

    print("\n===== EVALUATION COMPLETED =====")


if __name__ == "__main__":
    main()
