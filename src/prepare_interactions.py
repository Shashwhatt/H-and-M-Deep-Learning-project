
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"
PROCESSED_DIR = DATA_DIR / "processed"


def generate_sample_interactions(
    customers_path=None,
    articles_path=None,
    interactions_per_user=5,
    random_state=42,
):
    """Generate synthetic interactions for development and testing only."""

    customers_path = customers_path or SAMPLE_DIR / "customers_sample.csv"
    articles_path = articles_path or SAMPLE_DIR / "articles_sample.csv"

    customers = pd.read_csv(customers_path, dtype=str)
    articles = pd.read_csv(articles_path, dtype=str)

    if "customer_id" not in customers.columns:
        raise ValueError("Customers file must contain customer_id")

    if "article_id" not in articles.columns:
        raise ValueError("Articles file must contain article_id")

    customer_ids = customers["customer_id"].dropna().unique()
    article_ids = articles["article_id"].dropna().unique()

    if len(customer_ids) == 0 or len(article_ids) == 0:
        raise ValueError("Customer and article datasets must not be empty")

    if interactions_per_user < 1:
        raise ValueError("interactions_per_user must be at least 1")

    rng = np.random.default_rng(random_state)

    rows = []

    for customer_id in customer_ids:
        number_of_items = min(interactions_per_user, len(article_ids))

        selected_articles = rng.choice(
            article_ids,
            size=number_of_items,
            replace=False,
        )

        for article_id in selected_articles:
            rows.append({
                "customer_id": customer_id,
                "article_id": article_id,
                "label": 1,
            })

    return pd.DataFrame(rows)


def split_interactions(interactions, validation_size=0.2, random_state=42):
    """Split interactions into training and validation datasets."""

    required_columns = {"customer_id", "article_id", "label"}

    if not required_columns.issubset(interactions.columns):
        raise ValueError(f"Interactions must contain {required_columns}")

    if interactions.empty:
        raise ValueError("Interactions dataset cannot be empty")

    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")

    train, validation = train_test_split(
        interactions,
        test_size=validation_size,
        random_state=random_state,
    )

    return train.reset_index(drop=True), validation.reset_index(drop=True)



def generate_negative_samples(
    interactions,
    all_article_ids,
    negatives_per_positive=1,
    random_state=42,
    all_positive_interactions=None,
):
    """Generate negative customer-article pairs.

    Excludes all known positive articles for each customer.
    """

    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be at least 1")

    required_columns = {"customer_id", "article_id", "label"}

    if not required_columns.issubset(interactions.columns):
        raise ValueError(f"Interactions must contain {required_columns}")

    if interactions.empty:
        raise ValueError("Interactions dataset cannot be empty")

    if not interactions["label"].eq(1).all():
        raise ValueError("Input interactions must contain only positive labels")

    article_ids = np.asarray(list(dict.fromkeys(all_article_ids)))

    if len(article_ids) == 0:
        raise ValueError("Article list cannot be empty")

    reference_interactions = (
        all_positive_interactions
        if all_positive_interactions is not None
        else interactions
    )

    positive_by_user = (
        reference_interactions.groupby("customer_id")["article_id"]
        .apply(set)
        .to_dict()
    )

    rows = []
    rng = np.random.default_rng(random_state)

    for customer_id, positive_articles in positive_by_user.items():
        available_articles = np.array([
            article_id
            for article_id in article_ids
            if article_id not in positive_articles
        ])

        if len(available_articles) == 0:
            raise ValueError(
                f"No negative articles available for customer {customer_id}"
            )

        number_of_negatives = min(
            negatives_per_positive * len(
                interactions[interactions["customer_id"] == customer_id]
            ),
            len(available_articles),
        )

        selected_negatives = rng.choice(
            available_articles,
            size=number_of_negatives,
            replace=False,
        )

        for article_id in selected_negatives:
            rows.append({
                "customer_id": customer_id,
                "article_id": article_id,
                "label": 0,
            })

    return pd.DataFrame(rows)



def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating development-only sample interactions...")

    positives = generate_sample_interactions()

    train, validation = split_interactions(positives)

    articles = pd.read_csv(
        SAMPLE_DIR / "articles_sample.csv",
        dtype=str,
    )

    negative_samples = generate_negative_samples(
        train,
        articles["article_id"].dropna().unique(),
        negatives_per_positive=1,
        all_positive_interactions=positives,
    )

    train.to_csv(PROCESSED_DIR / "train_positive_interactions.csv", index=False)
    validation.to_csv(
        PROCESSED_DIR / "validation_positive_interactions.csv",
        index=False,
    )
    negative_samples.to_csv(
        PROCESSED_DIR / "train_negative_interactions.csv",
        index=False,
    )

    print(f"Positive interactions: {len(positives):,}")
    print(f"Training positives: {len(train):,}")
    print(f"Validation positives: {len(validation):,}")
    print(f"Training negatives: {len(negative_samples):,}")
    print("Development interaction files saved successfully.")


if __name__ == "__main__":
    main()
