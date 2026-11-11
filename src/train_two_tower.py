
import os

# Set before importing TensorFlow.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from src.models.two_tower_neural_network import HMTTwoTowerModel
from src.prepare_user_features import prepare_training_features
from src.user_feature_vocabularies import (
    OUTPUT_DIR,
    USER_CATEGORICAL_FEATURES,
)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "two_tower"

TRAIN_PATH = DATA_DIR / "train_positive_interactions.csv"
VALIDATION_PATH = DATA_DIR / "validation_positive_interactions.csv"
CUSTOMERS_PATH = PROJECT_ROOT / "data" / "sample" / "customers_sample.csv"

# Training configuration
EMBEDDING_DIM = 32
BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 0.001

CUSTOMER_FEATURE_COLUMNS = [
    "customer_id",
    "age",
    "age_missing",
    "club_member_status_id",
    "fashion_news_frequency_id",
]


def load_interactions():
    """Load and validate training and validation interactions."""

    for path in [TRAIN_PATH, VALIDATION_PATH, CUSTOMERS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")

    train_df = pd.read_csv(
        TRAIN_PATH,
        dtype={"customer_id": str, "article_id": str},
    )

    validation_df = pd.read_csv(
        VALIDATION_PATH,
        dtype={"customer_id": str, "article_id": str},
    )

    required_columns = {"customer_id", "article_id"}

    for name, df in [
        ("Training", train_df),
        ("Validation", validation_df),
    ]:
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} data is missing columns: {missing}"
            )

        if df[list(required_columns)].isnull().any().any():
            raise ValueError(f"{name} data contains missing IDs.")

    customers = pd.read_csv(
        CUSTOMERS_PATH,
        dtype={"customer_id": str},
    )

    print(f"Training interactions: {len(train_df):,}")
    print(f"Validation interactions: {len(validation_df):,}")
    print(f"Customer metadata records: {len(customers):,}")

    return train_df, validation_df, customers


def prepare_datasets(train_df, validation_df, customers):
    """Join customer metadata and create TensorFlow datasets."""

    print("\nPreparing customer metadata...")

    customer_features, vocabularies = prepare_training_features(
        customers
    )

    # Verify that every interaction customer has metadata.
    metadata_customer_ids = set(customer_features["customer_id"])

    for name, df in [
        ("Training", train_df),
        ("Validation", validation_df),
    ]:
        missing_ids = set(df["customer_id"]) - metadata_customer_ids

        if missing_ids:
            raise ValueError(
                f"{name} data contains {len(missing_ids)} "
                "customers without metadata."
            )

    # Build customer and article ID vocabularies.
    unique_customer_ids = sorted(
        set(train_df["customer_id"]) |
        set(validation_df["customer_id"])
    )

    unique_article_ids = sorted(
        set(train_df["article_id"]) |
        set(validation_df["article_id"])
    )

    def join_customer_features(interactions, name):
        merged = interactions.merge(
            customer_features,
            on="customer_id",
            how="left",
            validate="many_to_one",
        )

        if len(merged) != len(interactions):
            raise ValueError(
                f"{name} interaction count changed after metadata join."
            )

        if merged[CUSTOMER_FEATURE_COLUMNS].isnull().any().any():
            raise ValueError(
                f"{name} dataset contains missing model features."
            )

        return merged

    train_merged = join_customer_features(
        train_df,
        "Training",
    )

    validation_merged = join_customer_features(
        validation_df,
        "Validation",
    )

    def dataframe_to_dataset(df):
        features = {
            "customer_id": df["customer_id"].astype(str).values,
            "article_id": df["article_id"].astype(str).values,
            "age": df["age"].astype("float32").values,
            "age_missing": df["age_missing"].astype("int64").values,
            "club_member_status_id": (
                df["club_member_status_id"].astype("int64").values
            ),
            "fashion_news_frequency_id": (
                df["fashion_news_frequency_id"].astype("int64").values
            ),
        }

        return tf.data.Dataset.from_tensor_slices(features)

    train_ds = dataframe_to_dataset(train_merged)
    validation_ds = dataframe_to_dataset(validation_merged)

    train_ds = (
        train_ds
        .shuffle(len(train_merged), reshuffle_each_iteration=True)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    validation_ds = (
        validation_ds
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    candidates_ds = tf.data.Dataset.from_tensor_slices(
        unique_article_ids
    )

    club_status_vocab_size = len(
        vocabularies["club_member_status"]
    )

    fashion_frequency_vocab_size = len(
        vocabularies["fashion_news_frequency"]
    )

    print(f"Customer vocabulary: {len(unique_customer_ids):,}")
    print(f"Article vocabulary: {len(unique_article_ids):,}")
    print(f"Club status vocabulary: {club_status_vocab_size}")
    print(f"Fashion frequency vocabulary: {fashion_frequency_vocab_size}")
    print(f"Training feature rows: {len(train_merged):,}")
    print(f"Validation feature rows: {len(validation_merged):,}")

    return (
        train_ds,
        validation_ds,
        candidates_ds,
        unique_customer_ids,
        unique_article_ids,
        club_status_vocab_size,
        fashion_frequency_vocab_size,
        vocabularies,
    )


def train_model(
    train_ds,
    validation_ds,
    candidates_ds,
    unique_customer_ids,
    unique_article_ids,
    club_status_vocab_size,
    fashion_frequency_vocab_size,
):
    """Build, compile and train the context-aware Two-Tower model."""

    print("\nBuilding context-aware Two-Tower model...")

    model = HMTTwoTowerModel(
        unique_customer_ids=unique_customer_ids,
        unique_article_ids=unique_article_ids,
        parsed_candidates_ds=candidates_ds,
        club_status_vocab_size=club_status_vocab_size,
        fashion_frequency_vocab_size=fashion_frequency_vocab_size,
        embedding_dim=EMBEDDING_DIM,
    )

    optimizer = tf.keras.optimizers.Adagrad(
        learning_rate=LEARNING_RATE
    )

    model.compile(optimizer=optimizer)

    print("\nStarting model training...")

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCHS,
    )

    print("\nTraining completed!")

    return model, history


def save_training_results(
    model,
    history,
    unique_customer_ids,
    unique_article_ids,
    vocabularies,
):
    """Save model weights, vocabularies and training history."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save model weights.
    weights_path = MODEL_DIR / "two_tower.weights.h5"
    model.save_weights(weights_path)

    # Save customer vocabulary.
    with (MODEL_DIR / "customer_vocabulary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(unique_customer_ids, file, indent=2)

    # Save article vocabulary.
    with (MODEL_DIR / "article_vocabulary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(unique_article_ids, file, indent=2)

    # Save categorical vocabularies with model artifacts.
    for feature, vocabulary in vocabularies.items():
        output_path = MODEL_DIR / f"{feature}_vocabulary.json"

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(vocabulary, file, indent=2)

    # Save training age median.
    age_config_path = OUTPUT_DIR / "age_median.json"

    with age_config_path.open("r", encoding="utf-8") as file:
        age_config = json.load(file)

    with (MODEL_DIR / "age_median.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(age_config, file, indent=2)

    # Save training history.
    history_df = pd.DataFrame(history.history)
    history_path = MODEL_DIR / "training_history.csv"
    history_df.to_csv(history_path, index=False)

    print("\nTraining results saved successfully!")
    print(f"Model weights: {weights_path}")
    print(f"Customer vocabulary: {MODEL_DIR / 'customer_vocabulary.json'}")
    print(f"Article vocabulary: {MODEL_DIR / 'article_vocabulary.json'}")
    print(f"Age configuration: {MODEL_DIR / 'age_median.json'}")
    print(f"Training history: {history_path}")


def main():
    print("===== H&M CONTEXT-AWARE TWO-TOWER TRAINING =====")

    # Step 1: Load interactions and customer metadata.
    train_df, validation_df, customers = load_interactions()

    # Step 2: Prepare metadata-aware TensorFlow datasets.
    (
        train_ds,
        validation_ds,
        candidates_ds,
        unique_customer_ids,
        unique_article_ids,
        club_status_vocab_size,
        fashion_frequency_vocab_size,
        vocabularies,
    ) = prepare_datasets(
        train_df,
        validation_df,
        customers,
    )

    # Step 3: Train model.
    model, history = train_model(
        train_ds,
        validation_ds,
        candidates_ds,
        unique_customer_ids,
        unique_article_ids,
        club_status_vocab_size,
        fashion_frequency_vocab_size,
    )

    # Step 4: Save model and preprocessing artifacts.
    save_training_results(
        model,
        history,
        unique_customer_ids,
        unique_article_ids,
        vocabularies,
    )

    print("\n===== CONTEXT-AWARE TRAINING PIPELINE COMPLETED =====")


if __name__ == "__main__":
    main()
