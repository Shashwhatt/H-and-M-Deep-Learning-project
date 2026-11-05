
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.models.two_tower_neural_network import HMTTwoTowerModel


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "two_tower"

TRAIN_PATH = DATA_DIR / "train_positive_interactions.csv"
VALIDATION_PATH = DATA_DIR / "validation_positive_interactions.csv"

# Training configuration
EMBEDDING_DIM = 32
BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 0.001

def load_interactions():
    """Load and validate training and validation interactions."""

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(f"Validation file not found: {VALIDATION_PATH}")

    train_df = pd.read_csv(
        TRAIN_PATH,
        dtype={"customer_id": str, "article_id": str}
    )

    validation_df = pd.read_csv(
        VALIDATION_PATH,
        dtype={"customer_id": str, "article_id": str}
    )

    required_columns = {"customer_id", "article_id"}

    for name, df in [
        ("Training", train_df),
        ("Validation", validation_df)
    ]:
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} data is missing columns: {missing}"
            )

        if df[list(required_columns)].isnull().any().any():
            raise ValueError(f"{name} data contains missing IDs.")

    print(f"Training interactions: {len(train_df):,}")
    print(f"Validation interactions: {len(validation_df):,}")

    return train_df, validation_df

def prepare_datasets(train_df, validation_df):
    """Create TensorFlow datasets and ID vocabularies."""

    # Build vocabularies from training data
    unique_customer_ids = sorted(
        train_df["customer_id"].unique().tolist()
    )

    unique_article_ids = sorted(
        train_df["article_id"].unique().tolist()
    )

    # Ensure validation IDs are represented in the model vocabularies
    # so the validation pipeline can handle them.
    unique_customer_ids = sorted(
        set(unique_customer_ids) |
        set(validation_df["customer_id"].unique())
    )

    unique_article_ids = sorted(
        set(unique_article_ids) |
        set(validation_df["article_id"].unique())
    )

    # Prepare candidate dataset for retrieval evaluation
    candidates_ds = tf.data.Dataset.from_tensor_slices(
        unique_article_ids
    )

    def dataframe_to_dataset(df):
        features = {
            "customer_id": df["customer_id"].astype(str).values,
            "article_id": df["article_id"].astype(str).values,
        }

        return tf.data.Dataset.from_tensor_slices(features)

    train_ds = dataframe_to_dataset(train_df)
    validation_ds = dataframe_to_dataset(validation_df)

    train_ds = (
        train_ds
        .shuffle(len(train_df), reshuffle_each_iteration=True)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    validation_ds = (
        validation_ds
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    print(f"Customer vocabulary: {len(unique_customer_ids):,}")
    print(f"Article vocabulary: {len(unique_article_ids):,}")

    return (
        train_ds,
        validation_ds,
        candidates_ds,
        unique_customer_ids,
        unique_article_ids,
    )

def train_model(
    train_ds,
    validation_ds,
    candidates_ds,
    unique_customer_ids,
    unique_article_ids,
):
    """Build, compile and train the Two-Tower retrieval model."""

    print("\nBuilding Two-Tower model...")

    model = HMTTwoTowerModel(
        unique_customer_ids=unique_customer_ids,
        unique_article_ids=unique_article_ids,
        parsed_candidates_ds=candidates_ds,
        embedding_dim=EMBEDDING_DIM,
    )

    # Configure optimizer
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

def save_training_results(model, history):
    """Save trained model weights and training history."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save model weights
    weights_path = MODEL_DIR / "two_tower.weights.h5"
    model.save_weights(weights_path)

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_path = MODEL_DIR / "training_history.csv"
    history_df.to_csv(history_path, index=False)

    print("\nTraining results saved successfully!")
    print(f"Model weights: {weights_path}")
    print(f"Training history: {history_path}")


def main():
    print("===== H&M TWO-TOWER MODEL TRAINING =====")

    # Step 1: Load interaction data
    train_df, validation_df = load_interactions()

    # Step 2: Prepare TensorFlow datasets
    (
        train_ds,
        validation_ds,
        candidates_ds,
        unique_customer_ids,
        unique_article_ids,
    ) = prepare_datasets(train_df, validation_df)

    # Step 3: Train model
    model, history = train_model(
        train_ds,
        validation_ds,
        candidates_ds,
        unique_customer_ids,
        unique_article_ids,
    )

    # Step 4: Save results
    save_training_results(model, history)

    print("\n===== TRAINING PIPELINE COMPLETED =====")


if __name__ == "__main__":
    main()
