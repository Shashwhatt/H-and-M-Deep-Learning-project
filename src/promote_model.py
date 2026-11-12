import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.two_tower_model import HMTwoTowerModel


BASE_DIR = Path(__file__).resolve().parent.parent

VOCAB_DIR = BASE_DIR / "models" / "serving_vocabularies"
STAGED_MODEL = BASE_DIR / "models" / "staging" / "two_tower.weights.h5"
FINAL_MODEL = BASE_DIR / "models" / "two_tower.weights.h5"
BACKUP_DIR = BASE_DIR / "models" / "backups"

EMBEDDING_DIM = 32


def build_model(customer_ids, article_ids):
    model = HMTwoTowerModel(
        customer_ids,
        article_ids,
        embedding_dim=EMBEDDING_DIM
    )

    model({
        "customer_id": tf.constant([customer_ids[0]]),
        "article_id": tf.constant([article_ids[0]])
    })

    return model


def validate_model(model, customer_ids, article_ids):
    for weight in model.weights:
        if not np.isfinite(weight.numpy()).all():
            raise ValueError(
                "Model contains invalid or non-finite weights."
            )

    user_embedding = model.query_tower({
        "customer_id": tf.constant([customer_ids[0]])
    }).numpy()

    item_embedding = model.candidate_tower({
        "article_id": tf.constant([article_ids[0]])
    }).numpy()

    if user_embedding.shape != (1, EMBEDDING_DIM):
        raise ValueError("Invalid customer embedding shape.")

    if item_embedding.shape != (1, EMBEDDING_DIM):
        raise ValueError("Invalid article embedding shape.")

    if not np.isfinite(user_embedding).all():
        raise ValueError("Customer embedding contains invalid values.")

    if not np.isfinite(item_embedding).all():
        raise ValueError("Article embedding contains invalid values.")

    print("All model validation checks passed.")


def main():
    if not STAGED_MODEL.exists():
        raise FileNotFoundError(
            f"Staged model not found: {STAGED_MODEL}"
        )

    if STAGED_MODEL.stat().st_size == 0:
        raise ValueError("Staged model file is empty.")

    print("Loading vocabularies...")

    with open(
        VOCAB_DIR / "customer_id_vocabulary.json",
        "r",
        encoding="utf-8"
    ) as file:
        customer_ids = json.load(file)

    with open(
        VOCAB_DIR / "article_id_vocabulary.json",
        "r",
        encoding="utf-8"
    ) as file:
        article_ids = json.load(file)

    if not customer_ids or not article_ids:
        raise ValueError("Model vocabularies cannot be empty.")

    print("Validating staged model...")

    staged_model = build_model(customer_ids, article_ids)
    staged_model.load_weights(str(STAGED_MODEL))

    validate_model(staged_model, customer_ids, article_ids)

    # Copy staged model to a temporary file.
    temporary_file = FINAL_MODEL.with_name(
        "two_tower_temporary.weights.h5"
    )

    try:
        shutil.copy2(STAGED_MODEL, temporary_file)

        # Verify the copied file independently before promotion.
        verification_model = build_model(customer_ids, article_ids)
        verification_model.load_weights(str(temporary_file))

        validate_model(
            verification_model,
            customer_ids,
            article_ids
        )

        # Back up the current production model.
        if FINAL_MODEL.exists():
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = BACKUP_DIR / (
                f"two_tower_{timestamp}.weights.h5"
            )

            shutil.copy2(FINAL_MODEL, backup_file)
            print("Production backup created:", backup_file)

        # Replace production model only after successful validation.
        os.replace(temporary_file, FINAL_MODEL)

        print("Model promotion successful!")
        print("Production model:", FINAL_MODEL)

    finally:
        if temporary_file.exists():
            temporary_file.unlink()


if __name__ == "__main__":
    main()
