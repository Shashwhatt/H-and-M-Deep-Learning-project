import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_model import HMRetrievalModel, EMBEDDING_DIM
from two_tower_model import HMTwoTowerModel


def main():
    customer_ids = [
        "customer_001",
        "customer_002",
        "customer_003",
        "customer_004",
    ]

    article_ids = [
        "article_001",
        "article_002",
        "article_003",
        "article_004",
    ]

    print("1. Building training model...")

    training_model = HMRetrievalModel(
        customer_ids,
        article_ids
    )

    training_model.query_tower({
        "customer_id": tf.constant([customer_ids[0]])
    })

    training_model.candidate_tower({
        "article_id": tf.constant([article_ids[0]])
    })

    print("2. Building serving model...")

    serving_model = HMTwoTowerModel(
        customer_ids,
        article_ids,
        embedding_dim=EMBEDDING_DIM
    )

    sample = {
        "customer_id": tf.constant([customer_ids[0]]),
        "article_id": tf.constant([article_ids[0]])
    }

    serving_model(sample)

    print("3. Transferring tower weights...")

    serving_model.query_tower.set_weights(
        training_model.query_tower.get_weights()
    )

    serving_model.candidate_tower.set_weights(
        training_model.candidate_tower.get_weights()
    )

    expected_user = serving_model.query_tower({
        "customer_id": tf.constant([customer_ids[0]])
    }).numpy()

    expected_item = serving_model.candidate_tower({
        "article_id": tf.constant([article_ids[0]])
    }).numpy()

    test_file = Path("models/staging/compatibility_test.weights.h5")
    test_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        print("4. Saving serving weights...")
        serving_model.save_weights(str(test_file))

        print("5. Loading into a fresh serving model...")

        verification_model = HMTwoTowerModel(
            customer_ids,
            article_ids,
            embedding_dim=EMBEDDING_DIM
        )

        verification_model(sample)
        verification_model.load_weights(str(test_file))

        actual_user = verification_model.query_tower({
            "customer_id": tf.constant([customer_ids[0]])
        }).numpy()

        actual_item = verification_model.candidate_tower({
            "article_id": tf.constant([article_ids[0]])
        }).numpy()

        assert np.allclose(expected_user, actual_user, atol=1e-6)
        assert np.allclose(expected_item, actual_item, atol=1e-6)

        print("\nSUCCESS: Training-to-serving weight compatibility verified!")
        print("Customer embedding shape:", actual_user.shape)
        print("Article embedding shape:", actual_item.shape)

    finally:
        if test_file.exists():
            test_file.unlink()


if __name__ == "__main__":
    main()
