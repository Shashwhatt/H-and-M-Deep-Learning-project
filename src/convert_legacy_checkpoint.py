import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import h5py
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.two_tower_model import HMTwoTowerModel

BASE_DIR = Path("/opt/airflow/project")
VOCAB_DIR = BASE_DIR / "models" / "serving_vocabularies"

SOURCE = BASE_DIR / "models" / "two_tower.weights.h5"
OUTPUT = BASE_DIR / "models" / "staging" / "two_tower.weights.h5"

def load_vocabulary(filename):
    with open(VOCAB_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)

def read_layer_weights(file, group_path):
    group = file[group_path]["vars"]
    return [group[str(i)][:] for i in range(len(group))]

def main():
    print("Loading vocabularies...")

    customer_ids = load_vocabulary("customer_id_vocabulary.json")
    article_ids = load_vocabulary("article_id_vocabulary.json")

    print("Customers:", len(customer_ids))
    print("Articles:", len(article_ids))

    if OUTPUT.exists():
        raise FileExistsError(
            f"Staging checkpoint already exists: {OUTPUT}. "
            "Do not overwrite it without checking it first."
        )

    print("\nBuilding serving model...")

    model = HMTwoTowerModel(
        customer_ids,
        article_ids,
        embedding_dim=32
    )

    sample = {
        "customer_id": tf.constant([customer_ids[0]]),
        "article_id": tf.constant([article_ids[0]])
    }

    model(sample)

    print("Extracting original trained weights...")

    with h5py.File(SOURCE, "r") as f:

        query_embedding = read_layer_weights(
            f,
            "query_tower\\layers\\sequential\\layers\\embedding"
        )

        query_dense_1 = read_layer_weights(
            f,
            "query_tower\\layers\\sequential_1\\layers\\dense"
        )

        query_dense_2 = read_layer_weights(
            f,
            "query_tower\\layers\\sequential_1\\layers\\dense_1"
        )

        candidate_embedding = read_layer_weights(
            f,
            "layers\\hm_candidate_tower\\layers\\sequential\\layers\\embedding"
        )

        candidate_dense_1 = read_layer_weights(
            f,
            "layers\\hm_candidate_tower\\layers\\sequential_1\\layers\\dense"
        )

        candidate_dense_2 = read_layer_weights(
            f,
            "layers\\hm_candidate_tower\\layers\\sequential_1\\layers\\dense_1"
        )

    print("Assigning customer tower weights...")

    model.query_tower.embedding.layers[1].set_weights(query_embedding)
    model.query_tower.dense.layers[0].set_weights(query_dense_1)
    model.query_tower.dense.layers[1].set_weights(query_dense_2)

    print("Assigning article tower weights...")

    model.candidate_tower.embedding.layers[1].set_weights(candidate_embedding)
    model.candidate_tower.dense.layers[0].set_weights(candidate_dense_1)
    model.candidate_tower.dense.layers[1].set_weights(candidate_dense_2)

    print("\nSaving converted checkpoint...")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(OUTPUT))

    print("Testing converted checkpoint...")

    fresh_model = HMTwoTowerModel(
        customer_ids,
        article_ids,
        embedding_dim=32
    )

    expected = fresh_model(sample)
    fresh_model.load_weights(str(OUTPUT))
    actual = fresh_model(sample)

    if not np.allclose(expected.numpy(), actual.numpy(), atol=1e-6):
        raise RuntimeError("Converted checkpoint verification failed!")

    print("\nSUCCESS: Existing trained weights converted!")
    print("Output:", OUTPUT)
    print("File size:", OUTPUT.stat().st_size)
    print("Verification: PASSED")

if __name__ == "__main__":
    main()
