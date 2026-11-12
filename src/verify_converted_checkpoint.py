import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import h5py
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.two_tower_model import HMTwoTowerModel

BASE = Path("/opt/airflow/project")
VOCAB = BASE / "models" / "serving_vocabularies"
SOURCE = BASE / "models" / "two_tower.weights.h5"
STAGING = BASE / "models" / "staging" / "two_tower.weights.h5"

def read_weights(f, path):
    group = f[path]["vars"]
    return [group[str(i)][:] for i in range(len(group))]

def main():
    print("Loading vocabularies...")

    with open(VOCAB / "customer_id_vocabulary.json") as f:
        customers = json.load(f)

    with open(VOCAB / "article_id_vocabulary.json") as f:
        articles = json.load(f)

    sample = {
        "customer_id": tf.constant(customers[:4]),
        "article_id": tf.constant(articles[:4])
    }

    print("Building reference model...")
    reference = HMTwoTowerModel(customers, articles, 32)
    reference(sample)

    print("Restoring original trained weights...")
    with h5py.File(SOURCE, "r") as f:
        reference.query_tower.embedding.layers[1].set_weights(
            read_weights(f, "query_tower\\layers\\sequential\\layers\\embedding")
        )
        reference.query_tower.dense.layers[0].set_weights(
            read_weights(f, "query_tower\\layers\\sequential_1\\layers\\dense")
        )
        reference.query_tower.dense.layers[1].set_weights(
            read_weights(f, "query_tower\\layers\\sequential_1\\layers\\dense_1")
        )
        reference.candidate_tower.embedding.layers[1].set_weights(
            read_weights(f, "layers\\hm_candidate_tower\\layers\\sequential\\layers\\embedding")
        )
        reference.candidate_tower.dense.layers[0].set_weights(
            read_weights(f, "layers\\hm_candidate_tower\\layers\\sequential_1\\layers\\dense")
        )
        reference.candidate_tower.dense.layers[1].set_weights(
            read_weights(f, "layers\\hm_candidate_tower\\layers\\sequential_1\\layers\\dense_1")
        )

    expected = reference(sample).numpy()

    print("Loading converted staging checkpoint...")
    converted = HMTwoTowerModel(customers, articles, 32)
    converted(sample)
    converted.load_weights(str(STAGING))

    actual = converted(sample).numpy()

    print("Comparing outputs...")
    difference = np.max(np.abs(expected - actual))
    print("Maximum absolute difference:", difference)

    if not np.allclose(expected, actual, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Verification failed: outputs do not match.")

    print("\nSUCCESS: Converted checkpoint matches original trained weights!")
    print("Verification: PASSED")

if __name__ == "__main__":
    main()
