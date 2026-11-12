
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf


class HMQueryTower(tf.keras.Model):
    """Converts customer IDs into customer embeddings."""

    def __init__(self, customer_ids, embedding_dim=32):
        super().__init__()

        self.embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=customer_ids,
                mask_token=None
            ),
            tf.keras.layers.Embedding(
                input_dim=len(customer_ids) + 1,
                output_dim=embedding_dim
            )
        ])

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(embedding_dim)
        ])

    def call(self, inputs):
        customer_ids = inputs["customer_id"]
        embeddings = self.embedding(customer_ids)
        dense_out = self.dense(embeddings)
        return tf.linalg.l2_normalize(dense_out, axis=-1)


class HMCandidateTower(tf.keras.Model):
    """Converts article IDs into product embeddings."""

    def __init__(self, article_ids, embedding_dim=32):
        super().__init__()

        self.embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=article_ids,
                mask_token=None
            ),
            tf.keras.layers.Embedding(
                input_dim=len(article_ids) + 1,
                output_dim=embedding_dim
            )
        ])

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(embedding_dim)
        ])

    def call(self, inputs):
        article_ids = inputs["article_id"]
        embeddings = self.embedding(article_ids)
        dense_out = self.dense(embeddings)
        return tf.linalg.l2_normalize(dense_out, axis=-1)


class HMTwoTowerModel(tf.keras.Model):
    """Combines customer and product towers."""

    def __init__(self, customer_ids, article_ids, embedding_dim=32):
        super().__init__()

        self.query_tower = HMQueryTower(
            customer_ids, embedding_dim
        )

        self.candidate_tower = HMCandidateTower(
            article_ids, embedding_dim
        )

    def call(self, inputs):
        user_embeddings = self.query_tower({
            "customer_id": inputs["customer_id"]
        })

        item_embeddings = self.candidate_tower({
            "article_id": inputs["article_id"]
        })

        scores = tf.reduce_sum(
            user_embeddings * item_embeddings,
            axis=1
        )

        return scores

import h5py


def load_serving_weights(model, checkpoint_path):
    """Load trained tower weights directly from the HDF5 checkpoint."""

    def load_tower(tower, group_path, h5_file):
        embedding = h5_file[
            f"{group_path}/layers/sequential/layers/embedding/vars/0"
        ][()]

        dense1_kernel = h5_file[
            f"{group_path}/layers/sequential_1/layers/dense/vars/0"
        ][()]
        dense1_bias = h5_file[
            f"{group_path}/layers/sequential_1/layers/dense/vars/1"
        ][()]

        dense2_kernel = h5_file[
            f"{group_path}/layers/sequential_1/layers/dense_1/vars/0"
        ][()]
        dense2_bias = h5_file[
            f"{group_path}/layers/sequential_1/layers/dense_1/vars/1"
        ][()]

        tower.embedding.layers[1].set_weights([embedding])

        tower.dense.layers[0].set_weights([
            dense1_kernel,
            dense1_bias
        ])

        tower.dense.layers[1].set_weights([
            dense2_kernel,
            dense2_bias
        ])

    with h5py.File(checkpoint_path, "r") as f:
        load_tower(model.query_tower, "query_tower", f)
        load_tower(model.candidate_tower, "layers/hm_candidate_tower", f)

    print("SUCCESS: Customer and article tower weights loaded!")
