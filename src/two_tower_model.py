
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
        return self.dense(embeddings)


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
        return self.dense(embeddings)


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
