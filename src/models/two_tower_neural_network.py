
import os

# Set this before importing TensorFlow.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tensorflow_recommenders as tfrs


class HMQueryTower(tf.keras.Model):
    """Query tower combining customer ID and contextual features."""

    def __init__(
        self,
        unique_customer_ids,
        club_status_vocab_size,
        fashion_frequency_vocab_size,
        embedding_dim=32,
    ):
        super().__init__()

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_customer_ids,
                mask_token=None,
            ),
            tf.keras.layers.Embedding(
                input_dim=len(unique_customer_ids) + 1,
                output_dim=embedding_dim,
            ),
        ])

        self.club_status_embedding = tf.keras.layers.Embedding(
            input_dim=club_status_vocab_size,
            output_dim=8,
        )

        self.fashion_frequency_embedding = tf.keras.layers.Embedding(
            input_dim=fashion_frequency_vocab_size,
            output_dim=8,
        )

        self.dense_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(embedding_dim),
        ])

    def call(self, inputs):
        customer_embedding = self.user_embedding(
            inputs["customer_id"]
        )

        club_embedding = self.club_status_embedding(
            tf.cast(inputs["club_member_status_id"], tf.int32)
        )

        fashion_embedding = self.fashion_frequency_embedding(
            tf.cast(inputs["fashion_news_frequency_id"], tf.int32)
        )

        age = tf.cast(inputs["age"], tf.float32)
        age = tf.reshape(age, [-1, 1])

        age_missing = tf.cast(
            inputs["age_missing"], tf.float32
        )
        age_missing = tf.reshape(age_missing, [-1, 1])

        combined_features = tf.concat(
            [
                customer_embedding,
                club_embedding,
                fashion_embedding,
                age,
                age_missing,
            ],
            axis=1,
        )

        return self.dense_layer(combined_features)


class HMCandidateTower(tf.keras.Model):
    """Candidate tower generating article embeddings."""

    def __init__(self, unique_article_ids, embedding_dim=32):
        super().__init__()

        self.article_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_article_ids,
                mask_token=None,
            ),
            tf.keras.layers.Embedding(
                input_dim=len(unique_article_ids) + 1,
                output_dim=embedding_dim,
            ),
        ])

        self.dense_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(embedding_dim),
        ])

    def call(self, inputs):
        if isinstance(inputs, dict):
            article_id = inputs["article_id"]
        else:
            article_id = inputs

        item_embedding = self.article_embedding(article_id)

        return self.dense_layer(item_embedding)


class HMTTwoTowerModel(tfrs.Model):
    """Two-Tower retrieval model with customer context."""

    def __init__(
        self,
        unique_customer_ids,
        unique_article_ids,
        parsed_candidates_ds,
        club_status_vocab_size,
        fashion_frequency_vocab_size,
        embedding_dim=32,
    ):
        super().__init__()

        self.query_model = HMQueryTower(
            unique_customer_ids=unique_customer_ids,
            club_status_vocab_size=club_status_vocab_size,
            fashion_frequency_vocab_size=fashion_frequency_vocab_size,
            embedding_dim=embedding_dim,
        )

        self.candidate_model = HMCandidateTower(
            unique_article_ids=unique_article_ids,
            embedding_dim=embedding_dim,
        )

        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=parsed_candidates_ds
                .batch(128)
                .map(self.candidate_model)
            )
        )

    def compute_loss(self, features, training=False):
        user_embeddings = self.query_model(features)

        positive_candidate_embeddings = self.candidate_model(
            features["article_id"]
        )

        return self.task(
            user_embeddings,
            positive_candidate_embeddings,
        )


if __name__ == "__main__":
    print("H&M Context-Aware Two-Tower model loaded successfully.")
