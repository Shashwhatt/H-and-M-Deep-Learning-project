import os
import tensorflow as tf
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Ensuring compatibility with legacy Keras API
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np

class HMQueryTower(tf.keras.Model):
    """
    Query Tower: Processes user identifiers, interaction history, 
    and contextual features to generate dense user embedding vectors.
    """
    def __init__(self, unique_customer_ids, embedding_dim=32):
        super().__init__()
        
        # User ID embedding layer with StringLookup preprocessing
        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_customer_ids, mask_token=None
            ),
            tf.keras.layers.Embedding(
                input_dim=len(unique_customer_ids) + 1, 
                output_dim=embedding_dim
            )
        ])
        
        # Dense projection layers for user context
        self.dense_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(embedding_dim)
        ])

    def call(self, inputs):
        # Extract and embed user ID
        customer_id = inputs.get("customer_id")
        user_embed = self.user_embedding(customer_id)
        
        # Pass through dense layers
        return self.dense_layer(user_embed)


class HMCandidateTower(tf.keras.Model):
    """
    Candidate Tower: Processes article metadata (IDs, product groups, colors)
    and item popularity scores to generate dense item embedding vectors.
    """
    def __init__(self, unique_article_ids, embedding_dim=32):
        super().__init__()
        
        # Article ID embedding layer with StringLookup preprocessing
        self.article_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_article_ids, mask_token=None
            ),
            tf.keras.layers.Embedding(
                input_dim=len(unique_article_ids) + 1, 
                output_dim=embedding_dim
            )
        ])
        
        # Dense projection layers for item features
        self.dense_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(embedding_dim)
        ])

    def call(self, inputs):
        # Handle inputs whether passed as a dictionary or direct tensor
        if isinstance(inputs, dict):
            article_id = inputs.get("article_id")
        else:
            article_id = inputs
            
        item_embed = self.article_embedding(article_id)
        return self.dense_layer(item_embed)


class HMTTwoTowerModel(tfrs.Model):
    """
    Full Two-Tower Retrieval Model combining Query and Candidate towers
    using TFRS Retrieval task and FactorizedTopK evaluation metrics.
    """
    def __init__(self, unique_customer_ids, unique_article_ids, parsed_candidates_ds, embedding_dim=32):
        super().__init__()
        
        # Initialize individual towers
        self.query_model = HMQueryTower(unique_customer_ids, embedding_dim)
        self.candidate_model = HMCandidateTower(unique_article_ids, embedding_dim)
        
        # Define TFRS Retrieval task with evaluation metrics
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=parsed_candidates_ds.batch(128).map(self.candidate_model)
            )
        )

    def compute_loss(self, features, training=False):
        # Compute user and item embeddings
        user_embeddings = self.query_model(features)
        positive_candidate_embeddings = self.candidate_model(features["article_id"])
        
        # Compute task loss and metrics using in-batch negative sampling
        return self.task(user_embeddings, positive_candidate_embeddings)


if __name__ == "__main__":
    print("--- H&M Two-Tower Architecture Script Loaded Successfully ---")
    print("Ready to be integrated with training loops on engineered parquet datasets.")