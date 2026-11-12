import os
import tensorflow as tf
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Ensuring compatibility with legacy Keras API
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np

class HMQueryTower(tf.keras.Model):
    """
    Query Tower: Processes user identifiers, interaction history, 
    demographic features (age, club membership, fashion news frequency),
    and contextual features to generate dense user embedding vectors.
    """
    def __init__(self, unique_customer_ids, unique_club_statuses=None, unique_fashion_news=None, embedding_dim=32):
        super().__init__()
        
        # 1. User ID embedding layer with StringLookup preprocessing
        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_customer_ids, mask_token=None
            ),
            tf.keras.layers.Embedding(
                input_dim=len(unique_customer_ids) + 1, 
                output_dim=embedding_dim
            )
        ])
        
        # 2. Categorical Metadata Embedding: Club Member Status (if provided)
        self.use_club_status = unique_club_statuses is not None
        if self.use_club_status:
            self.club_status_embedding = tf.keras.Sequential([
                tf.keras.layers.StringLookup(
                    vocabulary=unique_club_statuses, mask_token=None
                ),
                tf.keras.layers.Embedding(
                    input_dim=len(unique_club_statuses) + 1, 
                    output_dim=16  # Smaller embedding dimension for low-cardinality features
                )
            ])

        # 3. Categorical Metadata Embedding: Fashion News Frequency (if provided)
        self.use_fashion_news = unique_fashion_news is not None
        if self.use_fashion_news:
            self.fashion_news_embedding = tf.keras.Sequential([
                tf.keras.layers.StringLookup(
                    vocabulary=unique_fashion_news, mask_token=None
                ),
                tf.keras.layers.Embedding(
                    input_dim=len(unique_fashion_news) + 1, 
                    output_dim=16
                )
            ])

        # 4. Normalization layer for numerical features (e.g., age)
        self.age_normalization = tf.keras.layers.Normalization(axis=None)
        
        # Dense projection layers for combined user context
        # Calculate total input dimension based on active features
        total_feature_dim = embedding_dim
        if self.use_club_status:
            total_feature_dim += 16
        if self.use_fashion_news:
            total_feature_dim += 16
        total_feature_dim += 1  # For age feature

        self.dense_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(embedding_dim)
        ])

    def call(self, inputs):
        feature_list = []
        
        # Extract and embed user ID
        customer_id = inputs.get("customer_id")
        user_embed = self.user_embedding(customer_id)
        feature_list.append(user_embed)
        
        # Extract and embed Club Member Status if configured
        if self.use_club_status and "club_member_status" in inputs:
            club_embed = self.club_status_embedding(inputs.get("club_member_status"))
            feature_list.append(club_embed)

        # Extract and embed Fashion News Frequency if configured
        if self.use_fashion_news and "fashion_news_frequency" in inputs:
            news_embed = self.fashion_news_embedding(inputs.get("fashion_news_frequency"))
            feature_list.append(news_embed)

        # Process numerical feature (Age)
        if "age" in inputs:
            age_input = tf.cast(inputs.get("age"), tf.float32)
            # Ensure proper shape (batch_size, 1)
            if len(age_input.shape) == 1:
                age_input = tf.expand_dims(age_input, -1)
            normalized_age = self.age_normalization(age_input)
            feature_list.append(normalized_age)

        # Concatenate all user features along the feature dimension
        combined_features = tf.concat(feature_list, axis=-1)
        
        # Pass through dense layers
        return self.dense_layer(combined_features)


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
        
        # Dense projection layers for the item features
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
    def __init__(self, unique_customer_ids, unique_article_ids, parsed_candidates_ds, 
                 unique_club_statuses=None, unique_fashion_news=None, embedding_dim=32):
        super().__init__()
        
        # Initialize the individual towers with metadata support
        self.query_model = HMQueryTower(
            unique_customer_ids, 
            unique_club_statuses=unique_club_statuses, 
            unique_fashion_news=unique_fashion_news, 
            embedding_dim=embedding_dim
        )
        self.candidate_model = HMCandidateTower(unique_article_ids, embedding_dim)
        
        # Define TFRS Retrieval task with evaluation metrics
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=parsed_candidates_ds.batch(128).map(self.candidate_model)
            )
        )

    def compute_loss(self, features, training=False):
        # Compute user and item embeddings (features dictionary contains user metadata and article_id)
        user_embeddings = self.query_model(features)
        positive_candidate_embeddings = self.candidate_model(features["article_id"])
        
        # Compute task loss and metrics using in-batch negative sampling
        return self.task(user_embeddings, positive_candidate_embeddings)


if __name__ == "__main__":
    print("--- H&M Two-Tower Neural Network Script Loaded Successfully ---")
    print("Ready to be integrated with the training loops on engineered parquet datasets.")