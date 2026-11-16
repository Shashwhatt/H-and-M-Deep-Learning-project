
import json

import tensorflow as tf

from api.schemas.recommendation import (
    RecommendedProduct,
    RecommendationResponse,
)
from api.services.redis_service import RedisService
from src.load_recommendation_model import load_recommendation_model


class RecommendationService:
    def __init__(self, redis_service: RedisService):
        self.redis_service = redis_service

        (
            self.model,
            self.customer_ids,
            self.article_ids,
            self.candidate_embeddings,
        ) = load_recommendation_model()

        self.customer_id_set = set(self.customer_ids)

    def get_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
    ) -> RecommendationResponse:

        if user_id not in self.customer_id_set:
            raise ValueError(f"Unknown customer ID: {user_id}")

        # Retrieve customer features from Redis.
        redis_key = f"user_features:{user_id}"
        raw_features = self.redis_service.get_user_features(redis_key)

        if raw_features is None:
            raise RuntimeError(
                f"Customer features not found in Redis for user: {user_id}"
            )

        try:
            user_features = json.loads(raw_features)
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(
                "Invalid customer feature data in Redis."
            ) from exc

        if not isinstance(user_features, dict):
            raise RuntimeError("Customer features must be a JSON object.")

        if user_features.get("customer_id") != user_id:
            raise RuntimeError("Customer feature ID mismatch in Redis.")

        # Validate all required context-aware features.
        required_features = [
            "age",
            "age_missing",
            "club_member_status_id",
            "fashion_news_frequency_id",
        ]

        missing_features = [
            feature
            for feature in required_features
            if feature not in user_features
        ]

        if missing_features:
            raise RuntimeError(
                f"Customer features missing from Redis: {missing_features}"
            )

        try:
            # Prepare all five inputs required by the query tower.
            query_features = {
                "customer_id": tf.constant([user_id]),
                "age": tf.constant(
                    [float(user_features["age"])],
                    dtype=tf.float32,
                ),
                "age_missing": tf.constant(
                    [int(user_features["age_missing"])],
                    dtype=tf.int64,
                ),
                "club_member_status_id": tf.constant(
                    [int(user_features["club_member_status_id"])],
                    dtype=tf.int64,
                ),
                "fashion_news_frequency_id": tf.constant(
                    [int(user_features["fashion_news_frequency_id"])],
                    dtype=tf.int64,
                ),
            }
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                "Customer features in Redis contain invalid values."
            ) from exc

        # Generate context-aware customer embedding.
        user_embedding = self.model.query_model(query_features)

        # Calculate similarity scores against all article embeddings.
        scores = tf.linalg.matmul(
            user_embedding,
            self.candidate_embeddings,
            transpose_b=True,
        )[0]

        k = min(top_k, len(self.article_ids))
        top_results = tf.math.top_k(scores, k=k)

        recommendations = []

        for index, score in zip(
            top_results.indices.numpy().tolist(),
            top_results.values.numpy().tolist(),
        ):
            recommendations.append(
                RecommendedProduct(
                    article_id=self.article_ids[index],
                    score=float(score),
                )
            )

        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
        )
