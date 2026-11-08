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

        # Retrieve customer features from Redis
        redis_key = f"user_features:{user_id}"
        raw_features = self.redis_service.get_user_features(redis_key)

        if raw_features is None:
            raise RuntimeError(
                f"Customer features not found in Redis for user: {user_id}"
            )

        try:
            user_features = json.loads(raw_features)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Invalid customer feature data in Redis."
            ) from exc

        if user_features.get("customer_id") != user_id:
            raise RuntimeError("Customer feature ID mismatch in Redis.")

        # Generate recommendations using the current ID-based query tower
        user_embedding = self.model.query_model({
            "customer_id": tf.constant([user_id])
        })

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
