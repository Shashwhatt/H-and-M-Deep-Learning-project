from functools import lru_cache

from fastapi import APIRouter, HTTPException

from api.schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
)
from api.services.recommendation_service import RecommendationService
from api.services.redis_service import RedisService


router = APIRouter(
    prefix="/recommendations",
    tags=["Recommendations"],
)


@lru_cache(maxsize=1)
def get_recommendation_service() -> RecommendationService:
    redis_service = RedisService()
    return RecommendationService(redis_service)


@router.post("/", response_model=RecommendationResponse)
def recommend_products(
    request: RecommendationRequest,
):
    try:
        service = get_recommendation_service()

        return service.get_recommendations(
            user_id=request.user_id,
            top_k=request.top_k,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except (RuntimeError, OSError) as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
