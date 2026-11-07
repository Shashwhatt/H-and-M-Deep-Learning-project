
from fastapi import APIRouter, HTTPException

from api.schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
)
from api.services.recommendation_service import RecommendationService


router = APIRouter(
    prefix="/recommendations",
    tags=["Recommendations"],
)


def get_recommendation_service() -> RecommendationService:
    from api.services.redis_service import RedisService

    return RecommendationService(RedisService())


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
    except NotImplementedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
