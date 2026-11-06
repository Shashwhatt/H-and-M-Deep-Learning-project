
from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Customer ID")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of recommendations")


class RecommendedProduct(BaseModel):
    article_id: str
    score: float


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: list[RecommendedProduct]
