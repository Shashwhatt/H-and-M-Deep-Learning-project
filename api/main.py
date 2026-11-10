

from fastapi import FastAPI

from api.routes.recommendations import (
    router as recommendations_router,
    get_recommendation_service,
)


app = FastAPI(
    title="H&M Recommendation API",
    description="Context-Aware Neural Recommendation Engine",
    version="1.0.0",
)

app.include_router(recommendations_router)


@app.get("/")
def root():
    return {
        "message": "H&M Recommendation API is running",
        "status": "active",
    }


@app.get("/health")
def health_check():
    try:
        service = get_recommendation_service()

        redis_connected = service.redis_service.ping()
        model_loaded = service.model is not None

        overall_status = (
            "healthy"
            if redis_connected and model_loaded
            else "degraded"
        )

        return {
            "status": overall_status,
            "service": "recommendation-api",
            "redis": "connected" if redis_connected else "disconnected",
            "model": "loaded" if model_loaded else "not loaded",
        }

    except Exception as exc:
        return {
            "status": "unhealthy",
            "service": "recommendation-api",
            "error": str(exc),
        }
