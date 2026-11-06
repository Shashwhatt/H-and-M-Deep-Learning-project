
from fastapi import FastAPI

from api.routes.recommendations import router as recommendations_router


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
    return {
        "status": "healthy",
        "service": "recommendation-api",
    }
