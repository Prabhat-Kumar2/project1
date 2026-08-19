"""
REST API (SDD 9.7 & 10.1) built with FastAPI.

On startup a single global RecommendationEngine is created against
processed_news.csv and its embeddings are loaded/generated once.

Endpoints:
    GET  /            -> service metadata
    GET  /health      -> health check + article count
    GET  /users       -> list simulated user_ids
    POST /recommend/  -> top-N recommendations for a reading history
"""
import ast
import os
import sys

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.recommendation import RecommendationEngine

app = FastAPI(
    title="Personalised Content Recommendation API",
    description="Content-based news recommendations using sentence embeddings.",
    version="1.0.0",
)

# Global engine instance (created on startup).
engine: RecommendationEngine | None = None


class RecommendationRequest(BaseModel):
    history_indices: list[int] = Field(
        ..., description="Row indices into processed_news.csv the user has read."
    )


class PreferenceRequest(BaseModel):
    """Interest profile for a brand-new user with no reading history."""

    categories: list[str] = Field(
        default_factory=list, description="Areas of interest (news categories)."
    )
    preferred_sentiment: str = Field(
        "Any", description="Positive | Neutral | Negative | Any."
    )
    interest_focus: float = Field(
        0.7, ge=0.0, le=1.0, description="Weight on topical match to chosen areas."
    )
    sentiment_weight: float = Field(
        0.3, ge=0.0, le=1.0, description="Weight on matching the preferred sentiment."
    )


class RecommendationItem(BaseModel):
    title: str
    sentiment: str
    category: str | None = None
    url: str | None = None
    description: str | None = None
    score: float | None = None
    topic_match: float | None = None
    sentiment_match: bool | None = None


class RecommendationResponse(BaseModel):
    recommendations: list[RecommendationItem]


@app.on_event("startup")
def _load_engine() -> None:
    global engine
    engine = RecommendationEngine()
    engine.generate_embeddings()
    print(f"[api] Engine ready: {len(engine.df)} articles, "
          f"backend={engine.backend}")


@app.get("/")
def root() -> dict:
    return {
        "service": "Personalised Content Recommendation API",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/users",
            "/categories",
            "/stats",
            "/recommend/ (POST)",
            "/recommend/preferences (POST)",
            "/docs",
        ],
    }


@app.get("/health")
def health() -> dict:
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised.")
    return {
        "status": "ok",
        "articles": int(len(engine.df)),
        "embedding_backend": engine.backend,
    }


@app.get("/users")
def users() -> dict:
    """Return the list of simulated user ids (convenience for the dashboard)."""
    if not os.path.exists(config.USER_SIMULATION_PATH):
        raise HTTPException(status_code=404, detail="user_simulation.csv not found.")
    import pandas as pd

    df = pd.read_csv(config.USER_SIMULATION_PATH)
    return {"users": df["user_id"].tolist()}


@app.get("/categories")
def categories() -> dict:
    """Available news categories with article counts (for the new-user form)."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised.")
    return {"categories": engine.category_counts()}


@app.get("/stats")
def stats() -> dict:
    """Corpus analytics summary."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised.")
    return engine.analytics_summary()


@app.post("/recommend/", response_model=RecommendationResponse)
def recommend(
    request: RecommendationRequest,
    top_n: int = Query(config.TOP_N, ge=1, le=50, description="Number of items to return."),
) -> RecommendationResponse:
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised.")
    recs = engine.get_top_n_recommendations(request.history_indices, top_n=top_n)
    return RecommendationResponse(recommendations=recs)


@app.post("/recommend/preferences", response_model=RecommendationResponse)
def recommend_preferences(
    request: PreferenceRequest,
    top_n: int = Query(config.TOP_N, ge=1, le=50, description="Number of items to return."),
) -> RecommendationResponse:
    """Cold-start recommendations from a self-declared interest profile."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised.")
    recs = engine.recommend_for_preferences(
        categories=request.categories,
        preferred_sentiment=request.preferred_sentiment,
        interest_focus=request.interest_focus,
        sentiment_weight=request.sentiment_weight,
        top_n=top_n,
    )
    return RecommendationResponse(recommendations=recs)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host=config.API_HOST, port=config.API_PORT, reload=False)
