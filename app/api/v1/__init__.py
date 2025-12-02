"""API v1 endpoints."""

from app.api.v1.embeddings import router as embeddings_router
from app.api.v1.health import router as health_router

__all__ = ["embeddings_router", "health_router"]
