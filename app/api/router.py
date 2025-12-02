"""API router aggregation."""

from fastapi import APIRouter

from app.api.v1.embeddings import router as embeddings_router
from app.api.v1.health import router as health_router

api_router = APIRouter(prefix="/api/v1")

# Include sub-routers
api_router.include_router(health_router)
api_router.include_router(embeddings_router)
