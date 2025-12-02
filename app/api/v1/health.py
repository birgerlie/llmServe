"""Health check endpoints."""

from fastapi import APIRouter, Depends

from app import __version__
from app.config import Settings, get_settings
from app.models.embedding import HealthResponse
from app.services.embedding_service import get_embedding_service

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the service and loaded models.",
)
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """
    Health check endpoint.

    Returns the status of the service including:
    - Overall service status
    - Whether NB-SBERT model is loaded
    - Whether NB-BERT model is loaded
    - Current device being used
    - Service version
    """
    service = get_embedding_service(settings)

    return HealthResponse(
        status="healthy",
        sbert_model_loaded=service.sbert_loaded,
        bert_model_loaded=service.bert_loaded,
        device=service.device,
        version=__version__,
    )


@router.get(
    "/ready",
    summary="Readiness check",
    description="Check if the service is ready to handle requests.",
)
async def readiness_check(
    settings: Settings = Depends(get_settings),
) -> dict:
    """
    Readiness check endpoint.

    Returns whether the service is ready to process embedding requests.
    Service is considered ready when at least one model is loaded.
    """
    service = get_embedding_service(settings)
    is_ready = service.sbert_loaded or service.bert_loaded

    return {
        "ready": is_ready,
        "message": "Service ready" if is_ready else "No models loaded yet",
    }
