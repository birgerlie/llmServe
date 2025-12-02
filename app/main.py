"""
llmServe - Norwegian Embedding Generator Service

A FastAPI-based service for generating embeddings using NB-BERT and NB-SBERT models.
Optimized for Norwegian text processing in RAG systems.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app import __version__
from app.api.router import api_router
from app.config import get_settings
from app.services.embedding_service import get_embedding_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{__version__}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Device: {settings.device}")

    # Preload models if not in debug mode
    if not settings.debug:
        logger.info("Preloading NB-SBERT model...")
        service = get_embedding_service(settings)
        service.load_sbert_model()
        logger.info("NB-SBERT model loaded")

    yield

    # Shutdown
    logger.info("Shutting down llmServe...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="""
## Norwegian Embedding Generator Service

A high-performance embedding service using **NB-BERT** and **NB-SBERT** models
from the National Library of Norway (NB AiLab).

### Features

- **NB-SBERT Embeddings**: Sentence embeddings optimized for semantic similarity
- **NB-BERT Encoder**: Token-level encoder representations
- **Norwegian Optimized**: Supports bokmål, nynorsk, dialects, and historical texts
- **RAG Ready**: Designed for Retrieval-Augmented Generation pipelines
- **CPU Optimized**: Runs efficiently on Apple Silicon (M1/M2) and standard CPUs

### Use Cases

- Semantic search and retrieval
- Document clustering and classification
- Knowledge graph construction
- Question-answering systems
- Text deduplication

### Models

- `NbAiLab/nb-sbert-base`: Sentence-BERT for 768-dim sentence embeddings
- `NbAiLab/nb-bert-base`: BERT encoder for token-level representations
        """,
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API router
    app.include_router(api_router)

    # Root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint with service information."""
        return {
            "service": settings.app_name,
            "version": __version__,
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
    )
