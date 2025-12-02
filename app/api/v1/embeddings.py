"""
Embedding generation endpoints.

Provides REST API for generating embeddings using NB-SBERT and NB-BERT models.
"""

from typing import Union

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from app.config import Settings, get_settings
from app.models.embedding import (
    BatchEmbeddingRequest,
    BatchEmbeddingResponse,
    EmbeddingMode,
    EmbeddingOutput,
    EmbeddingRequest,
    EmbeddingResponse,
    EncoderResponse,
    ErrorResponse,
)
from app.services.embedding_service import get_embedding_service

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post(
    "",
    response_model=Union[EmbeddingResponse, EncoderResponse],
    responses={
        200: {
            "description": "Embedding generated successfully",
            "content": {
                "application/json": {
                    "examples": {
                        "sbert_embedding": {
                            "summary": "NB-SBERT Embedding",
                            "value": {
                                "vector": [0.1, 0.2, "...768 floats..."],
                                "dimension": 768,
                                "model": "NbAiLab/nb-sbert-base",
                                "normalized": True,
                            },
                        },
                        "bert_encoder": {
                            "summary": "NB-BERT Encoder Output",
                            "value": {
                                "pooled_embedding": [0.1, 0.2, "...768 floats..."],
                                "token_embeddings": [[0.1], [0.2]],
                                "dimension": 768,
                                "model": "NbAiLab/nb-bert-base",
                                "normalized": True,
                            },
                        },
                    }
                }
            },
        },
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Generate embedding",
    description="""
Generate embedding for a single text using NB-SBERT or NB-BERT models.

**Modes:**
- `embedding`: Uses NB-SBERT for sentence embeddings (recommended for RAG/search)
- `encode`: Uses NB-BERT for token-level encoder representations

**Output formats:**
- `vector`: Single 768-dimensional pooled vector
- `token_vectors`: Per-token embeddings (only with mode=encode)
- `pooled_vector`: Pooled representation from BERT

**Preprocessing:**
- Removes noise (courtesy phrases, URLs, emojis)
- Preserves semantic content (entities, actions, relationships)
- Handles Norwegian text (bokmål, nynorsk, dialects)
    """,
)
async def create_embedding(
    request: EmbeddingRequest,
    settings: Settings = Depends(get_settings),
) -> Union[EmbeddingResponse, EncoderResponse]:
    """
    Generate embedding for a single text.

    Supports both NB-SBERT (sentence embeddings) and NB-BERT (encoder representations).
    """
    try:
        service = get_embedding_service(settings)

        if request.mode == EmbeddingMode.EMBEDDING:
            # NB-SBERT sentence embedding
            vector, dimension = service.embed(
                text=request.text,
                normalize=request.normalize,
            )

            return EmbeddingResponse(
                vector=vector,
                dimension=dimension,
                model=settings.sbert_model_name,
                normalized=request.normalize,
            )

        else:
            # NB-BERT encoder
            result = service.encode(
                text=request.text,
                output_type=request.output.value,
                normalize=request.normalize,
                return_tokens=(request.output == EmbeddingOutput.TOKEN_VECTORS),
            )

            return EncoderResponse(
                pooled_embedding=result["pooled_embedding"],
                token_embeddings=result.get("token_embeddings"),
                tokens=result.get("tokens"),
                dimension=result["dimension"],
                model=settings.bert_model_name,
                normalized=request.normalize,
            )

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Embedding generation failed: {str(e)}",
        )


@router.post(
    "/batch",
    response_model=BatchEmbeddingResponse,
    responses={
        200: {"description": "Batch embeddings generated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Generate batch embeddings",
    description="""
Generate embeddings for multiple texts in a single request.

Optimized for batch processing with configurable batch size.
Only supports NB-SBERT sentence embeddings (mode=embedding).
    """,
)
async def create_batch_embedding(
    request: BatchEmbeddingRequest,
    settings: Settings = Depends(get_settings),
) -> BatchEmbeddingResponse:
    """
    Generate embeddings for multiple texts.

    Uses NB-SBERT for efficient batch processing.
    """
    try:
        # Validate batch size
        if len(request.texts) > settings.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {len(request.texts)} exceeds maximum {settings.max_batch_size}",
            )

        if request.mode != EmbeddingMode.EMBEDDING:
            raise HTTPException(
                status_code=400,
                detail="Batch processing only supports mode='embedding' (NB-SBERT)",
            )

        service = get_embedding_service(settings)

        vectors, dimension = service.embed_batch(
            texts=request.texts,
            normalize=request.normalize,
        )

        return BatchEmbeddingResponse(
            vectors=vectors,
            dimension=dimension,
            count=len(vectors),
            model=settings.sbert_model_name,
            normalized=request.normalize,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch embedding generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch embedding generation failed: {str(e)}",
        )


@router.post(
    "/similarity",
    summary="Calculate similarity",
    description="Calculate cosine similarity between two texts.",
)
async def calculate_similarity(
    text1: str,
    text2: str,
    settings: Settings = Depends(get_settings),
) -> dict:
    """
    Calculate cosine similarity between two texts.

    Uses NB-SBERT embeddings with L2 normalization.
    """
    try:
        import numpy as np

        service = get_embedding_service(settings)

        vec1, _ = service.embed(text1, normalize=True)
        vec2, _ = service.embed(text2, normalize=True)

        # Cosine similarity (dot product of normalized vectors)
        similarity = float(np.dot(vec1, vec2))

        return {
            "text1": text1[:100] + "..." if len(text1) > 100 else text1,
            "text2": text2[:100] + "..." if len(text2) > 100 else text2,
            "similarity": similarity,
            "model": settings.sbert_model_name,
        }

    except Exception as e:
        logger.error(f"Similarity calculation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Similarity calculation failed: {str(e)}",
        )
