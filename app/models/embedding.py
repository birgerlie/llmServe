"""
Pydantic models for embedding request/response schemas.

Follows the specification for NB-BERT/NB-SBERT embedding generation.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class EmbeddingMode(str, Enum):
    """Embedding generation mode."""

    EMBEDDING = "embedding"  # NB-SBERT sentence embeddings
    ENCODE = "encode"  # NB-BERT encoder representations


class EmbeddingOutput(str, Enum):
    """Output format for embeddings."""

    VECTOR = "vector"  # Single pooled vector (768-dim)
    TOKEN_VECTORS = "token_vectors"  # Per-token embeddings
    POOLED_VECTOR = "pooled_vector"  # Pooled representation from BERT


class EmbeddingRequest(BaseModel):
    """
    Request model for embedding generation.

    Supports both single text and batch processing.
    """

    text: str = Field(
        ...,
        description="Text to generate embeddings for",
        min_length=0,
        examples=["Sykehjemmet tilbyr korttidsopphold og langtidsopphold."],
    )
    mode: EmbeddingMode = Field(
        default=EmbeddingMode.EMBEDDING,
        description="Embedding mode: 'embedding' for NB-SBERT, 'encode' for NB-BERT",
    )
    output: EmbeddingOutput = Field(
        default=EmbeddingOutput.VECTOR,
        description="Output format: 'vector', 'token_vectors', or 'pooled_vector'",
    )
    normalize: bool = Field(
        default=True,
        description="Whether to L2-normalize the output vectors",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Sykehjemmet tilbyr korttidsopphold, langtidsopphold og dagtilbud.",
                    "mode": "embedding",
                    "output": "vector",
                    "normalize": True,
                }
            ]
        }
    }


class BatchEmbeddingRequest(BaseModel):
    """
    Request model for batch embedding generation.

    Optimized for processing multiple texts in a single request.
    """

    texts: List[str] = Field(
        ...,
        description="List of texts to generate embeddings for",
        min_length=1,
        examples=[["Tekst en", "Tekst to", "Tekst tre"]],
    )
    mode: EmbeddingMode = Field(
        default=EmbeddingMode.EMBEDDING,
        description="Embedding mode: 'embedding' for NB-SBERT, 'encode' for NB-BERT",
    )
    output: EmbeddingOutput = Field(
        default=EmbeddingOutput.VECTOR,
        description="Output format for embeddings",
    )
    normalize: bool = Field(
        default=True,
        description="Whether to L2-normalize the output vectors",
    )


class EmbeddingResponse(BaseModel):
    """
    Response model for NB-SBERT embeddings.

    Returns a single 768-dimensional vector.
    """

    vector: List[float] = Field(
        ...,
        description="768-dimensional embedding vector",
    )
    dimension: int = Field(
        default=768,
        description="Dimension of the embedding vector",
    )
    model: str = Field(
        ...,
        description="Model used for embedding generation",
    )
    normalized: bool = Field(
        ...,
        description="Whether the vector is L2-normalized",
    )


class BatchEmbeddingResponse(BaseModel):
    """
    Response model for batch NB-SBERT embeddings.

    Returns multiple 768-dimensional vectors.
    """

    vectors: List[List[float]] = Field(
        ...,
        description="List of 768-dimensional embedding vectors",
    )
    dimension: int = Field(
        default=768,
        description="Dimension of each embedding vector",
    )
    count: int = Field(
        ...,
        description="Number of embeddings returned",
    )
    model: str = Field(
        ...,
        description="Model used for embedding generation",
    )
    normalized: bool = Field(
        ...,
        description="Whether vectors are L2-normalized",
    )


class EncoderResponse(BaseModel):
    """
    Response model for NB-BERT encoder output.

    Returns both pooled and token-level embeddings.
    """

    pooled_embedding: List[float] = Field(
        ...,
        description="768-dimensional pooled embedding ([CLS] token representation)",
    )
    token_embeddings: Optional[List[List[float]]] = Field(
        default=None,
        description="Per-token embeddings (each 768-dimensional)",
    )
    tokens: Optional[List[str]] = Field(
        default=None,
        description="Tokenized input text",
    )
    dimension: int = Field(
        default=768,
        description="Dimension of each embedding vector",
    )
    model: str = Field(
        ...,
        description="Model used for encoding",
    )
    normalized: bool = Field(
        ...,
        description="Whether vectors are L2-normalized",
    )


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(
        ...,
        description="Service status",
        examples=["healthy"],
    )
    sbert_model_loaded: bool = Field(
        ...,
        description="Whether NB-SBERT model is loaded",
    )
    bert_model_loaded: bool = Field(
        ...,
        description="Whether NB-BERT model is loaded",
    )
    device: str = Field(
        ...,
        description="Device being used (cpu/cuda/mps)",
    )
    version: str = Field(
        ...,
        description="Service version",
    )


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(
        ...,
        description="Error type",
    )
    message: str = Field(
        ...,
        description="Error message",
    )
    detail: Optional[str] = Field(
        default=None,
        description="Additional error details",
    )
