"""Generated gRPC code for embedding service.

Run `scripts/generate_grpc.sh` to regenerate from proto files.
"""

try:
    from app.grpc.generated.embedding_pb2 import (
        Mode,
        OutputType,
        EmbeddingRequest,
        BatchEmbeddingRequest,
        StreamingEmbeddingRequest,
        StreamConfig,
        TextChunk,
        Vector,
        EmbeddingResponse,
        EncoderResponse,
        BatchEmbeddingResponse,
        StreamingEmbeddingResponse,
        SimilarityRequest,
        SimilarityResponse,
        BulkSimilarityRequest,
        BulkSimilarityResponse,
        SimilarityResult,
        HealthRequest,
        HealthResponse,
    )
    from app.grpc.generated.embedding_pb2_grpc import (
        EmbeddingServiceServicer,
        EmbeddingServiceStub,
        add_EmbeddingServiceServicer_to_server,
    )

    __all__ = [
        "Mode",
        "OutputType",
        "EmbeddingRequest",
        "BatchEmbeddingRequest",
        "StreamingEmbeddingRequest",
        "StreamConfig",
        "TextChunk",
        "Vector",
        "EmbeddingResponse",
        "EncoderResponse",
        "BatchEmbeddingResponse",
        "StreamingEmbeddingResponse",
        "SimilarityRequest",
        "SimilarityResponse",
        "BulkSimilarityRequest",
        "BulkSimilarityResponse",
        "SimilarityResult",
        "HealthRequest",
        "HealthResponse",
        "EmbeddingServiceServicer",
        "EmbeddingServiceStub",
        "add_EmbeddingServiceServicer_to_server",
    ]
except ImportError:
    # Generated files not yet created
    import warnings
    warnings.warn(
        "gRPC generated files not found. Run 'scripts/generate_grpc.sh' to generate them.",
        ImportWarning,
    )
