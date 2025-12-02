#!/bin/bash
# Generate Python gRPC code from proto files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROTO_DIR="$PROJECT_ROOT/proto"
OUTPUT_DIR="$PROJECT_ROOT/app/grpc/generated"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate Python code
python -m grpc_tools.protoc \
    -I"$PROTO_DIR" \
    --python_out="$OUTPUT_DIR" \
    --pyi_out="$OUTPUT_DIR" \
    --grpc_python_out="$OUTPUT_DIR" \
    "$PROTO_DIR/embedding.proto"

# Fix imports in generated files (grpc_tools generates relative imports incorrectly)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' 's/import embedding_pb2/from app.grpc.generated import embedding_pb2/' "$OUTPUT_DIR/embedding_pb2_grpc.py"
else
    # Linux
    sed -i 's/import embedding_pb2/from app.grpc.generated import embedding_pb2/' "$OUTPUT_DIR/embedding_pb2_grpc.py"
fi

# Create __init__.py
cat > "$OUTPUT_DIR/__init__.py" << 'EOF'
"""Generated gRPC code for embedding service."""

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
EOF

echo "gRPC code generated successfully in $OUTPUT_DIR"
