"""gRPC server and service implementations."""

from app.grpc.service import EmbeddingGRPCService
from app.grpc.server import serve as serve_grpc

__all__ = ["EmbeddingGRPCService", "serve_grpc"]
