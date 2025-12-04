"""
gRPC server for NB-BERT/NB-SBERT embeddings.

Optimized for high-bandwidth connections like Thunderbolt 5.
Supports concurrent requests and streaming for large documents.
"""

import asyncio
from concurrent import futures
from typing import Optional

import grpc
from loguru import logger

from app import __version__
from app.config import Settings, get_settings
from app.grpc.service import create_servicer

# Import generated gRPC code
try:
    from app.grpc.generated import embedding_pb2_grpc
except ImportError:
    embedding_pb2_grpc = None


# Thunderbolt 5 optimized settings
# - Large message sizes for efficient batch transfers
# - High concurrency for parallel processing
# - Keepalive for persistent connections
GRPC_OPTIONS = [
    # Maximum message sizes (100MB for large batches)
    ("grpc.max_send_message_length", 100 * 1024 * 1024),
    ("grpc.max_receive_message_length", 100 * 1024 * 1024),

    # Keepalive settings for persistent connections
    ("grpc.keepalive_time_ms", 10000),  # Send keepalive every 10s
    ("grpc.keepalive_timeout_ms", 5000),  # Wait 5s for keepalive ack
    ("grpc.keepalive_permit_without_calls", True),  # Allow keepalive without active calls
    ("grpc.http2.max_pings_without_data", 0),  # Unlimited pings

    # Connection settings for high bandwidth
    ("grpc.http2.min_time_between_pings_ms", 5000),
    ("grpc.http2.min_ping_interval_without_data_ms", 5000),

    # Enable compression for smaller embeddings
    ("grpc.default_compression_algorithm", grpc.Compression.Gzip.value),
]


def serve(
    host: str = "0.0.0.0",
    port: int = 50051,
    max_workers: int = 10,
    settings: Optional[Settings] = None,
    block: bool = True,
) -> grpc.Server:
    """
    Start the gRPC server.

    Args:
        host: Host to bind to
        port: Port to bind to
        max_workers: Maximum number of worker threads
        settings: Application settings
        block: Whether to block until server stops

    Returns:
        The gRPC server instance
    """
    if embedding_pb2_grpc is None:
        raise ImportError(
            "gRPC generated files not found. Run 'scripts/generate_grpc.sh' first."
        )

    if settings is None:
        settings = get_settings()

    logger.info(f"Starting gRPC server v{__version__}")
    logger.info(f"Binding to {host}:{port}")
    logger.info(f"Max workers: {max_workers}")

    # Create server with thread pool for CPU-bound embedding operations
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=GRPC_OPTIONS,
    )

    # Add service
    servicer = create_servicer(settings)
    embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(servicer, server)

    # Bind to address
    address = f"{host}:{port}"
    server.add_insecure_port(address)

    # Start server
    server.start()
    logger.info(f"gRPC server started on {address}")

    if block:
        logger.info("Press Ctrl+C to stop")
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("Shutting down gRPC server...")
            server.stop(grace=5)
            logger.info("gRPC server stopped")

    return server


async def serve_async(
    host: str = "0.0.0.0",
    port: int = 50051,
    max_workers: int = 10,
    settings: Optional[Settings] = None,
) -> grpc.Server:
    """
    Start the gRPC server asynchronously.

    Useful for running alongside FastAPI in the same process.
    """
    if embedding_pb2_grpc is None:
        raise ImportError(
            "gRPC generated files not found. Run 'scripts/generate_grpc.sh' first."
        )

    if settings is None:
        settings = get_settings()

    server = grpc.aio.server(options=GRPC_OPTIONS)

    # Add service
    servicer = create_servicer(settings)
    embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(servicer, server)

    # Bind and start
    address = f"{host}:{port}"
    server.add_insecure_port(address)
    await server.start()

    logger.info(f"Async gRPC server started on {address}")

    return server


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the gRPC embedding server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=50051, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=10, help="Max worker threads")

    args = parser.parse_args()

    serve(host=args.host, port=args.port, max_workers=args.workers)
