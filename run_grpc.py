#!/usr/bin/env python3
"""
Entry point for running the gRPC embedding server.

Optimized for high-bandwidth connections like Thunderbolt 5.

Usage:
    python run_grpc.py
    python run_grpc.py --host 0.0.0.0 --port 50051
    python run_grpc.py --thunderbolt  # Uses Thunderbolt IP range
"""

import argparse
import sys

from loguru import logger

from app.config import get_settings


def get_thunderbolt_ip() -> str:
    """
    Get the Thunderbolt bridge interface IP.

    On macOS, Thunderbolt networking typically creates a bridge interface
    with IPs in the 169.254.x.x range (link-local) or a configured subnet.
    """
    import socket

    # Common Thunderbolt IP ranges
    thunderbolt_prefixes = [
        "169.254.",  # Link-local (auto-configured)
        "192.168.2.",  # Common manual config
        "10.0.0.",  # Alternative manual config
    ]

    try:
        # Get all IP addresses
        hostname = socket.gethostname()
        ips = socket.getaddrinfo(hostname, None, socket.AF_INET)

        for info in ips:
            ip = info[4][0]
            for prefix in thunderbolt_prefixes:
                if ip.startswith(prefix):
                    return ip

    except Exception as e:
        logger.warning(f"Could not detect Thunderbolt IP: {e}")

    return "0.0.0.0"  # Fallback to all interfaces


def main():
    """Run the gRPC server."""
    parser = argparse.ArgumentParser(
        description="Run the gRPC embedding server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard server on all interfaces
    python run_grpc.py

    # Bind to specific Thunderbolt IP
    python run_grpc.py --host 192.168.2.1 --port 50051

    # Auto-detect Thunderbolt interface
    python run_grpc.py --thunderbolt

    # High concurrency for batch processing
    python run_grpc.py --workers 20
        """,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Port to bind to (default: 50051)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of worker threads (default: 10)",
    )
    parser.add_argument(
        "--thunderbolt",
        action="store_true",
        help="Auto-detect and use Thunderbolt interface",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload models before starting server",
    )

    args = parser.parse_args()

    # Handle Thunderbolt mode
    host = args.host
    if args.thunderbolt:
        host = get_thunderbolt_ip()
        logger.info(f"Detected Thunderbolt IP: {host}")

    # Get settings
    settings = get_settings()

    # Preload models if requested
    if args.preload:
        logger.info("Preloading models...")
        from app.services.embedding_service import get_embedding_service

        service = get_embedding_service(settings)
        service.load_sbert_model()
        logger.info("Models preloaded")

    # Import and run server
    try:
        from app.grpc.server import serve

        serve(
            host=host,
            port=args.port,
            max_workers=args.workers,
            settings=settings,
            block=True,
        )
    except ImportError as e:
        logger.error(f"Failed to start gRPC server: {e}")
        logger.error("Run 'scripts/generate_grpc.sh' to generate gRPC code first")
        sys.exit(1)


if __name__ == "__main__":
    main()
