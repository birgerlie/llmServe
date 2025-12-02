#!/usr/bin/env python3
"""
Entry point script for running the llmServe application.

Usage:
    python run.py
    python run.py --host 0.0.0.0 --port 8000
"""

import argparse

import uvicorn

from app.config import get_settings


def main():
    """Run the llmServe application."""
    parser = argparse.ArgumentParser(
        description="Run the llmServe embedding service"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to (default: from settings)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (default: from settings)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes",
    )

    args = parser.parse_args()
    settings = get_settings()

    host = args.host or settings.host
    port = args.port or settings.port
    workers = args.workers or settings.workers

    print(f"Starting llmServe on {host}:{port}")
    print(f"API docs available at: http://{host}:{port}/docs")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=args.reload,
        workers=1 if args.reload else workers,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
