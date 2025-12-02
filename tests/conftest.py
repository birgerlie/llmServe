"""Pytest configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def sample_norwegian_texts():
    """Sample Norwegian texts for testing."""
    return [
        "Oslo er hovedstaden i Norge.",
        "Bergen ligger på vestlandet.",
        "Sykehjemmet tilbyr korttidsopphold og langtidsopphold.",
        "Kommunen har ansvar for eldreomsorg.",
        "Helsevesenet i Norge er offentlig finansiert.",
    ]


@pytest.fixture
def sample_embedding_request():
    """Sample embedding request payload."""
    return {
        "text": "Dette er en test av embedding-generering.",
        "mode": "embedding",
        "output": "vector",
        "normalize": True,
    }
