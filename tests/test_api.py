"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_service_info(self, client):
        """Test root endpoint returns service information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "docs" in data


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "sbert_model_loaded" in data
        assert "bert_model_loaded" in data
        assert "device" in data
        assert "version" in data

    def test_readiness_check(self, client):
        """Test readiness check endpoint."""
        response = client.get("/api/v1/ready")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert "message" in data


class TestEmbeddingEndpoints:
    """Tests for embedding endpoints.

    Note: These tests may take time on first run as models are loaded.
    """

    @pytest.mark.slow
    def test_create_embedding(self, client):
        """Test creating a single embedding."""
        response = client.post(
            "/api/v1/embeddings",
            json={
                "text": "Oslo er hovedstaden i Norge.",
                "mode": "embedding",
                "output": "vector",
                "normalize": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "vector" in data
        assert len(data["vector"]) == 768
        assert data["dimension"] == 768
        assert data["normalized"] is True

    @pytest.mark.slow
    def test_create_embedding_empty_text(self, client):
        """Test embedding with empty text returns zero vector."""
        response = client.post(
            "/api/v1/embeddings",
            json={
                "text": "",
                "mode": "embedding",
                "normalize": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["vector"]) == 768
        # Empty text should return zero vector
        assert all(v == 0.0 for v in data["vector"])

    @pytest.mark.slow
    def test_batch_embedding(self, client):
        """Test batch embedding generation."""
        response = client.post(
            "/api/v1/embeddings/batch",
            json={
                "texts": [
                    "Første setning",
                    "Andre setning",
                    "Tredje setning",
                ],
                "mode": "embedding",
                "normalize": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "vectors" in data
        assert len(data["vectors"]) == 3
        assert data["count"] == 3
        assert all(len(v) == 768 for v in data["vectors"])

    def test_batch_embedding_exceeds_limit(self, client):
        """Test batch embedding with too many texts."""
        # Create more texts than max_batch_size (default 32)
        texts = [f"Setning {i}" for i in range(100)]
        response = client.post(
            "/api/v1/embeddings/batch",
            json={
                "texts": texts,
                "mode": "embedding",
            },
        )
        assert response.status_code == 400
        assert "exceeds maximum" in response.json()["detail"]

    def test_batch_embedding_encode_mode_rejected(self, client):
        """Test that batch endpoint rejects encode mode."""
        response = client.post(
            "/api/v1/embeddings/batch",
            json={
                "texts": ["Test"],
                "mode": "encode",
            },
        )
        assert response.status_code == 400
        assert "only supports mode='embedding'" in response.json()["detail"]

    @pytest.mark.slow
    def test_encoder_mode(self, client):
        """Test NB-BERT encoder mode."""
        response = client.post(
            "/api/v1/embeddings",
            json={
                "text": "Test av encoder",
                "mode": "encode",
                "output": "token_vectors",
                "normalize": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "pooled_embedding" in data
        assert len(data["pooled_embedding"]) == 768
        assert "token_embeddings" in data
        assert data["token_embeddings"] is not None

    @pytest.mark.slow
    def test_similarity_calculation(self, client):
        """Test similarity calculation between texts."""
        response = client.post(
            "/api/v1/embeddings/similarity",
            params={
                "text1": "Oslo er hovedstaden i Norge",
                "text2": "Hovedstaden i Norge heter Oslo",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "similarity" in data
        # Similar texts should have high similarity
        assert data["similarity"] > 0.5

    def test_invalid_request(self, client):
        """Test invalid request handling."""
        response = client.post(
            "/api/v1/embeddings",
            json={
                # Missing required 'text' field
                "mode": "embedding",
            },
        )
        assert response.status_code == 422  # Validation error
