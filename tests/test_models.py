"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from app.models.embedding import (
    BatchEmbeddingRequest,
    EmbeddingMode,
    EmbeddingOutput,
    EmbeddingRequest,
    EmbeddingResponse,
    EncoderResponse,
    HealthResponse,
)


class TestEmbeddingRequest:
    """Tests for EmbeddingRequest model."""

    def test_valid_request(self):
        """Test valid embedding request."""
        request = EmbeddingRequest(
            text="Test tekst",
            mode=EmbeddingMode.EMBEDDING,
            output=EmbeddingOutput.VECTOR,
            normalize=True,
        )
        assert request.text == "Test tekst"
        assert request.mode == EmbeddingMode.EMBEDDING
        assert request.output == EmbeddingOutput.VECTOR
        assert request.normalize is True

    def test_default_values(self):
        """Test default values are applied."""
        request = EmbeddingRequest(text="Test")
        assert request.mode == EmbeddingMode.EMBEDDING
        assert request.output == EmbeddingOutput.VECTOR
        assert request.normalize is True

    def test_empty_text_allowed(self):
        """Test that empty text is allowed (handled by service)."""
        request = EmbeddingRequest(text="")
        assert request.text == ""

    def test_encode_mode(self):
        """Test encode mode request."""
        request = EmbeddingRequest(
            text="Test",
            mode=EmbeddingMode.ENCODE,
            output=EmbeddingOutput.TOKEN_VECTORS,
        )
        assert request.mode == EmbeddingMode.ENCODE
        assert request.output == EmbeddingOutput.TOKEN_VECTORS


class TestBatchEmbeddingRequest:
    """Tests for BatchEmbeddingRequest model."""

    def test_valid_batch_request(self):
        """Test valid batch request."""
        request = BatchEmbeddingRequest(
            texts=["Tekst 1", "Tekst 2", "Tekst 3"],
            mode=EmbeddingMode.EMBEDDING,
            normalize=True,
        )
        assert len(request.texts) == 3
        assert request.normalize is True

    def test_empty_texts_invalid(self):
        """Test that empty texts list is invalid."""
        with pytest.raises(ValidationError):
            BatchEmbeddingRequest(texts=[])


class TestEmbeddingResponse:
    """Tests for EmbeddingResponse model."""

    def test_valid_response(self):
        """Test valid embedding response."""
        vector = [0.1] * 768
        response = EmbeddingResponse(
            vector=vector,
            dimension=768,
            model="NbAiLab/nb-sbert-base",
            normalized=True,
        )
        assert len(response.vector) == 768
        assert response.dimension == 768
        assert response.normalized is True


class TestEncoderResponse:
    """Tests for EncoderResponse model."""

    def test_valid_encoder_response(self):
        """Test valid encoder response."""
        pooled = [0.1] * 768
        tokens = [[0.1] * 768, [0.2] * 768]
        response = EncoderResponse(
            pooled_embedding=pooled,
            token_embeddings=tokens,
            tokens=["[CLS]", "test", "[SEP]"],
            dimension=768,
            model="NbAiLab/nb-bert-base",
            normalized=True,
        )
        assert len(response.pooled_embedding) == 768
        assert len(response.token_embeddings) == 2
        assert response.tokens == ["[CLS]", "test", "[SEP]"]

    def test_encoder_response_without_tokens(self):
        """Test encoder response without token embeddings."""
        pooled = [0.1] * 768
        response = EncoderResponse(
            pooled_embedding=pooled,
            dimension=768,
            model="NbAiLab/nb-bert-base",
            normalized=True,
        )
        assert response.token_embeddings is None
        assert response.tokens is None


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_valid_health_response(self):
        """Test valid health response."""
        response = HealthResponse(
            status="healthy",
            sbert_model_loaded=True,
            bert_model_loaded=False,
            device="cpu",
            version="0.1.0",
        )
        assert response.status == "healthy"
        assert response.sbert_model_loaded is True
        assert response.bert_model_loaded is False
