"""Tests for configuration."""

import os
from unittest.mock import patch

import pytest

from app.config import Settings, get_settings


class TestSettings:
    """Tests for Settings class."""

    def test_default_values(self):
        """Test default configuration values."""
        settings = Settings()
        assert settings.app_name == "llmServe"
        assert settings.debug is False
        assert settings.device == "cpu"
        assert settings.max_batch_size == 32
        assert settings.max_sequence_length == 512
        assert settings.port == 8000

    def test_model_names(self):
        """Test default model names."""
        settings = Settings()
        assert settings.sbert_model_name == "NbAiLab/nb-sbert-base"
        assert settings.bert_model_name == "NbAiLab/nb-bert-base"

    def test_environment_override(self):
        """Test that environment variables override defaults."""
        with patch.dict(os.environ, {
            "APP_NAME": "TestService",
            "DEBUG": "true",
            "DEVICE": "cuda",
            "PORT": "9000",
        }):
            settings = Settings()
            assert settings.app_name == "TestService"
            assert settings.debug is True
            assert settings.device == "cuda"
            assert settings.port == 9000

    def test_get_settings_caching(self):
        """Test that get_settings returns cached instance."""
        # Clear cache first
        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_valid_device_values(self):
        """Test valid device values."""
        for device in ["cpu", "cuda", "mps"]:
            with patch.dict(os.environ, {"DEVICE": device}):
                settings = Settings()
                assert settings.device == device
