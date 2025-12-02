"""
Configuration management for llmServe.

Uses Pydantic Settings for environment-based configuration.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application settings
    app_name: str = "llmServe"
    debug: bool = False

    # Model settings
    sbert_model_name: str = "NbAiLab/nb-sbert-base"
    bert_model_name: str = "NbAiLab/nb-bert-base"
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    max_batch_size: int = 32
    max_sequence_length: int = 512

    # REST API settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # gRPC settings (for Thunderbolt connections)
    grpc_host: str = "0.0.0.0"
    grpc_port: int = 50051
    grpc_workers: int = 10

    # Logging
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
