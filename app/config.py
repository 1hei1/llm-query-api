from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    ragflow_base_url: AnyHttpUrl = Field(
        default="http://192.168.148.107",
        description="Base URL for the RAGFlow API backend.",
    )
    ragflow_api_key: Optional[str] = Field(
        default=None,
        description="API key used to authenticate with RAGFlow.",
    )
    openai_base_url: Optional[AnyHttpUrl] = Field(
        default=None,
        description="Optional base URL for an OpenAI-compatible API.",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="API key for the OpenAI-compatible API.",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="Default model name used for the OpenAI-compatible chat API.",
    )
    http_timeout: float = Field(
        default=30.0,
        ge=0,
        description="Default timeout (seconds) for outbound HTTP requests.",
    )
    ragflow_retry_attempts: int = Field(
        default=3,
        ge=1,
        description="Number of retry attempts for RAGFlow requests when recoverable errors occur.",
    )
    ragflow_retry_wait: float = Field(
        default=0.5,
        ge=0,
        description="Delay (seconds) between retries for RAGFlow requests.",
    )
    rag_answer_similarity_threshold: float = Field(
        default=0.2,
        ge=0,
        description="Similarity threshold applied when building RAG answers.",
    )
    rag_answer_vector_similarity_weight: float = Field(
        default=0.3,
        ge=0,
        description="Vector similarity weight applied when retrieving RAG context.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return application settings (cached)."""

    return Settings()
