from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import AliasChoices, AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPServerSettings(BaseSettings):
    """Settings for the read-only MCP server."""

    llm_api_base_url: AnyHttpUrl = Field(
        default="http://127.0.0.1:8000",
        description="Base URL for the upstream llm-query-api service.",
    )
    llm_api_key: SecretStr = Field(
        description="API key used to authenticate with the upstream service.",
    )
    http_timeout: float = Field(
        default=30.0,
        ge=0.5,
        description="Timeout in seconds for outbound HTTP requests.",
        validation_alias=AliasChoices("MCP_HTTP_TIMEOUT", "HTTP_TIMEOUT", "http_timeout"),
    )
    retry_attempts: int = Field(
        default=3,
        ge=1,
        description="Number of retry attempts for recoverable HTTP errors.",
        validation_alias=AliasChoices("MCP_RETRY_ATTEMPTS", "RETRY_ATTEMPTS", "retry_attempts"),
    )
    retry_wait: float = Field(
        default=0.5,
        ge=0.0,
        description="Delay in seconds between retry attempts.",
        validation_alias=AliasChoices("MCP_RETRY_WAIT", "RETRY_WAIT", "retry_wait"),
    )
    rate_limit_capacity: int = Field(
        default=10,
        ge=1,
        description="Default number of requests permitted per interval per tool.",
        validation_alias=AliasChoices("MCP_RATE_LIMIT_CAPACITY", "RATE_LIMIT_CAPACITY", "rate_limit_capacity"),
    )
    rate_limit_interval_seconds: float = Field(
        default=60.0,
        gt=0.0,
        description="Interval length in seconds for rate limiting buckets.",
        validation_alias=AliasChoices("MCP_RATE_LIMIT_INTERVAL_SECONDS", "RATE_LIMIT_INTERVAL_SECONDS", "rate_limit_interval_seconds"),
    )
    tool_rate_limits: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Optional per-tool overrides for rate limits. "
            "Example: {'list_glossaries': 20, 'search_terms': 5}."
        ),
        validation_alias=AliasChoices("MCP_TOOL_RATE_LIMITS", "TOOL_RATE_LIMITS", "tool_rate_limits"),
    )
    max_query_length: int = Field(
        default=256,
        ge=1,
        description="Maximum number of characters permitted in free-text queries.",
        validation_alias=AliasChoices("MCP_MAX_QUERY_LENGTH", "MAX_QUERY_LENGTH", "max_query_length"),
    )
    max_terms: int = Field(
        default=10,
        ge=1,
        description="Maximum number of terms accepted for definition retrieval.",
        validation_alias=AliasChoices("MCP_MAX_TERMS", "MAX_TERMS", "max_terms"),
    )
    max_term_length: int = Field(
        default=128,
        ge=1,
        description="Maximum number of characters permitted per individual term.",
        validation_alias=AliasChoices("MCP_MAX_TERM_LENGTH", "MAX_TERM_LENGTH", "max_term_length"),
    )
    dataset_id_pattern: str = Field(
        default=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
        description="Regular expression used to validate dataset identifiers.",
        validation_alias=AliasChoices("MCP_DATASET_ID_PATTERN", "DATASET_ID_PATTERN", "dataset_id_pattern"),
    )
    search_top_k: int = Field(
        default=8,
        ge=1,
        le=1024,
        description="Default top_k value for glossary search operations.",
        validation_alias=AliasChoices("MCP_SEARCH_TOP_K", "SEARCH_TOP_K", "search_top_k"),
    )
    definition_top_k: int = Field(
        default=12,
        ge=1,
        le=1024,
        description="Default top_k for definition retrieval across multiple terms.",
        validation_alias=AliasChoices("MCP_DEFINITION_TOP_K", "DEFINITION_TOP_K", "definition_top_k"),
    )
    similarity_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Similarity threshold applied to retrieval requests.",
        validation_alias=AliasChoices("MCP_SIMILARITY_THRESHOLD", "SIMILARITY_THRESHOLD", "similarity_threshold"),
    )
    vector_similarity_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Vector similarity weighting applied to retrieval requests.",
        validation_alias=AliasChoices("MCP_VECTOR_SIMILARITY_WEIGHT", "VECTOR_SIMILARITY_WEIGHT", "vector_similarity_weight"),
    )
    log_level: str = Field(
        default="INFO",
        description="Log level for the MCP server (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
        validation_alias=AliasChoices("MCP_LOG_LEVEL", "LOG_LEVEL", "log_level"),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("tool_rate_limits", mode="before")
    @classmethod
    def _parse_rate_limits(cls, value: Any) -> dict[str, int]:
        if value in (None, ""):
            return {}
        if isinstance(value, dict):
            return {str(k): int(v) for k, v in value.items()}
        if isinstance(value, str):
            import json

            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("tool_rate_limits must be valid JSON") from exc
            if not isinstance(parsed, dict):
                raise ValueError("tool_rate_limits must decode to a dictionary")
            return {str(k): int(v) for k, v in parsed.items()}
        raise ValueError("Unsupported type for tool_rate_limits")


@lru_cache(maxsize=1)
def get_settings() -> MCPServerSettings:
    """Return cached MCP server settings."""

    return MCPServerSettings()
