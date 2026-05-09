"""Runtime settings for the sol01 command line tools and LLM calls."""

import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import BaseModel, Field, model_validator

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-pro"
DEFAULT_PROVIDER_ONLY = "deepseek"
DEFAULT_CONCURRENCY = 4
DEFAULT_SCHEMA_RETRIEVAL_VERSION = "hybrid_v1"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_SCHEMA_EMBEDDING_MODEL = "qwen3-embedding:4b"
DEFAULT_SCHEMA_RERANKER_MODEL = "qwen3-reranker:4b"
DEFAULT_RETRIEVAL_CHUNK_TOP_K = 80
DEFAULT_RETRIEVAL_RERANK_TOP_K = 20
DEFAULT_RETRIEVAL_OBJECT_TOP_K = 12
DEFAULT_RETRIEVAL_FAMILY_TOP_K = 8
DEFAULT_FAMILY_SIMILARITY_THRESHOLD = 0.82
DEFAULT_MAX_LINKED_DOC_CHARS = 6000
DEFAULT_MAX_SCHEMA_PROMPT_CHARS = 24000
METHOD_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOTENV_PATH = METHOD_ROOT / ".env"


class RuntimeConfig(BaseModel):
    """Validated runtime options shared by the CLI, coordinator, and LLM client."""

    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    provider_only: str = DEFAULT_PROVIDER_ONLY
    allow_fallbacks: bool = False
    concurrency: int = Field(default=DEFAULT_CONCURRENCY, ge=1)

    @classmethod
    def from_env(
        cls,
        *,
        require_api_key: bool = False,
        dotenv_path: Path | None = None,
        concurrency: int | None = None,
    ) -> "RuntimeConfig":
        """Load settings from the shell, with optional local .env support."""

        _load_local_dotenv(dotenv_path)

        config = cls(
            api_key=_env_first("OPENROUTER_API_KEY", "LLM_API_KEY"),
            base_url=_env_first("OPENROUTER_BASE_URL", "LLM_BASE_URL") or DEFAULT_BASE_URL,
            model=_env_first("OPENROUTER_MODEL", "LLM_MODEL") or DEFAULT_MODEL,
            provider_only=_env_first("OPENROUTER_PROVIDER_ONLY") or DEFAULT_PROVIDER_ONLY,
            allow_fallbacks=_env_bool("OPENROUTER_ALLOW_FALLBACKS", default=False),
            concurrency=(
                concurrency
                if concurrency is not None
                else _env_positive_int("SOL01_CONCURRENCY", default=DEFAULT_CONCURRENCY)
            ),
        )
        if require_api_key and not config.api_key:
            raise ValueError("API key is required for live LLM runs")
        return config

    @property
    def provider_routing(self) -> dict[str, Any]:
        """Return the OpenRouter provider payload used on every LLM request."""

        return {
            "provider": {
                "only": [self.provider_only],
                "allow_fallbacks": self.allow_fallbacks,
            }
        }

    @model_validator(mode="after")
    def enforce_openrouter_policy(self) -> "RuntimeConfig":
        """Keep runs pinned to DeepSeek and fail instead of falling back silently."""

        if self.provider_only != DEFAULT_PROVIDER_ONLY:
            raise ValueError("provider_only must be deepseek")
        if self.allow_fallbacks:
            raise ValueError("provider fallback is disabled for sol01")
        return self


class SchemaRetrievalConfig(BaseModel):
    """Local schema-retrieval settings used before LLM planning."""

    schema_retrieval_version: str = DEFAULT_SCHEMA_RETRIEVAL_VERSION
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    embedding_model: str = DEFAULT_SCHEMA_EMBEDDING_MODEL
    reranker_model: str = DEFAULT_SCHEMA_RERANKER_MODEL
    chunk_top_k: int = Field(default=DEFAULT_RETRIEVAL_CHUNK_TOP_K, ge=1)
    rerank_top_k: int = Field(default=DEFAULT_RETRIEVAL_RERANK_TOP_K, ge=1)
    object_top_k: int = Field(default=DEFAULT_RETRIEVAL_OBJECT_TOP_K, ge=1)
    family_top_k: int = Field(default=DEFAULT_RETRIEVAL_FAMILY_TOP_K, ge=1)
    family_similarity_threshold: float = Field(
        default=DEFAULT_FAMILY_SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
    )
    max_linked_doc_chars: int = Field(default=DEFAULT_MAX_LINKED_DOC_CHARS, ge=1)
    max_schema_prompt_chars: int = Field(default=DEFAULT_MAX_SCHEMA_PROMPT_CHARS, ge=1)

    @classmethod
    def from_env(cls, *, dotenv_path: Path | None = None) -> "SchemaRetrievalConfig":
        """Load schema retrieval settings from the shell or one local .env file."""

        _load_local_dotenv(dotenv_path)

        return cls(
            schema_retrieval_version=(
                _env_first("SOL01_SCHEMA_RETRIEVAL_VERSION") or DEFAULT_SCHEMA_RETRIEVAL_VERSION
            ),
            ollama_base_url=_env_first("SOL01_OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL,
            embedding_model=(
                _env_first("SOL01_SCHEMA_EMBEDDING_MODEL") or DEFAULT_SCHEMA_EMBEDDING_MODEL
            ),
            reranker_model=(
                _env_first("SOL01_SCHEMA_RERANKER_MODEL") or DEFAULT_SCHEMA_RERANKER_MODEL
            ),
            chunk_top_k=_env_positive_int(
                "SOL01_SCHEMA_CHUNK_TOP_K",
                default=DEFAULT_RETRIEVAL_CHUNK_TOP_K,
            ),
            rerank_top_k=_env_positive_int(
                "SOL01_SCHEMA_RERANK_TOP_K",
                default=DEFAULT_RETRIEVAL_RERANK_TOP_K,
            ),
            object_top_k=_env_positive_int(
                "SOL01_SCHEMA_OBJECT_TOP_K",
                default=DEFAULT_RETRIEVAL_OBJECT_TOP_K,
            ),
            family_top_k=_env_positive_int(
                "SOL01_SCHEMA_FAMILY_TOP_K",
                default=DEFAULT_RETRIEVAL_FAMILY_TOP_K,
            ),
            family_similarity_threshold=_env_unit_float(
                "SOL01_SCHEMA_FAMILY_SIMILARITY_THRESHOLD",
                default=DEFAULT_FAMILY_SIMILARITY_THRESHOLD,
            ),
            max_linked_doc_chars=_env_positive_int(
                "SOL01_SCHEMA_MAX_LINKED_DOC_CHARS",
                default=DEFAULT_MAX_LINKED_DOC_CHARS,
            ),
            max_schema_prompt_chars=_env_positive_int(
                "SOL01_SCHEMA_MAX_PROMPT_CHARS",
                default=DEFAULT_MAX_SCHEMA_PROMPT_CHARS,
            ),
        )


def _env_first(*names: str) -> str | None:
    """Return the first non-empty environment value from the given names."""

    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _load_local_dotenv(dotenv_path: Path | None) -> None:
    """Load one local .env file without overwriting real non-empty shell variables."""

    if dotenv_path is None or not dotenv_path.exists():
        return
    for name, value in dotenv_values(dotenv_path).items():
        if value is None:
            continue
        current = os.environ.get(name)
        if current is None or not current.strip():
            os.environ[name] = value


def _env_bool(name: str, *, default: bool) -> bool:
    """Parse a boolean environment variable with clear accepted values."""

    value = os.environ.get(name)
    if value is None or not value.strip():
        return default

    normalized = value.strip().lower()
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    raise ValueError(f"{name} must be a boolean value")


def _env_positive_int(name: str, *, default: int) -> int:
    """Parse a positive integer environment variable with a clear error."""

    value = os.environ.get(name)
    if value is None or not value.strip():
        return default

    try:
        parsed = int(value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc

    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _env_unit_float(name: str, *, default: float) -> float:
    """Parse a float between 0 and 1 inclusive from the environment."""

    value = os.environ.get(name)
    if value is None or not value.strip():
        return default

    try:
        parsed = float(value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number between 0 and 1") from exc

    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be a number between 0 and 1")
    return parsed
