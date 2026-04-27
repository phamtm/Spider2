"""Runtime settings for the sol01 command line tools and LLM calls."""

import os
from typing import Any

from pydantic import BaseModel, Field, model_validator


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-pro"
DEFAULT_PROVIDER_ONLY = "deepseek"


class RuntimeConfig(BaseModel):
    """Validated runtime options shared by the CLI, coordinator, and LLM client."""

    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    provider_only: str = DEFAULT_PROVIDER_ONLY
    allow_fallbacks: bool = False
    concurrency: int = Field(default=2, ge=1)
    max_schema_tables: int = Field(default=12, ge=1)

    @classmethod
    def from_env(cls, *, require_api_key: bool = False) -> "RuntimeConfig":
        """Load OpenRouter settings, with generic LLM_* names as fallbacks."""

        config = cls(
            api_key=_env_first("OPENROUTER_API_KEY", "LLM_API_KEY"),
            base_url=_env_first("OPENROUTER_BASE_URL", "LLM_BASE_URL")
            or DEFAULT_BASE_URL,
            model=_env_first("OPENROUTER_MODEL", "LLM_MODEL") or DEFAULT_MODEL,
            provider_only=_env_first("OPENROUTER_PROVIDER_ONLY")
            or DEFAULT_PROVIDER_ONLY,
            allow_fallbacks=_env_bool("OPENROUTER_ALLOW_FALLBACKS", default=False),
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


def _env_first(*names: str) -> str | None:
    """Return the first non-empty environment value from the given names."""

    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


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
