"""Load prompt files and run structured LLM calls through OpenRouter."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from sol01.infra.config import RuntimeConfig
from sol01.infra.logging import get_logger
from sol01.llm.llm_logging import (
    SCHEMA_VERSION,
    LLMCallLogger,
    duration_ms,
    isoformat_utc,
    serialize_error,
    serialize_output,
    utc_now,
)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
OutputT = TypeVar("OutputT", bound=BaseModel)
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_MODEL_ATTEMPTS = 3
logger = get_logger(__name__)
Agent: Any | None = None

if TYPE_CHECKING:
    from pydantic_ai.models import Model


@dataclass(frozen=True)
class PromptSpec:
    """Prompt text plus its stable SHA-256 hash."""

    name: str
    text: str
    sha256: str


class LLMClient:
    """Thin wrapper around prompt loading and structured Pydantic AI calls."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        prompts_dir: Path = PROMPTS_DIR,
        call_logger: LLMCallLogger | None = None,
    ) -> None:
        self.config = config
        self.prompts_dir = prompts_dir
        self.call_logger = call_logger

    def load_prompt(self, prompt_name: str) -> PromptSpec:
        """Read one markdown prompt and compute its reproducible content hash."""

        prompt = _load_prompt_spec(str(self.prompts_dir.resolve()), prompt_name)
        logger.debug("prompt loaded", prompt_name=prompt_name, prompt_hash=prompt.sha256)
        return prompt

    def run_structured(
        self,
        user_prompt: str,
        *,
        prompt_name: str,
        output_type: type[OutputT],
        model: Model | str | None = None,
    ) -> OutputT:
        """Run one structured prompt and return the validated Pydantic output."""

        prompt = self.load_prompt(prompt_name)
        return self._run_structured_prompt(
            user_prompt,
            prompt=prompt,
            output_type=output_type,
            model=model,
        )

    def run_structured_with_prompt(
        self,
        user_prompt: str,
        *,
        prompt: PromptSpec,
        output_type: type[OutputT],
        model: Model | str | None = None,
    ) -> OutputT:
        """Run one structured prompt using an already loaded prompt spec."""

        return self._run_structured_prompt(
            user_prompt,
            prompt=prompt,
            output_type=output_type,
            model=model,
        )

    def _run_structured_prompt(
        self,
        user_prompt: str,
        *,
        prompt: PromptSpec,
        output_type: type[OutputT],
        model: Any | None,
    ) -> OutputT:
        """Run and optionally log one structured prompt."""

        resolved_model = _resolve_model(model, config=self.config)
        agent = _agent_class()(
            model=resolved_model,
            system_prompt=prompt.text,
            output_type=_structured_output(output_type),
        )
        logger.debug(
            "llm request start",
            prompt_name=prompt.name,
            output_type=output_type.__name__,
        )
        started_at = utc_now()
        started_perf = time.perf_counter()
        attempts: list[dict[str, Any]] = []
        try:
            result = _run_agent_sync(
                agent,
                user_prompt,
                prompt_name=prompt.name,
                attempts=attempts,
            )
            output = result.output
            if not isinstance(output, output_type):
                raise TypeError(f"Expected {output_type.__name__}, got {type(output).__name__}")
        except Exception as exc:
            self._write_call_log(
                status="error",
                started_at=started_at,
                started_perf=started_perf,
                prompt=prompt,
                output_type=output_type,
                model_label=_model_label(model, resolved_model=resolved_model, config=self.config),
                user_prompt=user_prompt,
                attempts=attempts,
                output=None,
                error=exc,
            )
            raise

        self._write_call_log(
            status="success",
            started_at=started_at,
            started_perf=started_perf,
            prompt=prompt,
            output_type=output_type,
            model_label=_model_label(model, resolved_model=resolved_model, config=self.config),
            user_prompt=user_prompt,
            attempts=attempts,
            output=output,
            error=None,
        )
        logger.debug(
            "llm request complete",
            prompt_name=prompt.name,
            output_type=output_type.__name__,
        )
        return output

    def _write_call_log(
        self,
        *,
        status: str,
        started_at: datetime,
        started_perf: float,
        prompt: PromptSpec,
        output_type: type[OutputT],
        model_label: str,
        user_prompt: str,
        attempts: list[dict[str, Any]],
        output: OutputT | None,
        error: BaseException | None,
    ) -> None:
        """Append one local LLM debug record when logging is enabled."""

        if self.call_logger is None:
            return

        sequence = self.call_logger.next_sequence()
        record = {
            "schema_version": SCHEMA_VERSION,
            "call_id": f"{sequence:04d}-{prompt.name}",
            "sequence": sequence,
            "started_at": isoformat_utc(started_at),
            "completed_at": isoformat_utc(utc_now()),
            "duration_ms": duration_ms(started_perf),
            "status": status,
            "prompt_name": prompt.name,
            "prompt_sha256": prompt.sha256,
            "output_type": output_type.__name__,
            "model": model_label,
            "base_url": self.config.base_url,
            "provider_routing": self.config.provider_routing,
            "request": {
                "system_prompt": prompt.text,
                "user_prompt": user_prompt,
                "output_schema": output_type.__name__,
            },
            "response": (
                {
                    "validated_output": serialize_output(output),
                }
                if output is not None
                else None
            ),
            "attempts": attempts,
            "error": serialize_error(error) if error is not None else None,
        }
        self.call_logger.write(record)


def prompt_sha256(text: str) -> str:
    """Hash prompt text so later traces can record exact prompt versions."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@cache
def _load_prompt_spec(prompts_dir: str, prompt_name: str) -> PromptSpec:
    """Load one prompt file and keep the parsed result in memory."""

    prompt_path = _prompt_path(prompt_name, prompts_dir=Path(prompts_dir))
    text = prompt_path.read_text(encoding="utf-8")
    return PromptSpec(
        name=prompt_name,
        text=text,
        sha256=prompt_sha256(text),
    )


def build_model_settings(config: RuntimeConfig) -> dict[str, Any]:
    """Build the OpenRouter request body additions required by this solver."""

    return {"extra_body": config.provider_routing}


def build_model(config: RuntimeConfig, *, model_name: str | None = None) -> Any:
    """Create the OpenAI-compatible chat model that points at OpenRouter."""

    resolved_model_name = model_name or config.model
    from pydantic_ai.models.openai import OpenAIChatModel  # noqa: PLC0415
    from pydantic_ai.profiles.openai import OpenAIModelProfile  # noqa: PLC0415
    from pydantic_ai.providers.openai import OpenAIProvider  # noqa: PLC0415

    provider = OpenAIProvider(
        base_url=config.base_url,
        api_key=config.api_key,
    )
    profile = replace(
        OpenAIModelProfile.from_profile(provider.model_profile(resolved_model_name)),
        openai_chat_send_back_thinking_parts="tags",
    )
    return OpenAIChatModel(
        resolved_model_name,
        provider=provider,
        profile=profile,
        settings=build_model_settings(config),
    )


def _prompt_path(prompt_name: str, *, prompts_dir: Path) -> Path:
    """Resolve a prompt name to its markdown file path."""

    file_name = prompt_name if prompt_name.endswith(".md") else f"{prompt_name}.md"
    return prompts_dir / file_name


def _resolve_model(model: Model | str | None, *, config: RuntimeConfig) -> Model:
    """Keep live model overrides on the OpenRouter path while allowing test models."""

    if model is None:
        return build_model(config)
    if isinstance(model, str):
        return build_model(config, model_name=model)
    return model


def _structured_output(output_type: type[OutputT]) -> Any:
    """Use prompted JSON output so DeepSeek does not need tool calling support."""

    from pydantic_ai import PromptedOutput  # noqa: PLC0415

    return PromptedOutput(output_type)


def _agent_class() -> Any:
    """Return the live Agent class, importing it only when needed."""

    global Agent
    if Agent is None:
        from pydantic_ai import Agent as _Agent  # noqa: PLC0415

        Agent = _Agent
    return Agent


def _run_agent_sync(
    agent: Any,
    user_prompt: str,
    *,
    prompt_name: str,
    attempts: list[dict[str, Any]],
) -> Any:
    """Retry transient provider errors a few times before failing the call."""

    from pydantic_ai.exceptions import ModelHTTPError  # noqa: PLC0415

    for attempt in range(1, MAX_MODEL_ATTEMPTS + 1):
        attempt_started = time.perf_counter()
        try:
            result = agent.run_sync(user_prompt)
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "success",
                    "duration_ms": duration_ms(attempt_started),
                }
            )
            return result
        except ModelHTTPError as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "error",
                    "duration_ms": duration_ms(attempt_started),
                    "error": serialize_error(exc),
                }
            )
            logger.warning(
                "llm retry",
                prompt_name=prompt_name,
                attempt=attempt,
                status_code=exc.status_code,
            )
            if exc.status_code not in TRANSIENT_STATUS_CODES or attempt >= MAX_MODEL_ATTEMPTS:
                raise
            time.sleep(2 ** (attempt - 1))
        except Exception as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "error",
                    "duration_ms": duration_ms(attempt_started),
                    "error": serialize_error(exc),
                }
            )
            raise


def _model_label(
    model: Model | str | None,
    *,
    resolved_model: Any,
    config: RuntimeConfig,
) -> str:
    """Return log-safe model identity without exposing credentials."""

    if isinstance(model, str):
        return model
    model_name = getattr(resolved_model, "model_name", None)
    if isinstance(model_name, str) and model_name:
        return model_name
    if model is None:
        return config.model
    return type(resolved_model).__name__
