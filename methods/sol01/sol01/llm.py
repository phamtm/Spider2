"""Load prompt files and run structured LLM calls through OpenRouter."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from sol01.config import RuntimeConfig

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass(frozen=True)
class PromptSpec:
    """Prompt text plus its stable SHA-256 hash."""

    name: str
    text: str
    sha256: str


class LLMClient:
    """Thin wrapper around prompt loading and structured Pydantic AI calls."""

    def __init__(self, config: RuntimeConfig, *, prompts_dir: Path = PROMPTS_DIR) -> None:
        self.config = config
        self.prompts_dir = prompts_dir

    def load_prompt(self, prompt_name: str) -> PromptSpec:
        """Read one markdown prompt and compute its reproducible content hash."""

        prompt_path = _prompt_path(prompt_name, prompts_dir=self.prompts_dir)
        text = prompt_path.read_text(encoding="utf-8")
        return PromptSpec(
            name=prompt_name,
            text=text,
            sha256=prompt_sha256(text),
        )

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
        agent = Agent(
            model=_resolve_model(model, config=self.config),
            system_prompt=prompt.text,
            output_type=output_type,
        )
        result = agent.run_sync(user_prompt)
        output = result.output
        if not isinstance(output, output_type):
            raise TypeError(f"Expected {output_type.__name__}, got {type(output).__name__}")
        return output

    def run_structured_with_prompt(
        self,
        user_prompt: str,
        *,
        prompt: PromptSpec,
        output_type: type[OutputT],
        model: Model | str | None = None,
    ) -> OutputT:
        """Run one structured prompt using an already loaded prompt spec."""

        agent = Agent(
            model=_resolve_model(model, config=self.config),
            system_prompt=prompt.text,
            output_type=output_type,
        )
        result = agent.run_sync(user_prompt)
        output = result.output
        if not isinstance(output, output_type):
            raise TypeError(f"Expected {output_type.__name__}, got {type(output).__name__}")
        return output


def prompt_sha256(text: str) -> str:
    """Hash prompt text so later traces can record exact prompt versions."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_model_settings(config: RuntimeConfig) -> dict[str, Any]:
    """Build the OpenRouter request body additions required by this solver."""

    return {"extra_body": config.provider_routing}


def build_model(config: RuntimeConfig, *, model_name: str | None = None) -> OpenAIChatModel:
    """Create the OpenAI-compatible chat model that points at OpenRouter."""

    provider = OpenAIProvider(
        base_url=config.base_url,
        api_key=config.api_key,
    )
    return OpenAIChatModel(
        model_name or config.model,
        provider=provider,
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
