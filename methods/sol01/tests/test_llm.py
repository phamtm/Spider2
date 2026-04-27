"""Tests for prompt loading and structured LLM helper behavior."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

from sol01.config import RuntimeConfig
from sol01.llm import LLMClient, build_model, build_model_settings, prompt_sha256
from sol01.models import Intent

FIXTURE_PROMPTS_DIR = Path(__file__).parent / "fixtures" / "prompts"


class SampleOutput(BaseModel):
    """Simple structured output used to verify the fake model path."""

    value: str


def test_prompt_hash_matches_fixture_prompt_text():
    prompt_text = (FIXTURE_PROMPTS_DIR / "sample.md").read_text(encoding="utf-8")

    assert prompt_sha256(prompt_text) == (
        "64190c76c47b986b3bfd08fa2747d54b88d85c06d444e6ad1e2754baabd56e1c"
    )


def test_build_model_settings_includes_openrouter_provider_policy():
    config = RuntimeConfig(api_key="test-key")

    assert build_model_settings(config) == {
        "extra_body": {
            "provider": {
                "only": ["deepseek"],
                "allow_fallbacks": False,
            }
        }
    }


def test_llm_client_loads_prompt_and_hashes_it():
    client = LLMClient(RuntimeConfig(api_key="test-key"), prompts_dir=FIXTURE_PROMPTS_DIR)

    prompt = client.load_prompt("sample")

    assert prompt.name == "sample"
    assert prompt.text.startswith("# Sample Prompt")
    assert prompt.sha256 == "64190c76c47b986b3bfd08fa2747d54b88d85c06d444e6ad1e2754baabd56e1c"


def test_llm_client_returns_structured_output_from_test_model():
    client = LLMClient(RuntimeConfig(api_key="test-key"), prompts_dir=FIXTURE_PROMPTS_DIR)

    output = client.run_structured(
        "Rewrite this question.",
        prompt_name="sample",
        output_type=Intent,
        model=TestModel(
            custom_output_args={
                "summary": "Find top customers by spend.",
                "entities": ["customers"],
                "metrics": ["total spend"],
                "filters": [],
                "time_constraints": [],
                "output_expectation": "customer and total spend columns",
                "assumptions": ["Use all orders."],
            }
        ),
    )

    assert output.summary == "Find top customers by spend."
    assert output.metrics == ["total spend"]


def test_llm_client_returns_simple_structured_output_from_test_model():
    client = LLMClient(RuntimeConfig(api_key="test-key"), prompts_dir=FIXTURE_PROMPTS_DIR)

    output = client.run_structured(
        "Say hi.",
        prompt_name="sample",
        output_type=SampleOutput,
        model=TestModel(custom_output_args={"value": "hello"}),
    )

    assert output.model_dump() == {"value": "hello"}


def test_default_prompt_directory_contains_packaged_prompts():
    client = LLMClient(RuntimeConfig(api_key="test-key"))

    prompt = client.load_prompt("intent")

    assert prompt.text.startswith("# Intent Extraction")


def test_build_model_uses_openrouter_wrapper_for_string_override():
    config = RuntimeConfig(api_key="test-key")

    model = build_model(config, model_name="deepseek/custom-model")

    assert model.model_name == "deepseek/custom-model"
    assert model.settings == {
        "extra_body": {
            "provider": {
                "only": ["deepseek"],
                "allow_fallbacks": False,
            }
        }
    }
