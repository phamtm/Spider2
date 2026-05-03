"""Tests for prompt loading and structured LLM helper behavior."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import ModelResponse, PromptedOutput, ThinkingPart
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles.openai import OpenAIModelProfile

from sol01.infra.config import RuntimeConfig
from sol01.llm.client import LLMClient, build_model, build_model_settings, prompt_sha256
from sol01.llm.llm_logging import LLMCallLogger
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
            custom_output_text="""
            {
              "summary": "Find top customers by spend.",
              "entities": ["customers"],
              "metrics": ["total spend"],
              "filters": [],
              "time_constraints": [],
              "output_expectation": "customer and total spend columns",
              "assumptions": ["Use all orders."]
            }
            """
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
        model=TestModel(custom_output_text='{"value":"hello"}'),
    )

    assert output.model_dump() == {"value": "hello"}


def test_default_prompt_directory_contains_packaged_prompts():
    client = LLMClient(RuntimeConfig(api_key="test-key"))

    prompt = client.load_prompt("intent")

    assert prompt.text.startswith("# Intent Extraction")


def test_default_sql_prompts_keep_grouped_identifiers():
    client = LLMClient(RuntimeConfig(api_key="test-key"))

    intent_prompt = client.load_prompt("intent")
    generation_prompt = client.load_prompt("sql_generation")
    repair_prompt = client.load_prompt("sql_repair")
    critic_prompt = client.load_prompt("result_critic")
    comparison_prompt = client.load_prompt("result_comparison")

    assert "grouped superlatives" in intent_prompt.text
    assert "highest number in any month" in intent_prompt.text
    assert "Do not bind a metric to a specific column" in intent_prompt.text
    assert "stable identifier and a display label" in generation_prompt.text
    assert "winning group key plus the metric" in generation_prompt.text
    assert "requested answer grain" in generation_prompt.text
    assert "column-name semantics" in generation_prompt.text
    assert "stable identifier and a display label" in repair_prompt.text
    assert "winning group key plus the metric" in repair_prompt.text
    assert "native metric column" in repair_prompt.text
    assert "column-name semantics" in repair_prompt.text
    assert "metric-source mismatches" in critic_prompt.text
    assert "line-item formulas" in critic_prompt.text
    assert "preserves a stable grouping identifier" in comparison_prompt.text
    assert "native metric column" in comparison_prompt.text
    assert "column-name semantics" in comparison_prompt.text


def test_build_model_uses_openrouter_wrapper_for_string_override():
    config = RuntimeConfig(api_key="test-key")

    model = build_model(config, model_name="deepseek/custom-model")

    assert model.model_name == "deepseek/custom-model"
    profile = OpenAIModelProfile.from_profile(model.profile)
    assert profile.supports_json_object_output is True
    assert profile.openai_chat_send_back_thinking_parts == "tags"
    assert model.settings == {
        "extra_body": {
            "provider": {
                "only": ["deepseek"],
                "allow_fallbacks": False,
            }
        }
    }


def test_build_model_maps_reasoning_retries_to_text_content():
    config = RuntimeConfig(api_key="test-key")
    model = build_model(config)

    context = model._MapModelResponseContext(model)
    message = context.map_assistant_message(
        ModelResponse(
            parts=[
                ThinkingPart(
                    id="reasoning",
                    content='{"value":"hello"}',
                    provider_name=model.system,
                )
            ]
        )
    )

    assert message["content"] == '<think>\n{"value":"hello"}\n</think>'
    assert "reasoning" not in message


def test_llm_client_uses_prompted_output_for_structured_calls(monkeypatch):
    client = LLMClient(RuntimeConfig(api_key="test-key"), prompts_dir=FIXTURE_PROMPTS_DIR)
    captured: dict[str, object] = {}

    class FakeResult:
        output = SampleOutput(value="hello")

    class FakeAgent:
        def __init__(self, *, model, system_prompt, output_type):
            captured["output_type"] = output_type

        def run_sync(self, user_prompt):
            return FakeResult()

    monkeypatch.setattr("sol01.llm.client.Agent", FakeAgent)

    output = client.run_structured(
        "Say hi.",
        prompt_name="sample",
        output_type=SampleOutput,
        model=TestModel(custom_output_args={"value": "hello"}),
    )

    assert output.model_dump() == {"value": "hello"}
    assert isinstance(captured["output_type"], PromptedOutput)


def test_llm_client_logs_successful_structured_call(monkeypatch, tmp_path):
    log_path = tmp_path / "calls.jsonl"
    client = LLMClient(
        RuntimeConfig(api_key="test-key"),
        prompts_dir=FIXTURE_PROMPTS_DIR,
        call_logger=LLMCallLogger(log_path),
    )

    class FakeResult:
        output = SampleOutput(value="hello")

    class FakeAgent:
        def __init__(self, *, model, system_prompt, output_type):
            pass

        def run_sync(self, user_prompt):
            return FakeResult()

    monkeypatch.setattr("sol01.llm.client.Agent", FakeAgent)

    output = client.run_structured(
        "Say hi.",
        prompt_name="sample",
        output_type=SampleOutput,
    )

    assert output.model_dump() == {"value": "hello"}
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["schema_version"] == 1
    assert record["call_id"] == "0001-sample"
    assert record["status"] == "success"
    assert record["prompt_name"] == "sample"
    assert record["prompt_sha256"] == prompt_sha256(
        (FIXTURE_PROMPTS_DIR / "sample.md").read_text(encoding="utf-8")
    )
    assert record["output_type"] == "SampleOutput"
    assert record["model"] == "deepseek/deepseek-v4-pro"
    assert record["base_url"] == "https://openrouter.ai/api/v1"
    assert record["request"]["system_prompt"].startswith("# Sample Prompt")
    assert record["request"]["user_prompt"] == "Say hi."
    assert record["request"]["output_schema"] == "SampleOutput"
    assert record["response"]["validated_output"] == {"value": "hello"}
    assert record["attempts"][0]["status"] == "success"
    assert record["error"] is None
    assert "test-key" not in log_path.read_text(encoding="utf-8")


def test_llm_client_retries_transient_model_http_errors(monkeypatch):
    client = LLMClient(RuntimeConfig(api_key="test-key"), prompts_dir=FIXTURE_PROMPTS_DIR)
    calls = {"count": 0}

    class FakeResult:
        output = SampleOutput(value="hello")

    class FakeAgent:
        def __init__(self, *, model, system_prompt, output_type):
            pass

        def run_sync(self, user_prompt):
            calls["count"] += 1
            if calls["count"] < 3:
                raise ModelHTTPError(
                    status_code=429,
                    model_name="deepseek/deepseek-v4-pro",
                    body={"message": "rate limited"},
                )
            return FakeResult()

    monkeypatch.setattr("sol01.llm.client.Agent", FakeAgent)
    monkeypatch.setattr("sol01.llm.client.time.sleep", lambda seconds: None)

    output = client.run_structured(
        "Say hi.",
        prompt_name="sample",
        output_type=SampleOutput,
        model=TestModel(custom_output_text='{"value":"hello"}'),
    )

    assert output.model_dump() == {"value": "hello"}
    assert calls["count"] == 3


def test_llm_client_logs_retry_attempts(monkeypatch, tmp_path):
    log_path = tmp_path / "calls.jsonl"
    client = LLMClient(
        RuntimeConfig(api_key="test-key"),
        prompts_dir=FIXTURE_PROMPTS_DIR,
        call_logger=LLMCallLogger(log_path),
    )
    calls = {"count": 0}

    class FakeResult:
        output = SampleOutput(value="hello")

    class FakeAgent:
        def __init__(self, *, model, system_prompt, output_type):
            pass

        def run_sync(self, user_prompt):
            calls["count"] += 1
            if calls["count"] < 3:
                raise ModelHTTPError(
                    status_code=429,
                    model_name="deepseek/deepseek-v4-pro",
                    body={"message": "rate limited"},
                )
            return FakeResult()

    monkeypatch.setattr("sol01.llm.client.Agent", FakeAgent)
    monkeypatch.setattr("sol01.llm.client.time.sleep", lambda seconds: None)

    output = client.run_structured(
        "Say hi.",
        prompt_name="sample",
        output_type=SampleOutput,
    )

    assert output.model_dump() == {"value": "hello"}
    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["status"] == "success"
    assert [attempt["status"] for attempt in record["attempts"]] == [
        "error",
        "error",
        "success",
    ]
    assert record["attempts"][0]["error"]["status_code"] == 429
    assert record["error"] is None


def test_llm_client_does_not_retry_non_transient_model_http_errors(monkeypatch):
    client = LLMClient(RuntimeConfig(api_key="test-key"), prompts_dir=FIXTURE_PROMPTS_DIR)
    calls = {"count": 0}

    class FakeAgent:
        def __init__(self, *, model, system_prompt, output_type):
            pass

        def run_sync(self, user_prompt):
            calls["count"] += 1
            raise ModelHTTPError(
                status_code=400,
                model_name="deepseek/deepseek-v4-pro",
                body={"message": "bad request"},
            )

    monkeypatch.setattr("sol01.llm.client.Agent", FakeAgent)
    monkeypatch.setattr("sol01.llm.client.time.sleep", lambda seconds: None)

    try:
        client.run_structured(
            "Say hi.",
            prompt_name="sample",
            output_type=SampleOutput,
            model=TestModel(custom_output_text='{"value":"hello"}'),
        )
    except ModelHTTPError as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("Expected ModelHTTPError")

    assert calls["count"] == 1


def test_llm_client_logs_non_transient_model_http_error(monkeypatch, tmp_path):
    log_path = tmp_path / "calls.jsonl"
    client = LLMClient(
        RuntimeConfig(api_key="test-key"),
        prompts_dir=FIXTURE_PROMPTS_DIR,
        call_logger=LLMCallLogger(log_path),
    )

    class FakeAgent:
        def __init__(self, *, model, system_prompt, output_type):
            pass

        def run_sync(self, user_prompt):
            raise ModelHTTPError(
                status_code=400,
                model_name="deepseek/deepseek-v4-pro",
                body={"message": "bad request"},
            )

    monkeypatch.setattr("sol01.llm.client.Agent", FakeAgent)
    monkeypatch.setattr("sol01.llm.client.time.sleep", lambda seconds: None)

    try:
        client.run_structured(
            "Say hi.",
            prompt_name="sample",
            output_type=SampleOutput,
        )
    except ModelHTTPError as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("Expected ModelHTTPError")

    log_text = log_path.read_text(encoding="utf-8")
    record = json.loads(log_text)
    assert record["status"] == "error"
    assert record["response"] is None
    assert record["error"]["type"] == "ModelHTTPError"
    assert record["error"]["status_code"] == 400
    assert record["error"]["body"] == {"message": "bad request"}
    assert record["attempts"][0]["error"]["status_code"] == 400
    assert "test-key" not in log_text
