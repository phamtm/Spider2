from pathlib import Path

import pytest

from sol01.infra.config import (
    DEFAULT_BASE_URL,
    DEFAULT_DOTENV_PATH,
    DEFAULT_MODEL,
    RuntimeConfig,
    SchemaContextConfig,
)

SCHEMA_CONTEXT_ENV_VARS = [
    "SOL01_SCHEMA_FAMILY_SIMILARITY_THRESHOLD",
    "SOL01_SCHEMA_MAX_LINKED_DOC_CHARS",
    "SOL01_SCHEMA_MAX_PROMPT_CHARS",
    "SOL01_SCHEMA_TOP_K_SPARSE",
    "SOL01_SCHEMA_TOP_K_OBJECTS",
]

LLM_ENV_VARS = [
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_MODEL",
    "OPENROUTER_PROVIDER_ONLY",
    "OPENROUTER_ALLOW_FALLBACKS",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
]


def _clear_schema_context_env(monkeypatch):
    for name in SCHEMA_CONTEXT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _clear_llm_env(monkeypatch):
    for name in LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_default_dotenv_path_points_to_method_root():
    assert DEFAULT_DOTENV_PATH == Path(__file__).resolve().parents[1] / ".env"


def test_default_config_uses_deepseek_openrouter_policy(monkeypatch):
    _clear_llm_env(monkeypatch)

    config = RuntimeConfig.from_env()

    assert config.api_key is None
    assert config.base_url == DEFAULT_BASE_URL
    assert config.model == DEFAULT_MODEL
    assert config.provider_only == "deepseek"
    assert config.allow_fallbacks is False
    assert config.concurrency == 4
    assert config.provider_routing == {"provider": {"only": ["deepseek"], "allow_fallbacks": False}}


@pytest.mark.parametrize(
    "env_setup,expected_key,expected_url,expected_model",
    [
        pytest.param(
            {
                "LLM_API_KEY": "generic-key",
                "LLM_BASE_URL": "https://generic.example/v1",
                "LLM_MODEL": "deepseek/generic",
                "OPENROUTER_API_KEY": "openrouter-key",
                "OPENROUTER_BASE_URL": "https://openrouter.example/v1",
                "OPENROUTER_MODEL": "deepseek/custom",
                "OPENROUTER_PROVIDER_ONLY": "deepseek",
                "OPENROUTER_ALLOW_FALLBACKS": "false",
            },
            "openrouter-key",
            "https://openrouter.example/v1",
            "deepseek/custom",
            id="openrouter-wins-over-generic",
        ),
        pytest.param(
            {
                "LLM_API_KEY": "generic-key",
                "LLM_BASE_URL": "https://generic.example/v1",
                "LLM_MODEL": "deepseek/generic",
            },
            "generic-key",
            "https://generic.example/v1",
            "deepseek/generic",
            id="generic-when-no-openrouter",
        ),
    ],
)
def test_env_precedence(monkeypatch, env_setup, expected_key, expected_url, expected_model):
    _clear_llm_env(monkeypatch)
    for k, v in env_setup.items():
        monkeypatch.setenv(k, v)

    config = RuntimeConfig.from_env()

    assert config.api_key == expected_key
    assert config.base_url == expected_url
    assert config.model == expected_model


def test_required_api_key_fails_fast_for_live_runs(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API key"):
        RuntimeConfig.from_env(require_api_key=True)


def test_fallbacks_remain_disabled_by_default_even_with_truthy_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_ALLOW_FALLBACKS", "true")

    with pytest.raises(ValueError, match="fallback"):
        RuntimeConfig.from_env()


@pytest.mark.parametrize(
    "shell_env,expected_key,expected_url,expected_model",
    [
        pytest.param(
            {},
            "dotenv-key",
            "https://dotenv.example/v1",
            "deepseek/dotenv",
            id="dotenv-loads-when-shell-empty",
        ),
        pytest.param(
            {
                "OPENROUTER_API_KEY": "shell-key",
                "OPENROUTER_BASE_URL": "https://shell.example/v1",
                "OPENROUTER_MODEL": "deepseek/shell",
            },
            "shell-key",
            "https://shell.example/v1",
            "deepseek/shell",
            id="shell-wins-over-dotenv",
        ),
        pytest.param(
            {
                "OPENROUTER_API_KEY": "   ",
                "OPENROUTER_BASE_URL": "",
                "OPENROUTER_MODEL": " ",
            },
            "dotenv-key",
            "https://dotenv.example/v1",
            "deepseek/dotenv",
            id="blank-shell-falls-back-to-dotenv",
        ),
    ],
)
def test_dotenv_fallback(
    monkeypatch, tmp_path, shell_env, expected_key, expected_url, expected_model
):
    _clear_llm_env(monkeypatch)
    for k, v in shell_env.items():
        monkeypatch.setenv(k, v)

    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "OPENROUTER_API_KEY=dotenv-key",
                "OPENROUTER_BASE_URL=https://dotenv.example/v1",
                "OPENROUTER_MODEL=deepseek/dotenv",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = RuntimeConfig.from_env(dotenv_path=dotenv_path)

    assert config.api_key == expected_key
    assert config.base_url == expected_url
    assert config.model == expected_model


def test_concurrency_validation(monkeypatch):
    monkeypatch.setenv("SOL01_CONCURRENCY", "7")
    assert RuntimeConfig.from_env().concurrency == 7

    monkeypatch.setenv("SOL01_CONCURRENCY", "0")
    with pytest.raises(ValueError, match="positive integer"):
        RuntimeConfig.from_env()


def test_default_from_env_does_not_read_dotenv_implicitly(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("OPENROUTER_API_KEY=dotenv-key\n", encoding="utf-8")

    config = RuntimeConfig.from_env(dotenv_path=None)

    assert config.api_key is None
    with pytest.raises(ValueError, match="API key"):
        RuntimeConfig.from_env(require_api_key=True, dotenv_path=None)


def test_schema_context_config_defaults(monkeypatch):
    _clear_schema_context_env(monkeypatch)

    config = SchemaContextConfig.from_env()

    assert 0.0 <= config.family_similarity_threshold <= 1.0
    assert config.max_linked_doc_chars > 0
    assert config.max_schema_prompt_chars > config.max_linked_doc_chars


@pytest.mark.parametrize(
    "env_overrides,expected",
    [
        pytest.param(
            {
                "SOL01_SCHEMA_FAMILY_SIMILARITY_THRESHOLD": "0.7",
                "SOL01_SCHEMA_MAX_LINKED_DOC_CHARS": "4000",
                "SOL01_SCHEMA_MAX_PROMPT_CHARS": "18000",
                "SOL01_SCHEMA_TOP_K_SPARSE": "60",
                "SOL01_SCHEMA_TOP_K_OBJECTS": "25",
            },
            {
                "family_similarity_threshold": 0.7,
                "max_linked_doc_chars": 4000,
                "max_schema_prompt_chars": 18000,
                "top_k_sparse": 60,
                "top_k_objects": 25,
            },
            id="env-overrides",
        ),
    ],
)
def test_schema_context_env_overrides(monkeypatch, env_overrides, expected):
    _clear_schema_context_env(monkeypatch)
    for k, v in env_overrides.items():
        monkeypatch.setenv(k, v)

    config = SchemaContextConfig.from_env()

    for field, value in expected.items():
        assert getattr(config, field) == value


@pytest.mark.parametrize(
    "bad_env,match",
    [
        pytest.param(
            {"SOL01_SCHEMA_FAMILY_SIMILARITY_THRESHOLD": "1.1"},
            "between 0 and 1",
            id="threshold-above-one",
        ),
    ],
)
def test_schema_context_numeric_validation(monkeypatch, bad_env, match):
    _clear_schema_context_env(monkeypatch)
    for k, v in bad_env.items():
        monkeypatch.setenv(k, v)

    with pytest.raises(ValueError, match=match):
        SchemaContextConfig.from_env()
