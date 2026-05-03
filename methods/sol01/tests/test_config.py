from pathlib import Path

import pytest

from sol01.infra.config import DEFAULT_BASE_URL, DEFAULT_MODEL, RuntimeConfig


def test_default_config_uses_deepseek_openrouter_policy(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_PROVIDER_ONLY", raising=False)
    monkeypatch.delenv("OPENROUTER_ALLOW_FALLBACKS", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    config = RuntimeConfig.from_env()

    assert config.api_key is None
    assert config.base_url == DEFAULT_BASE_URL
    assert config.model == DEFAULT_MODEL
    assert config.provider_only == "deepseek"
    assert config.allow_fallbacks is False
    assert config.concurrency == 4
    assert config.provider_routing == {"provider": {"only": ["deepseek"], "allow_fallbacks": False}}


def test_openrouter_env_overrides_generic_aliases(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "generic-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://generic.example/v1")
    monkeypatch.setenv("LLM_MODEL", "generic/model")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.example/v1")
    monkeypatch.setenv("OPENROUTER_MODEL", "deepseek/custom")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ONLY", "deepseek")
    monkeypatch.setenv("OPENROUTER_ALLOW_FALLBACKS", "false")

    config = RuntimeConfig.from_env()

    assert config.api_key == "openrouter-key"
    assert config.base_url == "https://openrouter.example/v1"
    assert config.model == "deepseek/custom"
    assert config.provider_routing == {"provider": {"only": ["deepseek"], "allow_fallbacks": False}}


def test_generic_llm_aliases_work_when_openrouter_env_is_absent(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "generic-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://generic.example/v1")
    monkeypatch.setenv("LLM_MODEL", "deepseek/generic")

    config = RuntimeConfig.from_env()

    assert config.api_key == "generic-key"
    assert config.base_url == "https://generic.example/v1"
    assert config.model == "deepseek/generic"


def test_required_api_key_fails_fast_for_live_runs(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API key"):
        RuntimeConfig.from_env(require_api_key=True)


def test_fallbacks_remain_disabled_by_default_even_with_truthy_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_ALLOW_FALLBACKS", "true")

    with pytest.raises(ValueError, match="fallback"):
        RuntimeConfig.from_env()


def test_dotenv_file_loads_openrouter_settings_when_shell_is_empty(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
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

    assert config.api_key == "dotenv-key"
    assert config.base_url == "https://dotenv.example/v1"
    assert config.model == "deepseek/dotenv"


def test_shell_env_still_wins_over_dotenv_file(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "shell-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://shell.example/v1")
    monkeypatch.setenv("OPENROUTER_MODEL", "deepseek/shell")
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

    assert config.api_key == "shell-key"
    assert config.base_url == "https://shell.example/v1"
    assert config.model == "deepseek/shell"


def test_blank_shell_env_allows_dotenv_fallback(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "")
    monkeypatch.setenv("OPENROUTER_MODEL", " ")
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

    assert config.api_key == "dotenv-key"
    assert config.base_url == "https://dotenv.example/v1"
    assert config.model == "deepseek/dotenv"


def test_sol01_concurrency_env_overrides_default(monkeypatch):
    monkeypatch.setenv("SOL01_CONCURRENCY", "7")

    config = RuntimeConfig.from_env()

    assert config.concurrency == 7


def test_sol01_concurrency_env_must_be_positive(monkeypatch):
    monkeypatch.setenv("SOL01_CONCURRENCY", "0")

    with pytest.raises(ValueError, match="positive integer"):
        RuntimeConfig.from_env()


def test_default_from_env_does_not_read_dotenv_implicitly(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("OPENROUTER_API_KEY=dotenv-key\n", encoding="utf-8")

    config = RuntimeConfig.from_env(dotenv_path=None)

    assert config.api_key is None
    with pytest.raises(ValueError, match="API key"):
        RuntimeConfig.from_env(require_api_key=True, dotenv_path=None)
