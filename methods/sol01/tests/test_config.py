import pytest

from sol01.config import DEFAULT_BASE_URL, DEFAULT_MODEL, RuntimeConfig


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
    assert config.provider_routing == {
        "provider": {"only": ["deepseek"], "allow_fallbacks": False}
    }


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
    assert config.provider_routing == {
        "provider": {"only": ["deepseek"], "allow_fallbacks": False}
    }


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
