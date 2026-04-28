"""Tests for optional Logfire instrumentation."""

from __future__ import annotations

from sol01 import observability


def test_configure_logfire_enables_pydantic_ai_instrumentation(monkeypatch):
    calls: list[tuple[str, object]] = []

    def fake_configure(**kwargs):
        calls.append(("configure", kwargs))

    def fake_instrument_pydantic_ai(**kwargs):
        calls.append(("instrument_pydantic_ai", kwargs))

    monkeypatch.setattr(observability.logfire, "configure", fake_configure)
    monkeypatch.setattr(
        observability.logfire,
        "instrument_pydantic_ai",
        fake_instrument_pydantic_ai,
    )
    monkeypatch.setattr(observability, "_LOGFIRE_CONFIGURED", False)

    assert observability.configure_logfire() is True
    assert calls[0][0] == "configure"
    assert calls[0][1]["send_to_logfire"] == "if-token-present"
    assert calls[0][1]["service_name"] == "sol01"
    assert calls[1] == ("instrument_pydantic_ai", {"include_content": True, "version": 3})
