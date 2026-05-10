"""Tests for duration formatting helpers."""

from __future__ import annotations

from sol01.infra.time_utils import format_duration


def test_format_duration_uses_short_human_readable_units():
    assert format_duration(0.4) == "0.4s"
    assert format_duration(12.0) == "12s"
    assert format_duration(65.2) == "1m 05s"
    assert format_duration(3661.0) == "1h 01m 01s"
