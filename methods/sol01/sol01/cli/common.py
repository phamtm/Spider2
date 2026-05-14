"""Small shared helpers for CLI modules."""

from __future__ import annotations

from datetime import UTC, datetime


def default_run_id(prefix: str) -> str:
    """Generate a timestamp-based run ID."""

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{prefix}-{stamp}"


def format_rate(value: float) -> str:
    """Render a metric as both percentage and decimal for CLI output."""

    return f"{value:.1%} ({value:.3f})"


def runtime_config_summary(config: object) -> dict[str, object]:
    """Extract log-safe runtime config fields without depending on the concrete type."""

    return {
        "model": getattr(config, "model", None),
        "base_url": getattr(config, "base_url", None),
        "concurrency": getattr(config, "concurrency", None),
    }


def slug(value: str) -> str:
    """Turn a short filter value into a filesystem-friendly label."""

    return "".join(char if char.isalnum() else "-" for char in value).strip("-") or "value"
