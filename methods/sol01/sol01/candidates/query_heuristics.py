"""Small SQL heuristics shared across candidate verification helpers."""

from __future__ import annotations

from numbers import Number
from typing import Any


def coerce_number(value: Any) -> float | None:
    """Convert one value to a number when it looks numeric."""

    if isinstance(value, bool):
        return None
    if isinstance(value, Number):
        return float(value)  # type: ignore[arg-type]
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def looks_aggregate_query(sql: str) -> bool:
    """Heuristically detect aggregate queries that deserve extra scrutiny."""

    normalized = " ".join(sql.lower().split())
    return any(
        keyword in normalized
        for keyword in (" count(", " sum(", " avg(", " min(", " max(", " group by ", " having ")
    )
