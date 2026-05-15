from __future__ import annotations

from typing import Any

import pandas as pd


def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, (bool, int)) else False


def missing_to_na(value: Any) -> Any:
    if is_missing_value(value):
        return pd.NA
    return value


def normalize_tag_values(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(tag) for tag in value if str(tag)]
    if is_missing_value(value):
        return []
    return [str(value)]


def truncate_text(value: Any, limit: int) -> str:
    if is_missing_value(value):
        return ""
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "\u2026"


def tier_display(value: Any) -> str:
    if is_missing_value(value):
        return "Uncategorized"
    try:
        return f"Tier {int(float(value))}"
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or "Uncategorized"
