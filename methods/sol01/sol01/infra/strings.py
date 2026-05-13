"""String utilities shared across modules."""

from __future__ import annotations


def column_looks_string_like(column_type: str | None) -> bool:
    """Return True when a schema column looks like a text field."""

    if column_type is None:
        return True
    lowered = column_type.lower()
    return any(token in lowered for token in ("char", "text", "string", "varchar", "variant"))


def question_preview(question: str, *, max_length: int = 120) -> str:
    """Shorten long questions so task logs stay readable."""

    normalized = " ".join(question.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1].rstrip() + "…"
