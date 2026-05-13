"""Internal helpers shared within the schema package."""

from __future__ import annotations

from sol01.models import SchemaObject


def _metadata_text(obj: SchemaObject, key: str) -> str:
    value = obj.metadata.get(key)
    return "" if value is None else str(value).strip()


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
