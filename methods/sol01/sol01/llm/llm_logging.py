"""Append local JSONL debug records for LLM calls."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel

SCHEMA_VERSION = 1


class LLMCallLogger:
    """Write one stable JSON object per logical LLM call."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._sequence = 0

    def next_sequence(self) -> int:
        """Return the next per-file call sequence."""

        self._sequence += 1
        return self._sequence

    def write(self, record: dict[str, Any]) -> None:
        """Append one JSONL record to disk."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")


def utc_now() -> datetime:
    """Return an aware UTC timestamp for log records."""

    return datetime.now(UTC)


def isoformat_utc(value: datetime) -> str:
    """Serialize timestamps in a consistent UTC format."""

    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def duration_ms(started_at: float) -> int:
    """Return elapsed wall time in milliseconds."""

    return round((perf_counter() - started_at) * 1000)


def serialize_output(output: Any) -> Any:
    """Convert a validated model output into JSON-compatible data."""

    if isinstance(output, BaseModel):
        return output.model_dump(mode="json")
    return _json_safe(output)


def serialize_error(exc: BaseException) -> dict[str, Any]:
    """Convert an exception into a safe JSON error object."""

    error: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        error["status_code"] = status_code
    body = getattr(exc, "body", None)
    if body is not None:
        error["body"] = _json_safe(body)
    return error


def _json_safe(value: Any) -> Any:
    """Recursively coerce values into JSON-compatible shapes."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value
