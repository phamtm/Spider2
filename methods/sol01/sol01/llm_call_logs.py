"""Read per-instance LLM call logs from JSONL artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LLMCallLogError:
    """A single row-level load error from one LLM call log file."""

    path: Path
    line_number: int
    message: str
    raw_line: str | None = None


@dataclass(frozen=True)
class LLMCallLogRecord:
    """One parsed LLM call row from the JSONL source of truth."""

    line_number: int
    sequence: int | None
    call_id: str | None
    prompt_name: str | None
    status: str | None
    started_at: datetime | None
    completed_at: datetime | None
    duration_ms: int | None
    model: str | None
    request: Any
    response: Any
    attempts: Any
    error: Any
    prompt_sha256: str | None = None
    output_type: str | None = None
    base_url: str | None = None
    provider_routing: Any = None
    schema_version: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMCallLog:
    """Parsed records plus any row-level load errors for one log file."""

    path: Path
    records: list[LLMCallLogRecord]
    errors: list[LLMCallLogError]


def load_llm_call_log(path: Path) -> LLMCallLog:
    """Load one per-instance LLM call log file."""

    if not path.exists():
        return LLMCallLog(path=path, records=[], errors=[])

    records: list[LLMCallLogRecord] = []
    errors: list[LLMCallLogError] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            raw_line = line.rstrip("\n")
            text = raw_line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(
                    LLMCallLogError(
                        path=path,
                        line_number=line_number,
                        message=str(exc),
                        raw_line=raw_line,
                    )
                )
                continue
            if not isinstance(row, dict):
                errors.append(
                    LLMCallLogError(
                        path=path,
                        line_number=line_number,
                        message="expected a JSON object row",
                        raw_line=raw_line,
                    )
                )
                continue
            records.append(_parse_record(line_number=line_number, row=row))

    return LLMCallLog(path=path, records=records, errors=errors)


def read_llm_call_log(path: Path) -> LLMCallLog:
    """Compatibility alias for loading one per-instance LLM call log file."""

    return load_llm_call_log(path)


def _parse_record(*, line_number: int, row: dict[str, Any]) -> LLMCallLogRecord:
    """Normalize one raw JSON row into a stable record shape."""

    return LLMCallLogRecord(
        line_number=line_number,
        sequence=_to_int(row.get("sequence")),
        call_id=_to_str(row.get("call_id")),
        prompt_name=_to_str(row.get("prompt_name")),
        status=_to_str(row.get("status")),
        started_at=_parse_datetime(row.get("started_at")),
        completed_at=_parse_datetime(row.get("completed_at")),
        duration_ms=_to_int(row.get("duration_ms")),
        model=_to_str(row.get("model")),
        request=row.get("request"),
        response=row.get("response"),
        attempts=row.get("attempts"),
        error=row.get("error"),
        prompt_sha256=_to_str(row.get("prompt_sha256")),
        output_type=_to_str(row.get("output_type")),
        base_url=_to_str(row.get("base_url")),
        provider_routing=row.get("provider_routing"),
        schema_version=_to_int(row.get("schema_version")),
        raw=dict(row),
    )


def _parse_datetime(value: Any) -> datetime | None:
    """Parse an ISO-ish timestamp into an aware UTC datetime when possible."""

    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value), tz=UTC)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _to_int(value: Any) -> int | None:
    """Coerce one value to an integer when possible."""

    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_str(value: Any) -> str | None:
    """Coerce one value to a stripped string when possible."""

    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None
