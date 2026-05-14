from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sol01.analysis.trace_diagnostics import summarize_trace_diagnostics
from sol01.infra.logging import get_logger
from sol01.progress_ui.constants import DEFAULT_DATASET, DEFAULT_SOURCE, ROOT
from sol01.progress_ui.models import Record

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    args, _ = parser.parse_known_args()
    return args


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        logger.warning("skipping corrupted json result file", path=str(path), error=str(exc))
        return {}
    except FileNotFoundError:
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "skipping corrupted jsonl row",
                            path=str(path),
                            line_number=line_number,
                            error=str(exc),
                        )
                        continue
    except FileNotFoundError:
        pass
    return rows


def parse_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=UTC)

    text = str(value).strip()
    formats = (
        "%Y%m%dT%H%M%S.%fZ",
        "%Y%m%dT%H%M%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def as_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_instance_id(item: dict[str, Any], fallback: str | None = None) -> str | None:
    for key in ("instance_id", "id", "question_id", "task_id", "output"):
        value = item.get(key)
        if value:
            return str(value).removesuffix(".csv").removesuffix(".sql")
    return fallback


_REGISTRY_STATUS_MAP: dict[str, tuple[str, float | None]] = {
    "pass": ("correct", 1.0),
    "official_fail": ("incorrect", 0.0),
    "eval_failed": ("incorrect", 0.0),
    "solver_failed": ("unanswered", None),
    "missing_csv": ("unanswered", None),
}


def classify(item: dict[str, Any]) -> tuple[str, float | None]:
    status = str(item.get("status") or "")
    if status in _REGISTRY_STATUS_MAP:
        display_status, default_score = _REGISTRY_STATUS_MAP[status]
        score = as_float(item.get("score"))
        return display_status, score if score is not None else default_score

    if item.get("csv_path") or item.get("sql_path") or item.get("csv_present"):
        return "answered", None
    return "unanswered", None


def normalize_item(
    item: dict[str, Any], source_path: Path, fallback_id: str | None = None
) -> Record | None:
    instance_id = find_instance_id(item, fallback=fallback_id)
    if not instance_id:
        return None

    status, score = classify(item)
    timestamp = parse_timestamp(item.get("timestamp") or item.get("generated_at"))
    note = item.get("failure_reason") or item.get("eval_error") or item.get("solver_status")
    diagnostics = summarize_trace_diagnostics(item)
    return Record(
        instance_id=instance_id,
        status=status,
        score=score,
        timestamp=timestamp,
        run_id=item.get("run_id"),
        db=item.get("db") or item.get("db_id"),
        note=str(note) if note else None,
        source_path=str(source_path),
        diagnostics=diagnostics,
    )
