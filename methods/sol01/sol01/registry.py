"""Persist a local registry of sol01 run and task outcomes."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from sol01.logging import get_logger
from sol01.tasks import REPO_ROOT

OUTPUTS_ROOT = REPO_ROOT / "methods" / "sol01" / "outputs"
REGISTRY_ROOT = OUTPUTS_ROOT / "registry"
RegistryStatus = Literal[
    "pass",
    "official_fail",
    "solver_failed",
    "missing_csv",
    "eval_failed",
]

logger = get_logger(__name__)


@dataclass(frozen=True)
class RegistryTaskRecord:
    """One per-instance record that feeds the local registry."""

    run_id: str
    instance_id: str
    db: str
    timestamp: str
    score: float | None
    run_path: str
    csv_path: str | None = None
    trace_path: str | None = None
    eval_path: str | None = None
    solver_status: Literal["success", "failed", "skipped"] | None = None
    eval_status: Literal["success", "failed"] | None = None
    eval_error: str | None = None
    extra_artifacts: dict[str, str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class RegistryPaths:
    """Filesystem paths for the registry."""

    root: Path
    runs_path: Path
    task_results_path: Path
    latest_path: Path


def ensure_registry_paths(*, outputs_root: Path = OUTPUTS_ROOT) -> RegistryPaths:
    """Create the registry directory layout if it does not already exist."""

    root = outputs_root / "registry"
    root.mkdir(parents=True, exist_ok=True)
    return RegistryPaths(
        root=root,
        runs_path=root / "runs.jsonl",
        task_results_path=root / "task_results.jsonl",
        latest_path=root / "latest.json",
    )


def record_registry_batch(
    records: Iterable[RegistryTaskRecord],
    *,
    outputs_root: Path = OUTPUTS_ROOT,
) -> dict[str, Any]:
    """Append one batch of task results and refresh the latest snapshot."""

    batch = list(records)
    if not batch:
        raise ValueError("records must not be empty")

    run_id = batch[0].run_id
    if any(record.run_id != run_id for record in batch):
        raise ValueError("all records in a batch must share one run_id")

    paths = ensure_registry_paths(outputs_root=outputs_root)
    rows = [_task_result_row(record) for record in batch]
    run_row = _run_row(rows, paths=paths)

    for row in rows:
        _append_jsonl(paths.task_results_path, row)
    _append_jsonl(paths.runs_path, run_row)

    latest = rebuild_latest(outputs_root=outputs_root)
    logger.info(
        "registry batch recorded",
        run_id=run_id,
        task_count=len(rows),
        registry_root=str(paths.root),
    )
    return latest


def rebuild_latest(*, outputs_root: Path = OUTPUTS_ROOT) -> dict[str, Any]:
    """Rebuild latest.json from the append-only task history."""

    paths = ensure_registry_paths(outputs_root=outputs_root)
    latest_by_instance: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(paths.task_results_path):
        latest_by_instance[row["instance_id"]] = row

    latest_payload = {
        "generated_at": _utc_now(),
        "task_count": len(latest_by_instance),
        "task_results": sorted(latest_by_instance.values(), key=lambda row: row["instance_id"]),
    }
    _atomic_write_json(paths.latest_path, latest_payload)
    return latest_payload


def load_latest(*, outputs_root: Path = OUTPUTS_ROOT) -> dict[str, Any]:
    """Load latest.json, rebuilding it from history when needed."""

    paths = ensure_registry_paths(outputs_root=outputs_root)
    if not paths.latest_path.exists() or _history_is_newer(paths):
        return rebuild_latest(outputs_root=outputs_root)

    try:
        return json.loads(paths.latest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return rebuild_latest(outputs_root=outputs_root)


def _task_result_row(record: RegistryTaskRecord) -> dict[str, Any]:
    """Convert one input record into a stable registry row."""

    row = asdict(record)
    row["status"] = _status_for(record)
    row["run_path"] = str(record.run_path)
    row["csv_path"] = _string_or_none(record.csv_path)
    row["trace_path"] = _string_or_none(record.trace_path)
    row["eval_path"] = _string_or_none(record.eval_path)
    row["extra_artifacts"] = {
        key: _string_or_none(value) for key, value in sorted(record.extra_artifacts.items())
    }
    row["score"] = _score_value(record.score)
    return row


def _run_row(task_rows: list[dict[str, Any]], *, paths: RegistryPaths) -> dict[str, Any]:
    """Summarize one batch for the append-only runs.jsonl history."""

    status_counts = Counter(row["status"] for row in task_rows)
    first_row = task_rows[0]
    return {
        "run_id": first_row["run_id"],
        "timestamp": _utc_now(),
        "task_count": len(task_rows),
        "status_counts": {status: status_counts.get(status, 0) for status in _status_order()},
        "run_path": first_row["run_path"],
        "task_results_path": str(paths.task_results_path),
        "latest_path": str(paths.latest_path),
    }


def _status_for(record: RegistryTaskRecord) -> RegistryStatus:
    """Classify one registry record using the requested status vocabulary."""

    if record.eval_status == "failed" or record.eval_error:
        return "eval_failed"
    if record.solver_status == "failed":
        return "solver_failed"
    if record.csv_path is None:
        return "missing_csv"
    if _score_value(record.score) == 1:
        return "pass"
    return "official_fail"


def _score_value(score: float | None) -> float | None:
    """Normalize numeric scores while preserving missing values."""

    if score is None:
        return None
    return float(score)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one JSON object as a single line."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True))
        handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file when it exists, skipping blank lines."""

    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("skipping invalid registry jsonl row", path=str(path))
    return rows


def _history_is_newer(paths: RegistryPaths) -> bool:
    """Return True when append-only history may have outpaced latest.json."""

    if not paths.task_results_path.exists():
        return False
    if not paths.latest_path.exists():
        return True
    return paths.task_results_path.stat().st_mtime_ns > paths.latest_path.stat().st_mtime_ns


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically so interrupted updates do not corrupt latest.json."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _string_or_none(value: str | None) -> str | None:
    """Keep optional paths as strings without changing nulls."""

    return str(value) if value is not None else None


def _utc_now() -> str:
    """Return a stable UTC timestamp string."""

    return datetime.now(UTC).isoformat()


def _status_order() -> tuple[RegistryStatus, ...]:
    """Keep status counts in a predictable order."""

    return ("pass", "official_fail", "solver_failed", "missing_csv", "eval_failed")
