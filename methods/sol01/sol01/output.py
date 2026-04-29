"""Create output folders, write run artifacts, and decide resume behavior."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sol01.tasks import REPO_ROOT

OUTPUTS_ROOT = REPO_ROOT / "methods" / "sol01" / "outputs"


@dataclass(frozen=True)
class RunPaths:
    """All filesystem locations used for one benchmark run."""

    root: Path
    manifest_path: Path
    sql_dir: Path
    csv_dir: Path
    traces_dir: Path
    eval_dir: Path
    analysis_dir: Path


@dataclass(frozen=True)
class AskPaths:
    """Filesystem locations used for one ad hoc ask run."""

    root: Path
    sql_path: Path
    csv_path: Path
    trace_path: Path


def ensure_run_paths(run_id: str, *, outputs_root: Path = OUTPUTS_ROOT) -> RunPaths:
    """Create the standard folder layout for a named benchmark run."""

    root = outputs_root / run_id
    sql_dir = root / "sql"
    csv_dir = root / "csv"
    traces_dir = root / "traces"
    eval_dir = root / "eval"
    analysis_dir = root / "analysis"

    for path in (sql_dir, csv_dir, traces_dir, eval_dir, analysis_dir):
        path.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        root=root,
        manifest_path=root / "manifest.json",
        sql_dir=sql_dir,
        csv_dir=csv_dir,
        traces_dir=traces_dir,
        eval_dir=eval_dir,
        analysis_dir=analysis_dir,
    )


def ensure_ask_paths(
    *,
    timestamp: str | None = None,
    outputs_root: Path = OUTPUTS_ROOT,
) -> AskPaths:
    """Create the default folder layout for one ad hoc ask command."""

    root = _unique_ask_root(outputs_root=outputs_root, timestamp=timestamp)
    root.mkdir(parents=True, exist_ok=False)
    return AskPaths(
        root=root,
        sql_path=root / "sql.sql",
        csv_path=root / "result.csv",
        trace_path=root / "trace.json",
    )


def write_manifest(
    run_paths: RunPaths,
    *,
    manifest: dict[str, Any],
) -> Path:
    """Write the run manifest as stable JSON."""

    payload = dict(manifest)
    payload.setdefault("run_id", run_paths.root.name)
    payload.setdefault("generated_at", datetime.now(UTC).isoformat())
    return _write_json(run_paths.manifest_path, payload)


def write_sql(run_paths: RunPaths, *, instance_id: str, sql: str) -> Path:
    """Write the final SQL text for one task."""

    path = run_paths.sql_dir / f"{instance_id}.sql"
    path.write_text(sql.rstrip() + "\n", encoding="utf-8")
    return path


def write_trace(run_paths: RunPaths, *, instance_id: str, trace: dict[str, Any]) -> Path:
    """Write one per-task trace file as stable JSON."""

    return _write_json(run_paths.traces_dir / f"{instance_id}.json", trace)


def csv_path_for(run_paths: RunPaths, *, instance_id: str) -> Path:
    """Return the expected CSV path for one task result."""

    return run_paths.csv_dir / f"{instance_id}.csv"


def trace_path_for(run_paths: RunPaths, *, instance_id: str) -> Path:
    """Return the expected trace path for one task."""

    return run_paths.traces_dir / f"{instance_id}.json"


def should_skip_task(
    run_paths: RunPaths,
    *,
    instance_id: str,
    skip_failed: bool = False,
) -> bool:
    """Decide whether resume mode should skip or rerun one task."""

    trace_path = trace_path_for(run_paths, instance_id=instance_id)
    if not trace_path.exists():
        return False

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    status = trace.get("status")
    csv_path = csv_path_for(run_paths, instance_id=instance_id)
    actual_retrieval_mode = _trace_retrieval_mode(trace)

    # Rerun traces produced by any removed fixed retrieval path.
    if actual_retrieval_mode != "llm_only":
        return False

    if status == "success":
        return csv_path.exists()
    if status == "skipped":
        return True
    if status == "failed":
        return skip_failed
    return False


def _trace_retrieval_mode(trace: dict[str, Any]) -> str:
    """Read the schema-selection marker and default old traces to llm_only."""

    mode = trace.get("retrieval_mode")
    if isinstance(mode, str):
        return mode

    schema = trace.get("schema_selection") or {}
    schema_mode = schema.get("retrieval_mode")
    if isinstance(schema_mode, str):
        return schema_mode
    return "llm_only"


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write one JSON file with a stable format that is easy to diff."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _unique_ask_root(*, outputs_root: Path, timestamp: str | None) -> Path:
    """Pick a unique ask output folder even when runs start very close together."""

    ask_root = outputs_root / "ask"
    ask_root.mkdir(parents=True, exist_ok=True)

    if timestamp is not None:
        return ask_root / timestamp

    base_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    candidate = ask_root / base_stamp
    suffix = 1
    while candidate.exists():
        candidate = ask_root / f"{base_stamp}-{suffix}"
        suffix += 1
    return candidate
