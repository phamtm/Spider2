"""Persisted local run mode for sol01."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from sol01.config import DEFAULT_DOTENV_PATH, RuntimeConfig
from sol01.coordinator import run_tasks
from sol01.eval_runner import run_persisted_eval
from sol01.logging import get_logger
from sol01.output import (
    OUTPUTS_ROOT,
    ensure_run_paths,
    run_log_path_for,
    stderr_log_path_for,
    stdout_log_path_for,
)
from sol01.registry import RegistryTaskRecord, record_registry_batch
from sol01.tasks import ALL_TASK_SELECTOR, select_tasks

logger = get_logger(__name__)


def run_persisted_mode(
    selectors: Sequence[str] | str | None = None,
    *,
    all_mode: bool = False,
    outputs_root: Path = OUTPUTS_ROOT,
    run_id: str | None = None,
    config: RuntimeConfig | None = None,
) -> dict[str, Any]:
    """Run the solver and evaluator with durable local artifacts."""

    normalized_selectors = _normalize_selectors(selectors, all_mode=all_mode)
    tasks = _resolve_tasks(normalized_selectors, all_mode=all_mode)
    mode_label = _mode_label(normalized_selectors, all_mode=all_mode)
    effective_run_id = run_id or _build_run_id(mode_label)
    run_root = outputs_root / effective_run_id
    if run_root.exists():
        raise ValueError(f"run_id collision: {effective_run_id}")

    run_paths = ensure_run_paths(effective_run_id, outputs_root=outputs_root)
    stdout_path = stdout_log_path_for(run_paths)
    stderr_path = stderr_log_path_for(run_paths)
    run_events_path = run_log_path_for(run_paths)

    _append_run_event(
        run_events_path,
        {
            "event": "run_started",
            "run_id": effective_run_id,
            "run_path": str(run_paths.root),
            "mode": mode_label,
            "selectors": normalized_selectors,
            "task_count": len(tasks),
        },
    )

    runtime_config = config or RuntimeConfig.from_env(
        require_api_key=True,
        dotenv_path=DEFAULT_DOTENV_PATH,
    )
    logger.info(
        "persisted run start",
        run_id=effective_run_id,
        run_path=str(run_paths.root),
        task_count=len(tasks),
        mode=mode_label,
    )

    try:
        results = run_tasks(
            tasks,
            run_id=effective_run_id,
            config=runtime_config,
            outputs_root=outputs_root,
            force=False,
            skip_failed=False,
        )
        _append_run_event(
            run_events_path,
            {
                "event": "solver_complete",
                "run_id": effective_run_id,
                "success_count": sum(1 for result in results if result.status == "success"),
                "failed_count": sum(1 for result in results if result.status == "failed"),
                "skipped_count": sum(1 for result in results if result.status == "skipped"),
            },
        )

        eval_summary = run_persisted_eval(
            effective_run_id,
            expected_instance_ids=[task.instance_id for task in tasks],
            outputs_root=outputs_root,
        )
        _append_run_event(
            run_events_path,
            {
                "event": "eval_complete",
                "run_id": effective_run_id,
                "correct_tasks": eval_summary["correct_tasks"],
                "attempted_tasks": eval_summary["attempted_tasks"],
                "missing_csv_count": eval_summary["missing_csv_count"],
            },
        )

        registry_summary = _update_registry(
            run_id=effective_run_id,
            run_path=run_paths.root,
            run_paths=run_paths,
            tasks=tasks,
            results=results,
            eval_summary=eval_summary,
            outputs_root=outputs_root,
        )
        _append_run_event(
            run_events_path,
            {
                "event": "registry_updated",
                "run_id": effective_run_id,
                "registered_tasks": len(registry_summary["task_results"]),
            },
        )

        stdout_lines = [
            f"Run ID: {effective_run_id}",
            f"Run path: {run_paths.root}",
            f"Task count: {len(tasks)}",
            (
                "Summary: "
                f"{eval_summary['correct_tasks']}/{eval_summary['attempted_tasks']} correct, "
                f"{sum(1 for row in eval_summary['per_instance'] if row.get('passed'))} passed, "
                f"{sum(1 for row in eval_summary['per_instance'] if not row.get('passed'))} failed"
            ),
        ]
        stdout_text = "\n".join(stdout_lines) + "\n"
        stdout_path.write_text(stdout_text, encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        print(stdout_text, end="")

        _append_run_event(
            run_events_path,
            {
                "event": "run_complete",
                "run_id": effective_run_id,
                "run_path": str(run_paths.root),
                "task_count": len(tasks),
                "passed_count": sum(1 for row in eval_summary["per_instance"] if row.get("passed")),
                "failed_count": sum(1 for row in eval_summary["per_instance"] if not row.get("passed")),
            },
        )

        return {
            "run_id": effective_run_id,
            "run_path": str(run_paths.root),
            "task_count": len(tasks),
            "tasks": tasks,
            "results": results,
            "eval_summary": eval_summary,
            "registry_summary": registry_summary,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "run_events_path": str(run_events_path),
        }
    except Exception as exc:
        error_text = f"{exc.__class__.__name__}: {exc}"
        stderr_path.write_text(error_text + "\n", encoding="utf-8")
        _append_run_event(
            run_events_path,
            {
                "event": "run_failed",
                "run_id": effective_run_id,
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
            },
        )
        logger.exception(
            "persisted run failed",
            run_id=effective_run_id,
            run_path=str(run_paths.root),
            error_type=exc.__class__.__name__,
            error_message=str(exc),
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for persisted run mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "patterns",
        nargs="*",
        help="Task selectors such as sf_bq320, sf_bq3*, or sf_bq3* sf_bq4*.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the full Spider2-snow task set.",
    )
    args = parser.parse_args(argv)

    try:
        run_persisted_mode(args.patterns or None, all_mode=args.all)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


def _update_registry(
    *,
    run_id: str,
    run_path: Path,
    run_paths,
    tasks,
    results,
    eval_summary,
    outputs_root: Path,
) -> dict[str, Any]:
    """Build registry rows from solver outcomes and eval results."""

    per_instance = {
        row["instance_id"]: row for row in eval_summary.get("per_instance", []) if row.get("instance_id")
    }
    eval_status = "failed" if eval_summary.get("returncode", 0) != 0 else "success"
    records: list[RegistryTaskRecord] = []
    now = _utc_now()
    for task, result in zip(tasks, results):
        row = per_instance.get(task.instance_id, {})
        failure_reason = row.get("failure_reason")
        records.append(
            RegistryTaskRecord(
                run_id=run_id,
                instance_id=task.instance_id,
                db=task.db,
                timestamp=now,
                score=row.get("score"),
                run_path=str(run_path),
                csv_path=None if failure_reason == "missing_csv" else result.csv_path,
                trace_path=result.trace_path,
                eval_path=str(run_paths.eval_dir / "summary.json"),
                solver_status=result.status,
                eval_status=eval_status,
                eval_error=None if eval_status == "success" else "official_eval_failed",
                extra_artifacts={
                    "solver_csv_path": result.csv_path,
                    "stdout_path": str(run_paths.logs_dir / "stdout.txt"),
                    "stderr_path": str(run_paths.logs_dir / "stderr.txt"),
                    "run_events_path": str(run_paths.logs_dir / "run.jsonl"),
                    "per_instance_path": str(run_paths.eval_dir / "per_instance.jsonl"),
                },
            )
        )
    return record_registry_batch(records, outputs_root=outputs_root)


def _resolve_tasks(selectors: Sequence[str], *, all_mode: bool) -> list[Any]:
    """Resolve selectors into dataset-ordered tasks."""

    if all_mode:
        return select_tasks([ALL_TASK_SELECTOR])
    return select_tasks(selectors)


def _normalize_selectors(
    selectors: Sequence[str] | str | None,
    *,
    all_mode: bool,
) -> list[str]:
    """Normalize user input into selector strings."""

    if all_mode:
        if selectors is not None:
            raise ValueError("all mode cannot be combined with selectors")
        return [ALL_TASK_SELECTOR]
    if selectors is None:
        raise ValueError("selectors must not be empty")
    if isinstance(selectors, str):
        return [selectors]
    return list(selectors)


def _mode_label(selectors: Sequence[str], *, all_mode: bool) -> str:
    """Build the run-id prefix from the selected mode."""

    if all_mode or selectors == [ALL_TASK_SELECTOR]:
        return "all"
    if len(selectors) == 1 and _is_exact_selector(selectors[0]):
        return f"exact-{_slug(selectors[0])}"
    digest = hashlib.sha256("\0".join(selectors).encode("utf-8")).hexdigest()[:8]
    return f"patterns-{digest}"


def _is_exact_selector(selector: str) -> bool:
    """Return True when a selector has no glob tokens."""

    return not any(token in selector for token in ("*", "?", "[", "]"))


def _build_run_id(mode_label: str) -> str:
    """Create a unique timestamped run id."""

    return f"run-{mode_label}-{_utc_now()}"


def _append_run_event(path: Path, event: dict[str, Any]) -> None:
    """Append one structured run event to the run log."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("timestamp", _utc_now())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def _utc_now() -> str:
    """Return a stable UTC timestamp."""

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")


def _slug(value: str) -> str:
    """Turn a selector into a compact filesystem-friendly label."""

    return "".join(char if char.isalnum() else "-" for char in value).strip("-") or "value"


if __name__ == "__main__":
    raise SystemExit(main())
