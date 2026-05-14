"""Run one Spider2-snow gold SQL question as a persisted gold run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from sol01.analysis.eval_runner import GOLD_DIR, run_official_eval
from sol01.execution.snowflake_runner import execute_sql
from sol01.infra.time_utils import format_duration
from sol01.loading.tasks import select_tasks
from sol01.models import ExecutionResult, Task
from sol01.output.output import (
    OUTPUTS_ROOT,
    RunPaths,
    ensure_run_paths,
    run_log_path_for,
    stderr_log_path_for,
    stdout_log_path_for,
    write_manifest,
    write_sql,
)

GOLD_SQL_DIR = GOLD_DIR / "sql"


class EvaluatorRunner(Protocol):
    """Subprocess-compatible callable used by tests."""

    def __call__(
        self,
        command: list[str],
        *,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]: ...


@dataclass(frozen=True)
class GoldRunResult:
    """Result for one persisted gold run."""

    question: Task
    execution: ExecutionResult
    score: int | None
    run_id: str
    run_paths: RunPaths
    eval_summary: dict[str, object]


class GoldRunInputError(ValueError):
    """Raised when the requested gold run inputs are invalid."""


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `just gold <instance_id>`."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance_id", help="Spider2-snow instance ID, for example sf_bq320.")
    args = parser.parse_args(argv)

    started_at = perf_counter()
    try:
        result = run_question(args.instance_id)
    except GoldRunInputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Run ID: {result.run_id}")
    print(f"Question: {result.question.instance_id}")
    print(f"Database: {result.question.db}")
    print(f"Rows: {result.execution.row_count}")
    print(f"Score: {result.score}")
    print(f"Run path: {result.run_paths.root}")
    print(f"Exec time: {format_duration(perf_counter() - started_at)}")
    if result.score == 1:
        print("Gold run passed.")
        return 0

    print("Gold run failed.", file=sys.stderr)
    return 1


def run_question(
    question_id: str,
    *,
    evaluator_runner: EvaluatorRunner | None = None,
    outputs_root: Path = OUTPUTS_ROOT,
) -> GoldRunResult:
    """Execute one gold SQL file and score it with the official CSV evaluator."""

    question = _load_question(question_id)
    sql_path = GOLD_SQL_DIR / f"{question_id}.sql"
    if not sql_path.exists():
        raise GoldRunInputError(f"Gold SQL not found for {question_id}: {sql_path}")

    sql = sql_path.read_text(encoding="utf-8")
    run_id = _build_run_id(question_id)
    run_paths = ensure_run_paths(run_id, outputs_root=outputs_root)
    write_manifest(
        run_paths,
        manifest={
            "task_ids": [question_id],
            "workflow": "gold",
            "question_id": question_id,
            "database": question.db,
            "gold_sql_path": str(sql_path),
        },
    )
    write_sql(run_paths, instance_id=question_id, sql=sql)
    _append_run_event(
        run_paths,
        {
            "event": "gold_run_started",
            "run_id": run_id,
            "question_id": question_id,
            "question_db": question.db,
            "run_path": str(run_paths.root),
        },
    )

    csv_path = run_paths.csv_dir / f"{question_id}.csv"
    execution = execute_sql(sql, db=question.db, csv_path=csv_path)
    if not execution.ok:
        error_message = execution.error or f"Snowflake query failed for {question_id}"
        _write_run_logs(
            run_paths,
            stdout_lines=[
                f"Run ID: {run_id}",
                f"Question: {question.instance_id}",
                f"Database: {question.db}",
                f"Run path: {run_paths.root}",
                "Status: failed",
            ],
            stderr_text=error_message,
        )
        _append_run_event(
            run_paths,
            {
                "event": "gold_run_failed",
                "run_id": run_id,
                "question_id": question_id,
                "stage": "sql_execution",
                "error": error_message,
            },
        )
        raise RuntimeError(error_message)

    try:
        eval_summary = run_official_eval(
            run_id,
            outputs_root=outputs_root,
            expected_instance_ids=[question_id],
            result_dir=run_paths.csv_dir,
            runner=evaluator_runner,
        )
    except subprocess.CalledProcessError as exc:
        error_message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        _write_run_logs(
            run_paths,
            stdout_lines=[
                f"Run ID: {run_id}",
                f"Question: {question.instance_id}",
                f"Database: {question.db}",
                f"Rows: {execution.row_count}",
                f"Run path: {run_paths.root}",
                "Status: failed",
            ],
            stderr_text=error_message,
        )
        _append_run_event(
            run_paths,
            {
                "event": "gold_run_failed",
                "run_id": run_id,
                "question_id": question_id,
                "stage": "official_eval",
                "returncode": exc.returncode,
                "error": error_message,
            },
        )
        raise

    score = (
        eval_summary["instance_scores"].get(question_id)
        if isinstance(eval_summary.get("instance_scores"), dict)
        else None
    )
    stdout_text = "\n".join(
        [
            f"Run ID: {run_id}",
            f"Question: {question.instance_id}",
            f"Database: {question.db}",
            f"Rows: {execution.row_count}",
            f"Score: {score}",
            f"Run path: {run_paths.root}",
        ]
    )
    _write_run_logs(run_paths, stdout_text=stdout_text, stderr_text="")
    _append_run_event(
        run_paths,
        {
            "event": "gold_run_complete",
            "run_id": run_id,
            "question_id": question_id,
            "score": score,
            "run_path": str(run_paths.root),
        },
    )
    return GoldRunResult(
        question=question,
        execution=execution,
        score=score,
        run_id=run_id,
        run_paths=run_paths,
        eval_summary=eval_summary,
    )


def _load_question(question_id: str) -> Task:
    """Load one exact Spider2-snow task."""

    try:
        matches = select_tasks([question_id])
    except ValueError as exc:
        raise GoldRunInputError(f"Unknown Spider2-snow question: {question_id}") from exc
    if not matches:
        raise GoldRunInputError(f"Unknown Spider2-snow question: {question_id}")
    return matches[0]


def _build_run_id(question_id: str) -> str:
    """Generate a stable timestamp-based gold run ID."""

    stamp = datetime.now(UTC).strftime("%y%m%d.%H%M")
    token = uuid4().hex[:8]
    return f"gold-{question_id}-{stamp}-{token}"


def _append_run_event(run_paths: RunPaths, event: dict[str, object]) -> None:
    """Record one run event in the append-only log."""

    run_log_path = run_log_path_for(run_paths)
    with run_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _write_run_logs(
    run_paths: RunPaths,
    *,
    stdout_text: str | None = None,
    stdout_lines: list[str] | None = None,
    stderr_text: str,
) -> None:
    """Write the concise run logs under the persisted run root."""

    if stdout_text is None:
        stdout_text = "\n".join(stdout_lines or [])
    if stdout_text and not stdout_text.endswith("\n"):
        stdout_text += "\n"
    if stderr_text and not stderr_text.endswith("\n"):
        stderr_text += "\n"
    stdout_log_path_for(run_paths).write_text(stdout_text, encoding="utf-8")
    stderr_log_path_for(run_paths).write_text(stderr_text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
