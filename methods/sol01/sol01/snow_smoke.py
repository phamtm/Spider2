"""Run one Spider2-snow gold SQL question as a smoke test."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from sol01.eval_runner import EVALUATE_SCRIPT, GOLD_DIR, parse_eval_stdout
from sol01.models import ExecutionResult, Task
from sol01.snowflake_runner import execute_sql
from sol01.tasks import REPO_ROOT, load_tasks

GOLD_SQL_DIR = GOLD_DIR / "sql"
SPIDER2_SNOW_JSONL = REPO_ROOT / "spider2-snow" / "spider2-snow.jsonl"


class EvaluatorRunner(Protocol):
    """Subprocess-compatible callable used by tests."""

    def __call__(
        self,
        command: list[str],
        *,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]: ...


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `just smoke <question>`."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Spider2-snow instance ID, for example sf_bq320.")
    args = parser.parse_args(argv)

    try:
        result = run_question(args.question)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Question: {result.question.instance_id}")
    print(f"Database: {result.question.db}")
    print(f"Rows: {result.execution.row_count}")
    print(f"Score: {result.score}")
    if result.score == 1:
        print("Smoke test passed.")
        return 0

    print("Smoke test failed.", file=sys.stderr)
    return 1


class SmokeResult:
    """Result for one smoke run."""

    def __init__(self, *, question: Task, execution: ExecutionResult, score: int | None) -> None:
        self.question = question
        self.execution = execution
        self.score = score


def run_question(
    question_id: str,
    *,
    evaluator_runner: EvaluatorRunner | None = None,
) -> SmokeResult:
    """Execute one gold SQL file and score it with the official CSV evaluator."""

    question = _load_question(question_id)
    sql_path = GOLD_SQL_DIR / f"{question_id}.sql"
    if not sql_path.exists():
        raise ValueError(f"Gold SQL not found for {question_id}: {sql_path}")

    sql = sql_path.read_text(encoding="utf-8")
    with TemporaryDirectory(prefix=f"sol01-smoke-{question_id}-") as temp_dir:
        result_dir = Path(temp_dir) / "result"
        csv_path = result_dir / f"{question_id}.csv"
        execution = execute_sql(sql, db=question.db, csv_path=csv_path)
        if not execution.ok:
            raise ValueError(f"Snowflake query failed for {question_id}: {execution.error}")

        evaluator = _run_official_csv_eval(
            result_dir,
            evaluator_runner=evaluator_runner,
        )
        if evaluator.returncode != 0:
            message = evaluator.stderr.strip() or evaluator.stdout.strip()
            raise ValueError(f"Official evaluator failed for {question_id}: {message}")

        summary = parse_eval_stdout(evaluator.stdout)
        return SmokeResult(
            question=question,
            execution=execution,
            score=summary["instance_scores"].get(question_id),
        )


def _load_question(question_id: str) -> Task:
    """Load one exact Spider2-snow task."""

    matches = load_tasks(instance_id=question_id)
    if not matches:
        raise ValueError(f"Unknown Spider2-snow question: {question_id}")
    return matches[0]


def _run_official_csv_eval(
    result_dir: Path,
    *,
    evaluator_runner: EvaluatorRunner | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the official evaluator in a temp cwd so it cannot write repo logs."""

    with TemporaryDirectory(prefix="sol01-smoke-eval-") as temp_dir:
        temp_root = Path(temp_dir)
        eval_cwd = temp_root / "spider2-snow" / "evaluation_suite"
        eval_cwd.mkdir(parents=True)
        shutil.copy2(SPIDER2_SNOW_JSONL, eval_cwd.parent / "spider2-snow.jsonl")

        command = [
            sys.executable,
            str(EVALUATE_SCRIPT),
            "--mode",
            "exec_result",
            "--result_dir",
            str(result_dir),
            "--gold_dir",
            str(GOLD_DIR),
            "--max_workers",
            "1",
        ]
        return (evaluator_runner or _run_subprocess)(command, cwd=eval_cwd)


def _run_subprocess(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and capture its output."""

    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
