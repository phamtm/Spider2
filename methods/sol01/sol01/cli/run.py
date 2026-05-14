"""Run command and handlers for solver batches."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Literal

import typer

from sol01.analysis.eval_runner import run_persisted_eval
from sol01.cli.common import default_run_id, runtime_config_summary
from sol01.coordinator import run_tasks
from sol01.infra.config import DEFAULT_DOTENV_PATH, RuntimeConfig, SchemaContextConfig
from sol01.infra.logging import get_logger
from sol01.infra.strings import question_preview
from sol01.infra.time_utils import format_duration
from sol01.loading.tasks import load_tasks, select_tasks
from sol01.models import FinalAnswer, Task
from sol01.output.output import OUTPUTS_ROOT, RunPaths, ensure_run_paths
from sol01.output.registry import RegistryTaskRecord, record_registry_batch

logger = get_logger(__name__)


def register(app: typer.Typer) -> None:
    app.command()(run)


def run(
    concurrency: Annotated[
        int | None,
        typer.Option(min=1, help="Maximum number of concurrent solver tasks."),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(help="Run ID used for outputs/<run_id>."),
    ] = None,
    instance_id: Annotated[
        str | None,
        typer.Option(help="Run one exact Spider2-snow instance."),
    ] = None,
    db: Annotated[
        str | None,
        typer.Option(help="Limit tasks to one database."),
    ] = None,
    question_contains: Annotated[
        str | None,
        typer.Option(help="Keep tasks whose question contains this text."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(min=0, help="Limit how many tasks to run."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(help="Rerun tasks even when resume would skip them."),
    ] = False,
    skip_failed: Annotated[
        bool,
        typer.Option(help="Skip failed traces during resume mode."),
    ] = False,
    selectors: Annotated[
        list[str] | None,
        typer.Argument(
            help=(
                "Optional task/category selectors: exact IDs, globs, tier:<n>, tag:<name>, or all."
            ),
        ),
    ] = None,
) -> None:
    """Run the solver over the selected Spider2-snow tasks."""

    started_at = perf_counter()
    logger.info(
        "run command",
        run_id=run_id,
        instance_id=instance_id,
        db=db,
        question_contains=question_contains,
        limit=limit,
        force=force,
        skip_failed=skip_failed,
        selectors=selectors or [],
    )
    kwargs: dict[str, Any] = {
        "run_id": run_id,
        "selectors": selectors or [],
        "instance_id": instance_id,
        "db": db,
        "question_contains": question_contains,
        "limit": limit,
        "force": force,
        "skip_failed": skip_failed,
    }
    if concurrency is not None:
        kwargs["concurrency"] = concurrency
    results = handle_run(**kwargs)
    typer.echo(f"Completed {len(results['results'])} task(s).")
    typer.echo(
        "Eval summary: "
        f"{results['eval_summary']['correct_tasks']}/"
        f"{results['eval_summary']['attempted_tasks']} correct, "
        f"missing CSV {results['eval_summary']['missing_csv_count']}"
    )
    if len(results["tasks"]) == 1:
        typer.echo(f"Exec time: {format_duration(perf_counter() - started_at)}")
    for line in run_eval_lines(
        tasks=results["tasks"],
        answers=results["results"],
        eval_summary=results["eval_summary"],
    ):
        typer.echo(line)


def handle_run(
    *,
    concurrency: int | None = None,
    run_id: str | None,
    instance_id: str | None,
    db: str | None,
    question_contains: str | None,
    limit: int | None,
    force: bool,
    skip_failed: bool,
    selectors: list[str] | None = None,
    outputs_root: Path | None = None,
) -> dict[str, Any]:
    """Load tasks, run the coordinator, evaluate outputs, and update the registry."""

    tasks = load_run_tasks(
        selectors=selectors,
        instance_id=instance_id,
        db=db,
        question_contains=question_contains,
        limit=limit,
    )
    if not tasks:
        logger.warning("no tasks matched the filters")
        raise typer.Exit(code=1)

    config = RuntimeConfig.from_env(
        require_api_key=True,
        dotenv_path=DEFAULT_DOTENV_PATH,
        **({"concurrency": concurrency} if concurrency is not None else {}),
    )
    schema_context_config = SchemaContextConfig.from_env(dotenv_path=DEFAULT_DOTENV_PATH)
    effective_run_id = run_id or default_run_id("run")
    effective_outputs_root = outputs_root or OUTPUTS_ROOT
    logger.info(
        "run start",
        run_id=effective_run_id,
        task_count=len(tasks),
        **runtime_config_summary(config),
    )
    results = run_tasks(
        tasks,
        run_id=effective_run_id,
        config=config,
        schema_context_config=schema_context_config,
        force=force,
        skip_failed=skip_failed,
    )
    eval_summary = run_persisted_eval(
        effective_run_id,
        expected_instance_ids=[task.instance_id for task in tasks],
        outputs_root=effective_outputs_root,
    )
    run_paths = ensure_run_paths(effective_run_id, outputs_root=effective_outputs_root)
    registry_records = build_registry_records(
        run_id=effective_run_id,
        tasks=tasks,
        results=results,
        eval_summary=eval_summary,
        run_paths=run_paths,
    )
    if registry_records:
        record_registry_batch(registry_records, outputs_root=effective_outputs_root)
    return {"tasks": tasks, "results": results, "eval_summary": eval_summary}


def build_registry_records(
    *,
    run_id: str,
    tasks: list[Task],
    results: list[FinalAnswer],
    eval_summary: dict[str, Any],
    run_paths: RunPaths,
) -> list[RegistryTaskRecord]:
    """Translate one solver and eval round into registry-ready task records."""

    timestamp = datetime.now(UTC).isoformat()
    overall_eval_error = eval_summary.get("eval_error")
    answers_by_id = {answer.instance_id: answer for answer in results}
    per_instance_by_id = {
        row["instance_id"]: row
        for row in eval_summary.get("per_instance", [])
        if isinstance(row, dict) and isinstance(row.get("instance_id"), str)
    }
    run_path = str(run_paths.root)
    eval_path = str(run_paths.eval_dir / "summary.json")

    records: list[RegistryTaskRecord] = []
    for task in tasks:
        answer = answers_by_id.get(task.instance_id)
        per_row = per_instance_by_id.get(task.instance_id)
        failure_reason = per_row.get("failure_reason") if per_row else None
        csv_present = per_row.get("csv_present") if per_row else None
        csv_path = None if csv_present is False else answer.csv_path if answer is not None else None

        if overall_eval_error:
            eval_status: Literal["success", "failed"] | None = "failed"
            eval_error: str | None = str(overall_eval_error)
        elif failure_reason == "eval_failed":
            eval_status = "failed"
            eval_error = "eval_failed"
        elif per_row is not None:
            eval_status = "success"
            eval_error = None
        else:
            eval_status = None
            eval_error = None

        records.append(
            RegistryTaskRecord(
                run_id=run_id,
                instance_id=task.instance_id,
                db=task.db,
                timestamp=timestamp,
                score=per_row.get("score") if per_row else None,
                run_path=run_path,
                csv_path=csv_path,
                trace_path=answer.trace_path if answer else None,
                eval_path=eval_path,
                solver_status=answer.status if answer else None,
                eval_status=eval_status,
                eval_error=eval_error,
                extra_artifacts={
                    "llm_call_log_path": str(run_paths.llm_calls_dir / f"{task.instance_id}.jsonl"),
                },
            )
        )
    return records


def load_filtered_tasks(
    *,
    instance_id: str | None,
    db: str | None,
    question_contains: str | None,
    limit: int | None,
) -> list[Task]:
    """Keep legacy filter parsing in one place."""

    return load_tasks(
        instance_id=instance_id,
        db=db,
        question_contains=question_contains,
        limit=limit,
    )


def load_run_tasks(
    *,
    selectors: Iterable[str] | None,
    instance_id: str | None,
    db: str | None,
    question_contains: str | None,
    limit: int | None,
) -> list[Task]:
    """Load tasks from selectors or from the older filter options."""

    normalized_selectors = [selector.strip() for selector in selectors or [] if selector.strip()]
    if not normalized_selectors:
        return load_filtered_tasks(
            instance_id=instance_id,
            db=db,
            question_contains=question_contains,
            limit=limit,
        )
    if instance_id is not None:
        raise typer.BadParameter("positional selectors cannot be combined with --instance-id")

    tasks = select_tasks(normalized_selectors)
    if db is not None:
        tasks = [task for task in tasks if task.db == db]
    if question_contains:
        needle = question_contains.casefold()
        tasks = [task for task in tasks if needle in task.question.casefold()]
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def run_eval_lines(
    *,
    tasks: list[Task],
    answers: list[FinalAnswer],
    eval_summary: dict[str, Any],
) -> list[str]:
    """Render one concise per-task eval line for the run command output."""

    answer_by_id = {answer.instance_id: answer for answer in answers}
    eval_by_id = {
        row["instance_id"]: row
        for row in eval_summary.get("per_instance", [])
        if row.get("instance_id")
    }
    lines: list[str] = []
    for task in tasks:
        answer = answer_by_id.get(task.instance_id)
        eval_row = eval_by_id.get(task.instance_id, {})
        eval_label = "PASS" if eval_row.get("passed") else "FAIL"
        task_status = answer.status if answer is not None else "missing"
        question = question_preview(task.question, max_length=90)
        lines.append(f"- {task.instance_id}: {eval_label} | task {task_status} | {question}")
    return lines
