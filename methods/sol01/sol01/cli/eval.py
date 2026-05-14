"""Eval command and helpers for saved solver runs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Any

import typer

from sol01.analysis.eval_runner import run_official_eval
from sol01.cli.common import slug
from sol01.infra.logging import get_logger
from sol01.loading.tasks import load_tasks
from sol01.output.output import ensure_run_paths, eval_input_csv_dir_for

logger = get_logger(__name__)


def register(app: typer.Typer) -> None:
    app.command("eval")(eval_command)


def eval_command(
    run_id: Annotated[
        str,
        typer.Option(help="Existing run ID to evaluate."),
    ],
    instance_id: Annotated[
        str | None,
        typer.Option(help="Restrict eval accounting to one instance."),
    ] = None,
    db: Annotated[
        str | None,
        typer.Option(help="Restrict eval accounting to one database."),
    ] = None,
    question_contains: Annotated[
        str | None,
        typer.Option(help="Restrict eval accounting by question text."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(min=0, help="Restrict eval accounting to the first N tasks."),
    ] = None,
) -> None:
    """Run the official evaluator over one saved run."""

    logger.info(
        "eval command",
        run_id=run_id,
        instance_id=instance_id,
        db=db,
        question_contains=question_contains,
        limit=limit,
    )
    summary = handle_eval(
        run_id=run_id,
        instance_id=instance_id,
        db=db,
        question_contains=question_contains,
        limit=limit,
    )
    typer.echo(
        "Eval summary: "
        f"{summary['correct_tasks']}/{summary['attempted_tasks']} correct, "
        f"missing CSV {summary['missing_csv_count']}"
    )
    logger.info(
        "eval complete",
        run_id=run_id,
        correct_tasks=summary["correct_tasks"],
        attempted_tasks=summary["attempted_tasks"],
        missing_csv_count=summary["missing_csv_count"],
    )


def handle_eval(
    *,
    run_id: str,
    instance_id: str | None,
    db: str | None,
    question_contains: str | None,
    limit: int | None,
) -> dict[str, Any]:
    """Optionally narrow task IDs, then run the official evaluator."""

    if any(value is not None for value in (instance_id, db, question_contains, limit)):
        logger.info(
            "running filtered eval",
            run_id=run_id,
            instance_id=instance_id,
            db=db,
            question_contains=question_contains,
            limit=limit,
        )
        task_ids = [
            task.instance_id
            for task in load_tasks(
                instance_id=instance_id,
                db=db,
                question_contains=question_contains,
                limit=limit,
            )
        ]
        if not task_ids:
            raise typer.Exit(code=1)
        artifact_tag = filtered_eval_tag(
            instance_id=instance_id,
            db=db,
            question_contains=question_contains,
            limit=limit,
        )
        run_paths = ensure_run_paths(run_id)
        staged_dir = stage_filtered_eval_results(
            run_id,
            run_paths,
            task_ids=task_ids,
            destination=eval_input_csv_dir_for(run_paths, eval_id=artifact_tag),
        )
        return run_official_eval(
            run_id,
            expected_instance_ids=task_ids,
            artifact_tag=artifact_tag,
            result_dir=staged_dir,
        )
    return run_official_eval(run_id)


def filtered_eval_tag(
    *,
    instance_id: str | None,
    db: str | None,
    question_contains: str | None,
    limit: int | None,
) -> str:
    """Build a short artifact tag for filtered eval outputs."""

    parts = ["filtered"]
    raw_parts = [
        instance_id or "",
        db or "",
        question_contains or "",
        "" if limit is None else str(limit),
    ]
    if instance_id:
        parts.append(slug(instance_id))
    if db:
        parts.append(slug(db))
    if question_contains:
        parts.append(slug(question_contains))
    if limit is not None:
        parts.append(f"limit-{limit}")
    digest = hashlib.sha256("\0".join(raw_parts).encode("utf-8")).hexdigest()[:8]
    parts.append(digest)
    return "-".join(parts)


def stage_filtered_eval_results(
    run_id: str,
    run_paths: Any,
    *,
    task_ids: list[str],
    destination: Path,
) -> Path:
    """Copy only the requested CSVs into a durable filtered eval input directory."""

    destination.mkdir(parents=True, exist_ok=True)
    for stale_csv in destination.glob("*.csv"):
        stale_csv.unlink()
    for task_id in task_ids:
        source = run_paths.csv_dir / f"{task_id}.csv"
        if not source.exists():
            raise typer.BadParameter(
                f"Missing CSV for {task_id} in run {run_id}",
                param_hint="--run-id",
            )
        (destination / source.name).write_bytes(source.read_bytes())
    return destination
