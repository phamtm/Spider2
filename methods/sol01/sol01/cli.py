"""Wire the sol01 commands to the existing index, run, eval, and analysis code."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory
from typing import Annotated, Any

import typer

from sol01.analysis import analyze_run
from sol01.config import DEFAULT_DOTENV_PATH, RuntimeConfig
from sol01.coordinator import run_task, run_tasks
from sol01.eval_runner import run_official_eval
from sol01.index import CACHE_PATH, build_index_cache
from sol01.logging import configure_logging, get_logger
from sol01.models import FinalAnswer, Task
from sol01.output import OUTPUTS_ROOT, ensure_ask_paths, ensure_run_paths
from sol01.tasks import load_tasks

app = typer.Typer(
    help="SQLite-local Spider2-Lite solver.",
    no_args_is_help=True,
)
logger = get_logger(__name__)


@app.command()
def index() -> None:
    """Build the local schema index cache."""

    logger.info("index start")
    payload = handle_index()
    typer.echo(f"Indexed {len(payload)} databases into {CACHE_PATH}")
    logger.info("index complete", database_count=len(payload), cache_path=str(CACHE_PATH))


@app.command()
def run(
    run_id: Annotated[
        str | None,
        typer.Option(help="Run ID used for outputs/<run_id>."),
    ] = None,
    instance_id: Annotated[
        str | None,
        typer.Option(help="Run one exact Spider2-Lite instance."),
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
    local_only: Annotated[
        bool,
        typer.Option("--local-only/--all", help="sol01 only supports local tasks."),
    ] = True,
    force: Annotated[
        bool,
        typer.Option(help="Rerun tasks even when resume would skip them."),
    ] = False,
    skip_failed: Annotated[
        bool,
        typer.Option(help="Skip failed traces during resume mode."),
    ] = False,
) -> None:
    """Run the solver over the selected local tasks."""

    logger.info(
        "run command",
        run_id=run_id,
        instance_id=instance_id,
        db=db,
        question_contains=question_contains,
        limit=limit,
        local_only=local_only,
        force=force,
        skip_failed=skip_failed,
    )
    results = handle_run(
        run_id=run_id,
        instance_id=instance_id,
        db=db,
        question_contains=question_contains,
        limit=limit,
        local_only=local_only,
        force=force,
        skip_failed=skip_failed,
    )
    typer.echo(f"Completed {len(results)} task(s).")
    logger.info("run complete", task_count=len(results))


@app.command("eval")
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
        f"{summary['correct_local_tasks']}/{summary['attempted_local_tasks']} correct, "
        f"missing CSV {summary['missing_csv_count']}"
    )
    logger.info(
        "eval complete",
        run_id=run_id,
        correct_local_tasks=summary["correct_local_tasks"],
        attempted_local_tasks=summary["attempted_local_tasks"],
        missing_csv_count=summary["missing_csv_count"],
    )


@app.command()
def analyze(
    run_id: Annotated[
        str,
        typer.Option(help="Existing run ID to analyze."),
    ],
) -> None:
    """Summarize one run's failures and result buckets."""

    logger.info("analysis command", run_id=run_id)
    report = handle_analyze(run_id=run_id)
    typer.echo(
        "Analysis summary: "
        f"{report['trace_count']} traces, "
        f"{report['status_counts']['failed']} failed"
    )
    logger.info(
        "analysis complete",
        run_id=run_id,
        trace_count=report["trace_count"],
        failed_count=report["status_counts"]["failed"],
    )


@app.command()
def ask(
    db: Annotated[
        str,
        typer.Option(help="SQLite database name to query."),
    ],
    question: Annotated[
        str,
        typer.Argument(help="Ad hoc question to answer."),
    ],
) -> None:
    """Run one ad hoc question against one local SQLite database."""

    logger.info("ask command", db=db)
    answer = handle_ask(db=db, question=question)
    typer.echo(f"Ask status: {answer.status}")
    if answer.csv_path:
        typer.echo(f"CSV: {answer.csv_path}")
    logger.info("ask complete", status=answer.status, csv_path=answer.csv_path)


@app.callback()
def main(
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            envvar="SOL01_LOG_LEVEL",
            help="Logging level for structured console output.",
        ),
    ] = "INFO",
) -> None:
    """Keep the root command focused on subcommands."""

    configure_logging(log_level)
    logger.info("logging configured", log_level=log_level)


def handle_index() -> dict[str, Any]:
    """Build the schema cache and return the in-memory payload."""

    return build_index_cache()


def handle_run(
    *,
    run_id: str | None,
    instance_id: str | None,
    db: str | None,
    question_contains: str | None,
    limit: int | None,
    local_only: bool,
    force: bool,
    skip_failed: bool,
) -> list[Any]:
    """Load local tasks, then pass them to the batch coordinator."""

    _require_local_only(local_only)
    logger.info(
        "loading tasks",
        run_id=run_id,
        instance_id=instance_id,
        db=db,
        question_contains=question_contains,
        limit=limit,
    )
    tasks = _load_filtered_tasks(
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
    )
    effective_run_id = run_id or _default_run_id("run")
    config_summary = _runtime_config_summary(config)
    logger.info(
        "starting batch run",
        run_id=effective_run_id,
        task_count=len(tasks),
        **config_summary,
    )
    return run_tasks(
        tasks,
        run_id=effective_run_id,
        config=config,
        force=force,
        skip_failed=skip_failed,
    )


def handle_eval(
    *,
    run_id: str,
    instance_id: str | None,
    db: str | None,
    question_contains: str | None,
    limit: int | None,
) -> dict[str, Any]:
    """Optionally narrow the manifest task IDs, then run official eval."""

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
            for task in _load_filtered_tasks(
                instance_id=instance_id,
                db=db,
                question_contains=question_contains,
                limit=limit,
            )
        ]
        if not task_ids:
            raise typer.Exit(code=1)
        with TemporaryDirectory(prefix="sol01-eval-") as temp_dir:
            staged_dir = _stage_filtered_eval_results(
                run_id,
                task_ids=task_ids,
                destination=Path(temp_dir),
            )
            return run_official_eval(
                run_id,
                expected_instance_ids=task_ids,
                artifact_tag=_filtered_eval_tag(
                    instance_id=instance_id,
                    db=db,
                    question_contains=question_contains,
                    limit=limit,
                ),
                result_dir=staged_dir,
            )
    return run_official_eval(run_id)


def handle_analyze(*, run_id: str) -> dict[str, Any]:
    """Run the local trace analyzer for one run."""

    return analyze_run(run_id)


def handle_ask(*, db: str, question: str) -> FinalAnswer:
    """Wrap one ad hoc question in a synthetic task and keep the ask layout stable."""

    config = RuntimeConfig.from_env(
        require_api_key=True,
        dotenv_path=DEFAULT_DOTENV_PATH,
    )
    ask_paths = ensure_ask_paths(outputs_root=OUTPUTS_ROOT)
    run_paths = ensure_run_paths("_internal", outputs_root=ask_paths.root)
    task = Task(instance_id="ask", db=db, question=question)
    try:
        logger.info(
            "running ask task",
            db=db,
            ask_root=str(ask_paths.root),
            csv_path=str(ask_paths.csv_path),
            trace_path=str(ask_paths.trace_path),
        )
        answer = run_task(
            task,
            run_paths=run_paths,
            config=config,
            force=True,
        )
        _move_if_exists(run_paths.sql_dir / "ask.sql", ask_paths.sql_path)
        _move_if_exists(run_paths.traces_dir / "ask.json", ask_paths.trace_path)
        if answer.csv_path:
            _move_if_exists(run_paths.csv_dir / "ask.csv", ask_paths.csv_path)
        return FinalAnswer(
            instance_id=answer.instance_id,
            status=answer.status,
            sql=answer.sql,
            csv_path=str(ask_paths.csv_path) if ask_paths.csv_path.exists() else None,
            trace_path=str(ask_paths.trace_path),
        )
    finally:
        rmtree(run_paths.root, ignore_errors=True)
        logger.info("ask cleanup complete", ask_root=str(ask_paths.root))


def _require_local_only(local_only: bool) -> None:
    """Fail fast if someone asks this SQLite-only method to run everything."""

    if not local_only:
        raise typer.BadParameter("sol01 only supports local Spider2-Lite tasks")


def _load_filtered_tasks(
    *,
    instance_id: str | None,
    db: str | None,
    question_contains: str | None,
    limit: int | None,
) -> list[Task]:
    """Keep all task-filter parsing in one small helper."""

    return load_tasks(
        instance_id=instance_id,
        db=db,
        question_contains=question_contains,
        limit=limit,
    )


def _default_run_id(prefix: str) -> str:
    """Generate a stable timestamp-based run ID."""

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{prefix}-{stamp}"


def _runtime_config_summary(config: Any) -> dict[str, Any]:
    """Extract log-safe runtime config fields without assuming the full type."""

    return {
        "model": getattr(config, "model", None),
        "base_url": getattr(config, "base_url", None),
        "concurrency": getattr(config, "concurrency", None),
    }


def _filtered_eval_tag(
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
        parts.append(_slug(instance_id))
    if db:
        parts.append(_slug(db))
    if question_contains:
        parts.append(_slug(question_contains))
    if limit is not None:
        parts.append(f"limit-{limit}")
    digest = hashlib.sha256("\0".join(raw_parts).encode("utf-8")).hexdigest()[:8]
    parts.append(digest)
    return "-".join(parts)


def _move_if_exists(source: Any, destination: Any) -> None:
    """Move one artifact into the ask layout when the source file exists."""

    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)


def _slug(value: str) -> str:
    """Turn a short filter value into a filesystem-friendly label."""

    return "".join(char if char.isalnum() else "-" for char in value).strip("-") or "value"


def _stage_filtered_eval_results(
    run_id: str,
    *,
    task_ids: list[str],
    destination: Path,
) -> Path:
    """Copy only the requested CSVs into a temporary eval result directory."""

    run_paths = ensure_run_paths(run_id)
    destination.mkdir(parents=True, exist_ok=True)
    for task_id in task_ids:
        source = run_paths.csv_dir / f"{task_id}.csv"
        if not source.exists():
            raise typer.BadParameter(
                f"Missing CSV for {task_id} in run {run_id}",
                param_hint="--run-id",
            )
        (destination / source.name).write_bytes(source.read_bytes())
    return destination
