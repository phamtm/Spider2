"""Ad hoc ask command."""

from __future__ import annotations

from pathlib import Path
from shutil import rmtree
from typing import Annotated, Any

import typer

from sol01.coordinator import run_task
from sol01.infra.config import DEFAULT_DOTENV_PATH, RuntimeConfig, SchemaContextConfig
from sol01.infra.logging import get_logger
from sol01.models import FinalAnswer, Task
from sol01.output.output import OUTPUTS_ROOT, ensure_ask_paths, ensure_run_paths

logger = get_logger(__name__)


def register(app: typer.Typer) -> None:
    app.command()(ask)


def ask(
    db: Annotated[
        str,
        typer.Option(help="Snowflake database name to query."),
    ],
    question: Annotated[
        str,
        typer.Argument(help="Ad hoc question to answer."),
    ],
) -> None:
    """Run one ad hoc question against one Snowflake database."""

    logger.info("ask command", db=db)
    answer = handle_ask(db=db, question=question)
    typer.echo(f"Ask status: {answer.status}")
    if answer.csv_path:
        typer.echo(f"CSV: {answer.csv_path}")
    logger.info("ask complete", status=answer.status, csv_path=answer.csv_path)


def handle_ask(*, db: str, question: str) -> FinalAnswer:
    """Wrap one ad hoc question in a synthetic task and keep the ask layout stable."""

    config = RuntimeConfig.from_env(require_api_key=True, dotenv_path=DEFAULT_DOTENV_PATH)
    schema_context_config = SchemaContextConfig.from_env(dotenv_path=DEFAULT_DOTENV_PATH)
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
            schema_context_config=schema_context_config,
            force=True,
        )
        move_if_exists(run_paths.sql_dir / "ask.sql", ask_paths.sql_path)
        move_if_exists(run_paths.traces_dir / "ask.json", ask_paths.trace_path)
        if answer.csv_path:
            move_if_exists(run_paths.csv_dir / "ask.csv", ask_paths.csv_path)
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


def move_if_exists(source: Any, destination: Path) -> None:
    """Move one artifact into the ask layout when the source file exists."""

    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)
