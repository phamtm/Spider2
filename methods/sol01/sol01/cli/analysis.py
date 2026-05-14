"""CLI command for persisted run analysis."""

from __future__ import annotations

from typing import Annotated, Any

import typer

from sol01.analysis.analysis import analyze_run
from sol01.infra.logging import get_logger

logger = get_logger(__name__)


def register(app: typer.Typer) -> None:
    app.command()(analyze)


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


def handle_analyze(*, run_id: str) -> dict[str, Any]:
    """Run the local trace analyzer for one run."""

    return analyze_run(run_id)
