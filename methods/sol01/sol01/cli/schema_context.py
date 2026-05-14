"""Commands for schema-context cache warming and offline evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any

import typer

from sol01.analysis.schema_context_eval import (
    DEFAULT_GOLD_TABLE_PATH,
    load_gold_tables,
    load_schema_context_eval_task_rows,
    run_schema_context_eval,
    write_schema_context_eval_report,
)
from sol01.cli.common import format_rate
from sol01.cli.run import load_run_tasks
from sol01.infra.config import DEFAULT_DOTENV_PATH, SchemaContextConfig
from sol01.output.output import OUTPUTS_ROOT, ensure_run_paths
from sol01.schema.schema_context_cache import (
    DEFAULT_SCHEMA_CONTEXT_CACHE_ROOT,
    prewarm_schema_context_caches,
)


def register(app: typer.Typer) -> None:
    app.command("prewarm-schema-context")(prewarm_schema_context_command)
    app.command("schema-context-eval")(schema_context_eval_command)


def prewarm_schema_context_command(
    dbs: Annotated[
        list[str],
        typer.Argument(help="Database names to prewarm, for example E_COMMERCE."),
    ],
) -> None:
    """Build schema metadata cache artifacts before running batch workers."""

    caches = prewarm_schema_context_caches_for(dbs)
    typer.echo(
        f"Prewarmed {len(caches)} schema metadata cache(s) into {DEFAULT_SCHEMA_CONTEXT_CACHE_ROOT}"
    )


def schema_context_eval_command(
    gold_path: Annotated[
        Path,
        typer.Option(
            "--gold-path",
            help="Offline JSONL file with instance_id and gold_tables fields.",
        ),
    ] = DEFAULT_GOLD_TABLE_PATH,
    instance_id: Annotated[
        str | None,
        typer.Option(help="Evaluate one exact Spider2-snow instance."),
    ] = None,
    db: Annotated[
        str | None,
        typer.Option(help="Limit evaluation to one database."),
    ] = None,
    question_contains: Annotated[
        str | None,
        typer.Option(help="Keep tasks whose question contains this text."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(min=0, help="Limit how many selected tasks to evaluate."),
    ] = None,
    object_cutoff: Annotated[
        int | None,
        typer.Option(min=1, help="Override the schema-context object cutoff for eval."),
    ] = None,
    covered_only: Annotated[
        bool,
        typer.Option(
            "--covered-only/--all-gold",
            help="Evaluate only tasks whose gold tables touch curated large-schema summaries.",
        ),
    ] = False,
    baseline_path: Annotated[
        Path | None,
        typer.Option(
            "--baseline-path",
            help=(
                "Previous schema-context-eval report.json or tasks.jsonl "
                "for recall regression checks."
            ),
        ),
    ] = None,
    trace_run_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--trace-run-id",
            help="Existing solver run ID whose traces should be scanned for hallucinated columns.",
        ),
    ] = None,
    output_id: Annotated[
        str | None,
        typer.Option(
            help="Persist report artifacts under outputs/<output_id>/schema_context_eval."
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the full evaluation report as JSON."),
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
    """Evaluate schema context coverage against offline gold-table labels."""

    report = handle_schema_context_eval(
        gold_path=gold_path,
        selectors=selectors or [],
        instance_id=instance_id,
        db=db,
        question_contains=question_contains,
        limit=limit,
        object_cutoff=object_cutoff,
        covered_only=covered_only,
        baseline_path=baseline_path,
        trace_run_ids=trace_run_ids or [],
    )
    output_dir = None
    if output_id:
        output_dir = write_schema_context_eval_report(
            report,
            OUTPUTS_ROOT / output_id / "schema_context_eval",
        )
    if json_output:
        payload = report.payload()
        if output_dir is not None:
            payload["output_dir"] = str(output_dir)
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    typer.echo(f"Evaluated {report.task_count} task(s) at object cutoff {report.object_cutoff}.")
    typer.echo(f"Covered-schema tasks: {report.covered_task_count}")
    typer.echo(f"Pre-resolver gold recall: {format_rate(report.pre_resolver_gold_recall)}")
    typer.echo(f"Pre-resolver any-gold recall: {format_rate(report.pre_resolver_any_gold_recall)}")
    typer.echo(f"Post-resolver gold recall: {format_rate(report.post_resolver_gold_recall)}")
    typer.echo(
        f"Post-resolver all-gold recall: {format_rate(report.post_resolver_all_gold_recall)}"
    )
    if report.family_expansion_success is None:
        typer.echo("Family expansion success: n/a")
    else:
        typer.echo(f"Family expansion success: {format_rate(report.family_expansion_success)}")
    typer.echo(f"Average prompt reduction: {format_rate(report.average_prompt_reduction)}")
    typer.echo(f"Recall regressions: {len(report.recall_regressions)}")
    typer.echo(f"Prompt-size wins: {len(report.prompt_size_wins)}")
    typer.echo(f"Hallucinated-column failures: {len(report.hallucinated_column_failures)}")
    if report.failures:
        typer.echo("Missing gold tables:")
        for failure in report.failures[:5]:
            typer.echo(f"- {failure['instance_id']}: {', '.join(failure['missing_gold_tables'])}")
    if output_dir is not None:
        typer.echo(f"Report output: {output_dir}")


def prewarm_schema_context_caches_for(dbs: Iterable[str]) -> list[Any]:
    """Build schema metadata caches once before concurrent solver work begins."""

    config = SchemaContextConfig.from_env(dotenv_path=DEFAULT_DOTENV_PATH)
    return prewarm_schema_context_caches(dbs, config=config)


def handle_schema_context_eval(
    *,
    gold_path: Path,
    selectors: list[str] | None,
    instance_id: str | None,
    db: str | None,
    question_contains: str | None,
    limit: int | None,
    object_cutoff: int | None = None,
    covered_only: bool = False,
    baseline_path: Path | None = None,
    trace_run_ids: list[str] | None = None,
) -> Any:
    """Run offline schema context evaluation for selected tasks."""

    tasks = load_run_tasks(
        selectors=selectors,
        instance_id=instance_id,
        db=db,
        question_contains=question_contains,
        limit=limit,
    )
    gold_tables = load_gold_tables(gold_path)
    tasks = [task for task in tasks if task.instance_id in gold_tables]
    if not tasks:
        raise typer.Exit(code=1)

    config = SchemaContextConfig.from_env(dotenv_path=DEFAULT_DOTENV_PATH)
    if object_cutoff is not None:
        config = config.model_copy(update={"object_cutoff": object_cutoff})
    baseline_tasks = load_schema_context_eval_task_rows(baseline_path) if baseline_path else {}
    trace_dirs = [ensure_run_paths(run_id).traces_dir for run_id in trace_run_ids or []]
    return run_schema_context_eval(
        tasks,
        gold_tables_by_instance=gold_tables,
        config=config,
        covered_only=covered_only,
        baseline_tasks=baseline_tasks,
        trace_dirs=trace_dirs,
    )
