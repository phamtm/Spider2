"""Wire the sol01 commands to the existing index, run, eval, and analysis code."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from shutil import rmtree
from time import perf_counter
from typing import Annotated, Any

import typer

from sol01.analysis.analysis import analyze_run
from sol01.analysis.eval_runner import run_official_eval, run_persisted_eval
from sol01.coordinator import run_task, run_tasks
from sol01.infra.config import DEFAULT_DOTENV_PATH, RuntimeConfig, SchemaRetrievalConfig
from sol01.infra.logging import configure_logging, get_logger
from sol01.infra.observability import configure_logfire
from sol01.infra.time_utils import format_duration
from sol01.llm.llm_call_logs import (
    build_llm_call_detail_sections,
    build_llm_call_summary_row,
    build_llm_call_summary_rows,
    format_llm_call_value,
    load_llm_call_log,
)
from sol01.loading.tasks import load_tasks, select_tasks
from sol01.models import FinalAnswer, Task
from sol01.output.output import (
    OUTPUTS_ROOT,
    ensure_ask_paths,
    ensure_run_paths,
    eval_input_csv_dir_for,
)
from sol01.output.registry import resolve_llm_call_log_path
from sol01.schema.index import CACHE_PATH, build_index_cache
from sol01.schema.retrieval_eval import (
    DEFAULT_GOLD_TABLE_PATH,
    load_gold_tables,
    load_retrieval_eval_task_rows,
    run_retrieval_eval,
    write_retrieval_eval_report,
)
from sol01.schema.retrieval_index import (
    DEFAULT_RETRIEVAL_INDEX_CACHE_ROOT,
    prewarm_retrieval_indexes,
)

app = typer.Typer(
    help="Snowflake Spider2-snow solver.",
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
    handle_run_kwargs: dict[str, Any] = {
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
        handle_run_kwargs["concurrency"] = concurrency
    results = handle_run(**handle_run_kwargs)
    typer.echo(f"Completed {len(results['results'])} task(s).")
    typer.echo(
        "Eval summary: "
        f"{results['eval_summary']['correct_tasks']}/"
        f"{results['eval_summary']['attempted_tasks']} correct, "
        f"missing CSV {results['eval_summary']['missing_csv_count']}"
    )
    if len(results["tasks"]) == 1:
        typer.echo(f"Exec time: {format_duration(perf_counter() - started_at)}")
    for line in _run_eval_lines(
        tasks=results["tasks"],
        answers=results["results"],
        eval_summary=results["eval_summary"],
    ):
        typer.echo(line)


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


@app.command("llm-calls")
def llm_calls(
    run_id: Annotated[
        str,
        typer.Option(help="Existing run ID to inspect."),
    ],
    instance_id: Annotated[
        str,
        typer.Option(help="Inspect one exact instance within the run."),
    ],
    call_id: Annotated[
        str | None,
        typer.Option(help="Show full details for one call ID."),
    ] = None,
    all_calls: Annotated[
        bool,
        typer.Option("--all", help="Show full details for every call."),
    ] = False,
) -> None:
    """Inspect the local LLM call log for one task."""

    if call_id is not None and all_calls:
        raise typer.BadParameter("Use either --call-id or --all, not both.")

    log_path = resolve_llm_call_log_path(
        {"run_id": run_id, "instance_id": instance_id},
        outputs_root=OUTPUTS_ROOT,
    )
    log = load_llm_call_log(log_path)
    if not log.records:
        typer.echo(f"No usable LLM call rows were found for {run_id}/{instance_id} at {log_path}.")
        raise typer.Exit(code=1)

    if log.errors:
        typer.secho(
            f"Skipped {len(log.errors)} corrupted LLM call row(s) while loading {log_path}.",
            err=True,
            fg=typer.colors.YELLOW,
        )

    summary_rows = build_llm_call_summary_rows(log)
    if call_id is None and not all_calls:
        _echo_llm_call_summary(summary_rows)
        return

    selected_records = log.records
    if call_id is not None:
        selected_records = [record for record in log.records if record.call_id == call_id]
        if not selected_records:
            typer.echo(f"No call with ID {call_id} was found in {log_path}.")
            raise typer.Exit(code=1)

    for record in selected_records:
        _echo_llm_call_detail(record)


def _echo_llm_call_summary(summary_rows: list[dict[str, Any]]) -> None:
    """Print one compact summary line per LLM call."""

    for row in summary_rows:
        attempts = int(row["attempts"])
        error_state = str(row["error_state"] or "").strip()
        error_text = error_state if error_state and error_state != "—" else "no error"
        typer.echo(
            "- "
            f"{row['sequence']}: {row['prompt_name']} | {row['status']} | "
            f"{row['duration']} | {row['model']} | "
            f"{attempts} attempt{'s' if attempts != 1 else ''} | {error_text}"
        )


def _echo_llm_call_detail(record: Any) -> None:
    """Print the full request and response payload for one LLM call."""

    summary = build_llm_call_summary_row(record)
    sections = build_llm_call_detail_sections(record)
    typer.echo(
        f"Call {summary['sequence']}: {summary['prompt_name']} | {summary['status']} | "
        f"{summary['duration']} | {summary['model']}"
    )
    typer.echo(f"Call ID: {summary['call_id']}")
    typer.echo(f"Attempts: {summary['attempts']}")
    typer.echo("")
    typer.echo("System prompt:")
    typer.echo(format_llm_call_value(sections["system_prompt"]))
    typer.echo("")
    typer.echo("User prompt:")
    typer.echo(format_llm_call_value(sections["user_prompt"]))
    typer.echo("")
    typer.echo("Output schema:")
    typer.echo(format_llm_call_value(sections["output_schema"]))
    typer.echo("")
    typer.echo("Validated response:")
    typer.echo(format_llm_call_value(sections["validated_output"]))
    typer.echo("")
    typer.echo("Attempts:")
    typer.echo(format_llm_call_value(sections["attempts"]))
    typer.echo("")
    typer.echo("Error:")
    typer.echo(format_llm_call_value(sections["error"]))
    typer.echo("")


@app.command("prewarm-schema-index")
def prewarm_schema_index_command(
    dbs: Annotated[
        list[str],
        typer.Argument(help="Database names to prewarm, for example E_COMMERCE."),
    ],
) -> None:
    """Build retrieval index cache artifacts before running batch workers."""

    indexes = _prewarm_schema_retrieval_indexes(dbs)
    typer.echo(
        f"Prewarmed {len(indexes)} schema retrieval index(es) into "
        f"{DEFAULT_RETRIEVAL_INDEX_CACHE_ROOT}"
    )


@app.command("retrieval-eval")
def retrieval_eval_command(
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
        typer.Option(min=1, help="Override the retrieval object cutoff."),
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
            help="Previous retrieval-eval report.json or tasks.jsonl for recall regression checks.",
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
        typer.Option(help="Persist report artifacts under outputs/<output_id>/retrieval_eval."),
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
    """Evaluate schema retrieval coverage against offline gold-table labels."""

    report = handle_retrieval_eval(
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
        output_dir = write_retrieval_eval_report(
            report,
            OUTPUTS_ROOT / output_id / "retrieval_eval",
        )
    if json_output:
        payload = report.payload()
        if output_dir is not None:
            payload["output_dir"] = str(output_dir)
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    typer.echo(f"Evaluated {report.task_count} task(s) at object cutoff {report.object_cutoff}.")
    typer.echo(f"Covered-schema tasks: {report.covered_task_count}")
    typer.echo(f"Pre-resolver gold recall: {_format_rate(report.pre_resolver_gold_recall)}")
    typer.echo(f"Pre-resolver any-gold recall: {_format_rate(report.pre_resolver_any_gold_recall)}")
    typer.echo(f"Post-resolver gold recall: {_format_rate(report.post_resolver_gold_recall)}")
    typer.echo(
        f"Post-resolver all-gold recall: {_format_rate(report.post_resolver_all_gold_recall)}"
    )
    if report.family_expansion_success is None:
        typer.echo("Family expansion success: n/a")
    else:
        typer.echo(f"Family expansion success: {_format_rate(report.family_expansion_success)}")
    typer.echo(f"Average prompt reduction: {_format_rate(report.average_prompt_reduction)}")
    typer.echo(f"Recall regressions: {len(report.recall_regressions)}")
    typer.echo(f"Prompt-size wins: {len(report.prompt_size_wins)}")
    typer.echo(f"Hallucinated-column failures: {len(report.hallucinated_column_failures)}")
    if report.failures:
        typer.echo("Missing gold tables:")
        for failure in report.failures[:5]:
            typer.echo(f"- {failure['instance_id']}: {', '.join(failure['missing_gold_tables'])}")
    if output_dir is not None:
        typer.echo(f"Report output: {output_dir}")


@app.command()
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
    configure_logfire()
    logger.info("logging configured", log_level=log_level)


def handle_index() -> dict[str, Any]:
    """Build the schema cache and return the in-memory payload."""

    return build_index_cache()


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
) -> dict[str, Any]:
    """Load tasks, then pass them to the batch coordinator."""

    tasks = _load_run_tasks(
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
    effective_run_id = run_id or _default_run_id("run")
    logger.info(
        "run start",
        run_id=effective_run_id,
        task_count=len(tasks),
        **_runtime_config_summary(config),
    )
    results = run_tasks(
        tasks,
        run_id=effective_run_id,
        config=config,
        force=force,
        skip_failed=skip_failed,
    )
    eval_summary = run_persisted_eval(
        effective_run_id,
        expected_instance_ids=[task.instance_id for task in tasks],
    )
    return {
        "tasks": tasks,
        "results": results,
        "eval_summary": eval_summary,
    }


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
        artifact_tag = _filtered_eval_tag(
            instance_id=instance_id,
            db=db,
            question_contains=question_contains,
            limit=limit,
        )
        run_paths = ensure_run_paths(run_id)
        staged_dir = _stage_filtered_eval_results(
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


def handle_analyze(*, run_id: str) -> dict[str, Any]:
    """Run the local trace analyzer for one run."""

    return analyze_run(run_id)


def handle_retrieval_eval(
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
    """Run offline retrieval evaluation for selected tasks."""

    tasks = _load_run_tasks(
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

    config = SchemaRetrievalConfig.from_env(dotenv_path=DEFAULT_DOTENV_PATH)
    if object_cutoff is not None:
        config = config.model_copy(update={"object_top_k": object_cutoff})
    baseline_tasks = load_retrieval_eval_task_rows(baseline_path) if baseline_path else {}
    trace_dirs = [ensure_run_paths(run_id).traces_dir for run_id in trace_run_ids or []]
    return run_retrieval_eval(
        tasks,
        gold_tables_by_instance=gold_tables,
        config=config,
        covered_only=covered_only,
        baseline_tasks=baseline_tasks,
        trace_dirs=trace_dirs,
    )


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


def _prewarm_schema_retrieval_indexes(dbs: Iterable[str]) -> list[Any]:
    """Build schema retrieval indexes once before concurrent solver work begins."""

    config = SchemaRetrievalConfig.from_env(dotenv_path=DEFAULT_DOTENV_PATH)
    return prewarm_retrieval_indexes(dbs, config=config)


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


def _load_run_tasks(
    *,
    selectors: list[str] | None,
    instance_id: str | None,
    db: str | None,
    question_contains: str | None,
    limit: int | None,
) -> list[Task]:
    """Load tasks from either explicit selectors or the legacy filter options."""

    normalized_selectors = [selector.strip() for selector in selectors or [] if selector.strip()]
    if not normalized_selectors:
        return _load_filtered_tasks(
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


def _run_eval_lines(
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
        question = _question_preview(task.question, max_length=90)
        lines.append(f"- {task.instance_id}: {eval_label} | task {task_status} | {question}")
    return lines


def _move_if_exists(source: Any, destination: Any) -> None:
    """Move one artifact into the ask layout when the source file exists."""

    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)


def _slug(value: str) -> str:
    """Turn a short filter value into a filesystem-friendly label."""

    return "".join(char if char.isalnum() else "-" for char in value).strip("-") or "value"


def _question_preview(question: str, *, max_length: int = 90) -> str:
    """Shorten long questions so CLI summaries stay readable."""

    normalized = " ".join(question.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1].rstrip() + "…"


def _format_rate(value: float) -> str:
    """Render a metric as both percentage and decimal for CLI output."""

    return f"{value:.1%} ({value:.3f})"


def _stage_filtered_eval_results(
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
