"""Coordinate retrieval, LLM calls, validation, execution, and trace writing."""

from __future__ import annotations

import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from sol01.candidate_evaluator import _dataframe_records, evaluate_candidate
from sol01.candidate_scoring import _best_attempt
from sol01.candidate_verification import (
    _aggregate_grain_guidance,
    _aggregate_verification_reason,
    _augment_intent_with_value_groundings,
    _metric_source_guidance,
    _table_schemas_for_selection,
)
from sol01.config import RuntimeConfig
from sol01.index import CACHE_PATH
from sol01.llm_logging import LLMCallLogger
from sol01.logging import get_logger
from sol01.models import (
    CandidateComparisonReport,
    ConfidenceReport,
    ExecutionResult,
    FinalAnswer,
    Intent,
    SchemaSelection,
    SQLCandidate,
    Task,
)
from sol01.output import (
    OUTPUTS_ROOT,
    RunPaths,
    csv_path_for,
    ensure_run_paths,
    llm_call_log_path_for,
    should_skip_task,
    trace_path_for,
    write_manifest,
    write_sql,
    write_trace,
)
from sol01.prompt_builders import (
    _aggregate_repair_prompt,
    _aggregate_verification_prompt,
    _candidate_comparison_prompt,
    _comparison_attempt_summary,
    _critic_prompt,
    _intent_user_prompt,
    _question_preview,
    _semantic_repair_prompt,
    _sql_generation_prompt,
    _sql_reference_context,
    _sql_repair_prompt,
)
from sol01.retrieval import load_db_index, retrieve_schema

logger = get_logger(__name__)
LLMClient: Any | None = None

__all__ = ["run_task", "run_tasks"]


def load_document_text(file_name: str) -> str:
    """Load task-linked document text without importing the docs module at startup."""

    from sol01.docs import load_document_text as _load_document_text

    return _load_document_text(file_name)


def _llm_client_class() -> Any:
    """Return the live LLM client class, importing it only when needed."""

    global LLMClient
    if LLMClient is None:
        from sol01.llm import LLMClient as _LLMClient

        LLMClient = _LLMClient
    return LLMClient


class StructuredLLM(Protocol):
    """Minimal LLM interface the coordinator needs for structured calls."""

    def load_prompt(self, prompt_name: str) -> Any: ...

    def run_structured(
        self,
        user_prompt: str,
        *,
        prompt_name: str,
        output_type: type[Any],
        model: Any = None,
    ) -> Any: ...

    def run_structured_with_prompt(
        self,
        user_prompt: str,
        *,
        prompt: Any,
        output_type: type[Any],
        model: Any = None,
    ) -> Any: ...


def run_tasks(
    tasks: list[Task],
    *,
    run_id: str,
    config: RuntimeConfig,
    llm_client: StructuredLLM | None = None,
    outputs_root: Path | None = None,
    force: bool = False,
    skip_failed: bool = False,
) -> list[FinalAnswer]:
    """Run a batch of tasks and write a manifest before processing them."""

    run_paths = ensure_run_paths(run_id, outputs_root=outputs_root or OUTPUTS_ROOT)
    write_manifest(
        run_paths,
        manifest={
            "task_ids": [task.instance_id for task in tasks],
            "model": config.model,
            "provider_routing": config.provider_routing,
            "config": {
                "concurrency": config.concurrency,
            },
        },
    )
    logger.info(
        "run start",
        run_id=run_id,
        task_count=len(tasks),
        model=getattr(config, "model", None),
        provider_routing=getattr(config, "provider_routing", None),
        concurrency=getattr(config, "concurrency", None),
        force=force,
        skip_failed=skip_failed,
    )

    _prewarm_schema_indexes(tasks)
    results = _run_task_batch(
        tasks,
        run_paths=run_paths,
        config=config,
        llm_client=llm_client,
        force=force,
        skip_failed=skip_failed,
    )
    logger.info(
        "run complete",
        run_id=run_id,
        success_count=sum(1 for result in results if result.status == "success"),
        failed_count=sum(1 for result in results if result.status == "failed"),
        skipped_count=sum(1 for result in results if result.status == "skipped"),
    )
    return results


def _prewarm_schema_indexes(tasks: list[Task], *, cache_path: Path = CACHE_PATH) -> None:
    """Load each selected database into the local schema cache before workers start."""

    seen: set[str] = set()
    for task in tasks:
        if task.db in seen:
            continue
        seen.add(task.db)
        load_db_index(task.db, cache_path=cache_path)


def _run_task_batch(
    tasks: list[Task],
    *,
    run_paths: RunPaths,
    config: RuntimeConfig,
    llm_client: StructuredLLM | None,
    force: bool,
    skip_failed: bool,
) -> list[FinalAnswer]:
    """Run tasks sequentially or with bounded concurrency, preserving input order."""

    if config.concurrency <= 1 or llm_client is not None:
        return [
            _run_single_batch_task(
                task,
                run_paths=run_paths,
                config=config,
                llm_client=llm_client,
                force=force,
                skip_failed=skip_failed,
            )
            for task in tasks
        ]

    results: list[FinalAnswer | None] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        future_to_index = {
            executor.submit(
                _run_single_batch_task,
                task,
                run_paths=run_paths,
                config=config,
                llm_client=None,
                force=force,
                skip_failed=skip_failed,
            ): index
            for index, task in enumerate(tasks)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
    return [result for result in results if result is not None]


def _run_single_batch_task(
    task: Task,
    *,
    run_paths: RunPaths,
    config: RuntimeConfig,
    llm_client: StructuredLLM | None,
    force: bool,
    skip_failed: bool,
) -> FinalAnswer:
    """Run one task and convert unexpected exceptions into failed traces."""

    try:
        return run_task(
            task,
            run_paths=run_paths,
            config=config,
            llm_client=llm_client,
            force=force,
            skip_failed=skip_failed,
        )
    except Exception as exc:
        return _record_batch_task_failure(
            task=task,
            run_paths=run_paths,
            live_logging_enabled=llm_client is None,
            error=exc,
        )


def _record_batch_task_failure(
    *,
    task: Task,
    run_paths: RunPaths,
    live_logging_enabled: bool,
    error: Exception,
) -> FinalAnswer:
    """Write a failed trace for an unexpected batch-level task crash."""

    task_trace_path = trace_path_for(run_paths, instance_id=task.instance_id)
    task_llm_log_path = llm_call_log_path_for(run_paths, instance_id=task.instance_id)
    trace_payload: dict[str, Any] = {
        "instance_id": task.instance_id,
        "db": task.db,
        "question": task.question,
        "status": "failed",
        "prompt_hashes": {},
        "attempts": [],
        "final_sql": None,
        "csv_path": None,
        "error": {
            "type": error.__class__.__name__,
            "message": str(error) or error.__class__.__name__,
        },
    }
    if live_logging_enabled:
        trace_payload["llm_call_log_path"] = str(task_llm_log_path)
    write_trace(run_paths, instance_id=task.instance_id, trace=trace_payload)
    logger.exception(
        "task crashed",
        instance_id=task.instance_id,
        db=task.db,
        error_type=error.__class__.__name__,
        error_message=str(error) or error.__class__.__name__,
        trace_path=str(task_trace_path),
    )
    return FinalAnswer(
        instance_id=task.instance_id,
        status="failed",
        sql=None,
        csv_path=None,
        trace_path=str(task_trace_path),
    )


# ---------------------------------------------------------------------------
# Per-task pipeline stages
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _TaskCtx:
    """Shared context threaded through each pipeline stage of run_task."""

    task: Task
    client: StructuredLLM
    intent: Intent
    schema: SchemaSelection
    table_schemas: dict[str, Any]
    sql_reference_context: str
    docs_context: str
    aggregate_grain_guidance: str | None
    metric_source_guidance: str | None
    prompt_hashes: dict[str, str]


def _check_skip(
    task: Task,
    run_paths: RunPaths,
    task_trace_path: Path,
    *,
    force: bool,
    skip_failed: bool,
) -> FinalAnswer | None:
    """Return an existing FinalAnswer if the task should be skipped, else None."""

    if force or not should_skip_task(
        run_paths, instance_id=task.instance_id, skip_failed=skip_failed
    ):
        return None
    existing_trace = json.loads(task_trace_path.read_text(encoding="utf-8"))
    logger.info(
        "task skipped",
        instance_id=task.instance_id,
        db=task.db,
        run_root=str(run_paths.root),
        trace_path=str(task_trace_path),
        csv_path=existing_trace.get("csv_path"),
    )
    return FinalAnswer(
        instance_id=task.instance_id,
        status="skipped",
        sql=existing_trace.get("final_sql"),
        csv_path=existing_trace.get("csv_path"),
        trace_path=str(task_trace_path),
    )


def _build_context(
    task: Task,
    client: StructuredLLM,
    prompt_hashes: dict[str, str],
    run_paths: RunPaths,
) -> _TaskCtx:
    """Retrieve schema, extract intent, and assemble the shared pipeline context."""

    logger.info(
        "task start",
        instance_id=task.instance_id,
        db=task.db,
        question_preview=_question_preview(task.question),
        question_length=len(task.question),
        run_root=str(run_paths.root),
    )
    schema = retrieve_schema(task.question, task.db, llm_client=client)
    logger.info(
        "schema selected",
        instance_id=task.instance_id,
        selected_count=len(schema.selected_tables),
        expanded_count=len(schema.expanded_tables),
        selected_tables=schema.selected_tables,
        expanded_tables=schema.expanded_tables,
        confidence=schema.confidence,
    )
    docs_context = (
        load_document_text(task.external_knowledge)
        if task.external_knowledge
        else "No task-linked document context."
    )
    table_schemas = _table_schemas_for_selection(schema)
    intent = _run_prompt(
        client,
        prompt_hashes=prompt_hashes,
        prompt_name="intent",
        output_type=Intent,
        user_prompt=_intent_user_prompt(task, schema, docs_context, table_schemas),
    )
    intent = _augment_intent_with_value_groundings(
        intent, task=task, schema=schema, table_schemas=table_schemas
    )
    logger.info(
        "intent extracted",
        instance_id=task.instance_id,
        metrics=intent.metrics,
        filters=intent.filters,
        time_constraints=intent.time_constraints,
    )
    return _TaskCtx(
        task=task,
        client=client,
        intent=intent,
        schema=schema,
        table_schemas=table_schemas,
        sql_reference_context=_sql_reference_context(schema, table_schemas),
        docs_context=docs_context,
        aggregate_grain_guidance=_aggregate_grain_guidance(task, intent, schema, table_schemas),
        metric_source_guidance=_metric_source_guidance(task, intent, table_schemas),
        prompt_hashes=prompt_hashes,
    )


def _log_candidate(instance_id: str, attempt: dict[str, Any]) -> None:
    """Log a standard candidate-evaluated event."""

    logger.info(
        "candidate evaluated",
        instance_id=instance_id,
        stage=attempt["stage"],
        validation_ok=attempt["validation"]["ok"],
        execution_ok=attempt["execution_result"]["ok"],
        score=attempt["score"],
        elapsed_seconds=attempt["elapsed_seconds"],
        row_count=attempt["execution_result"]["row_count"],
    )


def _generate_initial_candidates(
    ctx: _TaskCtx,
    attempts: list[dict[str, Any]],
    *,
    initial_candidates: int,
    max_attempts: int,
) -> dict[str, Any] | None:
    """Generate up to initial_candidates SQL candidates; appends to attempts in place."""

    logger.info(
        "generating candidates",
        instance_id=ctx.task.instance_id,
        initial_candidates=initial_candidates,
        max_attempts=max_attempts,
    )
    for candidate_index in range(initial_candidates):
        if len(attempts) >= max_attempts:
            break
        stage = f"initial_{candidate_index + 1}"
        logger.info(
            "candidate request",
            instance_id=ctx.task.instance_id,
            stage=stage,
            prompt_name="sql_generation",
        )
        candidate = _run_prompt(
            ctx.client,
            prompt_hashes=ctx.prompt_hashes,
            prompt_name="sql_generation",
            output_type=SQLCandidate,
            user_prompt=_sql_generation_prompt(
                ctx.task,
                ctx.intent,
                ctx.sql_reference_context,
                ctx.docs_context,
                aggregate_grain_guidance=ctx.aggregate_grain_guidance,
                metric_source_guidance=ctx.metric_source_guidance,
            ),
        )
        attempt = evaluate_candidate(
            task=ctx.task,
            candidate=candidate,
            intent=ctx.intent,
            schema=ctx.schema,
            table_schemas=ctx.table_schemas,
            stage=stage,
        )
        attempts.append(attempt)
        _log_candidate(ctx.task.instance_id, attempt)
    return _best_attempt(attempts)


def _repair_failed_execution(
    ctx: _TaskCtx,
    attempts: list[dict[str, Any]],
    best_attempt: dict[str, Any] | None,
    *,
    max_attempts: int,
) -> dict[str, Any] | None:
    """Repair the best attempt if it failed execution; appends to attempts in place."""

    if (
        best_attempt is None
        or best_attempt["execution_result"]["ok"]
        or len(attempts) >= max_attempts
    ):
        return best_attempt

    logger.info(
        "repair requested",
        instance_id=ctx.task.instance_id,
        stage="repair",
        best_stage=best_attempt["stage"],
    )
    repaired_candidate = _run_prompt(
        ctx.client,
        prompt_hashes=ctx.prompt_hashes,
        prompt_name="sql_repair",
        output_type=SQLCandidate,
        user_prompt=_sql_repair_prompt(
            ctx.task,
            ctx.intent,
            best_attempt,
            ctx.sql_reference_context,
            ctx.docs_context,
            aggregate_grain_guidance=ctx.aggregate_grain_guidance,
            metric_source_guidance=ctx.metric_source_guidance,
        ),
    )
    attempt = evaluate_candidate(
        task=ctx.task,
        candidate=repaired_candidate,
        intent=ctx.intent,
        schema=ctx.schema,
        table_schemas=ctx.table_schemas,
        stage="repair",
    )
    attempts.append(attempt)
    _log_candidate(ctx.task.instance_id, attempt)
    return _best_attempt(attempts)


def _compare_candidates(
    ctx: _TaskCtx,
    attempts: list[dict[str, Any]],
    best_attempt: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Compare all executable candidates and return (updated_best, comparison_payload)."""

    executable_attempts = [a for a in attempts if a["execution_result"]["ok"]]
    if len(executable_attempts) <= 1:
        return best_attempt, None

    logger.info(
        "candidate comparison requested",
        instance_id=ctx.task.instance_id,
        baseline_stage=best_attempt["stage"] if best_attempt is not None else None,
        executable_attempts=len(executable_attempts),
    )
    comparison = _run_prompt(
        ctx.client,
        prompt_hashes=ctx.prompt_hashes,
        prompt_name="result_comparison",
        output_type=CandidateComparisonReport,
        user_prompt=_candidate_comparison_prompt(
            ctx.task,
            ctx.intent,
            executable_attempts,
            ctx.sql_reference_context,
            ctx.docs_context,
            aggregate_grain_guidance=ctx.aggregate_grain_guidance,
            metric_source_guidance=ctx.metric_source_guidance,
            baseline_stage=best_attempt["stage"] if best_attempt is not None else None,
        ),
    )
    comparison_payload = {
        **comparison.model_dump(mode="json"),
        "candidates": [_comparison_attempt_summary(a) for a in executable_attempts],
    }
    logger.info(
        "candidate comparison reviewed",
        instance_id=ctx.task.instance_id,
        baseline_stage=comparison.baseline_stage,
        preferred_stage=comparison.preferred_stage,
        compared_stages=comparison.compared_stages,
        reasons=comparison.reasons,
    )
    if comparison.preferred_stage:
        preferred_attempt = next(
            (a for a in executable_attempts if a["stage"] == comparison.preferred_stage),
            None,
        )
        if preferred_attempt is not None:
            best_attempt = preferred_attempt
    return best_attempt, comparison_payload


def _verify_aggregates(
    ctx: _TaskCtx,
    attempts: list[dict[str, Any]],
    best_attempt: dict[str, Any] | None,
    *,
    max_attempts: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Verify aggregate result quality and repair if needed; appends to attempts in place.

    Returns (updated_best_attempt, verification_payload).
    Also sets best_attempt["aggregate_verification"] in place when verification runs.
    """

    if (
        best_attempt is None
        or not best_attempt["execution_result"]["ok"]
        or len(attempts) >= max_attempts
    ):
        return best_attempt, None

    verification_reason = _aggregate_verification_reason(best_attempt)
    if verification_reason is None:
        return best_attempt, None

    logger.info(
        "aggregate verification requested",
        instance_id=ctx.task.instance_id,
        stage=best_attempt["stage"],
        reason=verification_reason,
    )
    aggregate_verification = _run_prompt(
        ctx.client,
        prompt_hashes=ctx.prompt_hashes,
        prompt_name="aggregate_verification",
        output_type=ConfidenceReport,
        user_prompt=_aggregate_verification_prompt(
            ctx.task,
            best_attempt,
            ctx.sql_reference_context,
            ctx.docs_context,
            reason=verification_reason,
        ),
    )
    verification_payload: dict[str, Any] = {
        "reason": verification_reason,
        **aggregate_verification.model_dump(mode="json"),
    }
    best_attempt["aggregate_verification"] = verification_payload
    logger.info(
        "aggregate verification reviewed",
        instance_id=ctx.task.instance_id,
        should_repair=aggregate_verification.should_repair,
        confidence=aggregate_verification.confidence,
        issues=aggregate_verification.issues,
        repair_focus=aggregate_verification.repair_focus,
    )

    if not aggregate_verification.should_repair:
        return best_attempt, verification_payload

    logger.info(
        "aggregate repair requested",
        instance_id=ctx.task.instance_id,
        stage="aggregate_repair",
        focus=aggregate_verification.repair_focus,
    )
    repaired_candidate = _run_prompt(
        ctx.client,
        prompt_hashes=ctx.prompt_hashes,
        prompt_name="sql_repair",
        output_type=SQLCandidate,
        user_prompt=_aggregate_repair_prompt(
            ctx.task,
            best_attempt,
            aggregate_verification,
            ctx.sql_reference_context,
            ctx.docs_context,
            intent=ctx.intent,
        ),
    )
    attempt = evaluate_candidate(
        task=ctx.task,
        candidate=repaired_candidate,
        intent=ctx.intent,
        schema=ctx.schema,
        table_schemas=ctx.table_schemas,
        stage="aggregate_repair",
    )
    attempts.append(attempt)
    _log_candidate(ctx.task.instance_id, attempt)
    if attempt["execution_result"]["ok"]:
        best_attempt = attempt
    else:
        best_attempt = _best_attempt(attempts)
    return best_attempt, verification_payload


def _apply_critic_repair(
    ctx: _TaskCtx,
    attempts: list[dict[str, Any]],
    best_attempt: dict[str, Any] | None,
    *,
    max_attempts: int,
    semantic_repairs: int,
) -> dict[str, Any] | None:
    """Run the critic and apply a semantic repair if needed; appends to attempts in place.

    Also sets best_attempt["critic"] in place when the critic runs.
    semantic_repairs acts as a budget: 0 skips the critic entirely, 1 (default) runs it once.
    """

    if (
        best_attempt is None
        or not best_attempt["execution_result"]["ok"]
        or semantic_repairs < 1
        or len(attempts) >= max_attempts
    ):
        return best_attempt

    logger.info(
        "critic requested",
        instance_id=ctx.task.instance_id,
        stage="critic",
        best_stage=best_attempt["stage"],
    )
    critic = _run_prompt(
        ctx.client,
        prompt_hashes=ctx.prompt_hashes,
        prompt_name="result_critic",
        output_type=ConfidenceReport,
        user_prompt=_critic_prompt(
            ctx.task,
            ctx.intent,
            best_attempt,
            ctx.sql_reference_context,
            ctx.docs_context,
            metric_source_guidance=ctx.metric_source_guidance,
        ),
    )
    best_attempt["critic"] = critic.model_dump(mode="json")
    logger.info(
        "critic reviewed",
        instance_id=ctx.task.instance_id,
        should_repair=critic.should_repair,
        confidence=critic.confidence,
        issues=critic.issues,
    )

    if not critic.should_repair:
        return best_attempt

    logger.info(
        "semantic repair requested",
        instance_id=ctx.task.instance_id,
        stage="critic_repair",
        focus=critic.repair_focus,
    )
    repaired_candidate = _run_prompt(
        ctx.client,
        prompt_hashes=ctx.prompt_hashes,
        prompt_name="sql_repair",
        output_type=SQLCandidate,
        user_prompt=_semantic_repair_prompt(
            ctx.task,
            ctx.intent,
            best_attempt,
            critic,
            ctx.sql_reference_context,
            ctx.docs_context,
            metric_source_guidance=ctx.metric_source_guidance,
        ),
    )
    attempt = evaluate_candidate(
        task=ctx.task,
        candidate=repaired_candidate,
        intent=ctx.intent,
        schema=ctx.schema,
        table_schemas=ctx.table_schemas,
        stage="critic_repair",
    )
    attempts.append(attempt)
    _log_candidate(ctx.task.instance_id, attempt)
    if attempt["execution_result"]["ok"]:
        return attempt
    return _best_attempt(attempts)


def _write_task_output(
    ctx: _TaskCtx,
    attempts: list[dict[str, Any]],
    best_attempt: dict[str, Any] | None,
    run_paths: RunPaths,
    task_trace_path: Path,
    task_llm_log_path: Path,
    *,
    candidate_comparison_payload: dict[str, Any] | None,
    aggregate_verification_payload: dict[str, Any] | None,
    live_logging_enabled: bool,
    started_at: float,
) -> FinalAnswer:
    """Write final SQL, CSV, and trace; return the FinalAnswer."""

    task = ctx.task
    trace_payload: dict[str, Any] = {
        "instance_id": task.instance_id,
        "db": task.db,
        "question": task.question,
        "retrieval_mode": ctx.schema.retrieval_mode,
        "schema_selection": ctx.schema.model_dump(mode="json"),
        "intent": ctx.intent.model_dump(mode="json"),
        "prompt_hashes": ctx.prompt_hashes,
        "attempts": attempts,
    }
    if candidate_comparison_payload is not None:
        trace_payload["candidate_comparison"] = candidate_comparison_payload
    if aggregate_verification_payload is not None:
        trace_payload["aggregate_verification"] = aggregate_verification_payload
    if live_logging_enabled:
        trace_payload["llm_call_log_path"] = str(task_llm_log_path)

    if best_attempt is not None and best_attempt["execution_result"]["ok"]:
        sql_path = write_sql(run_paths, instance_id=task.instance_id, sql=best_attempt["sql"])
        csv_path = csv_path_for(run_paths, instance_id=task.instance_id)
        best_dataframe = best_attempt["_dataframe"]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        best_dataframe.to_csv(csv_path, index=False)
        final_execution = ExecutionResult(
            ok=True,
            row_count=len(best_dataframe),
            columns=[str(column) for column in best_dataframe.columns],
            sample_rows=_dataframe_records(best_dataframe.head(3)),
            csv_path=str(csv_path),
            error=None,
        )
        trace_payload.update(
            {
                "status": "success",
                "final_sql": best_attempt["sql"],
                "sql_path": str(sql_path),
                "csv_path": str(csv_path),
                "final_execution": final_execution.model_dump(mode="json"),
            }
        )
        trace_payload["attempts"] = [_trace_attempt(a) for a in attempts]
        write_trace(run_paths, instance_id=task.instance_id, trace=trace_payload)
        elapsed_seconds = round(perf_counter() - started_at, 3)
        logger.info(
            "task complete",
            instance_id=task.instance_id,
            status="success",
            run_root=str(run_paths.root),
            attempts=len(attempts),
            best_stage=best_attempt["stage"],
            best_score=best_attempt["score"],
            row_count=len(best_dataframe),
            columns=[str(column) for column in best_dataframe.columns],
            elapsed_seconds=elapsed_seconds,
            sql_path=str(sql_path),
            csv_path=str(csv_path),
        )
        return FinalAnswer(
            instance_id=task.instance_id,
            status="success",
            sql=best_attempt["sql"],
            csv_path=str(csv_path),
            trace_path=str(task_trace_path),
        )

    trace_payload.update(
        {
            "status": "failed",
            "final_sql": best_attempt["sql"] if best_attempt is not None else None,
            "csv_path": None,
        }
    )
    trace_payload["attempts"] = [_trace_attempt(a) for a in attempts]
    write_trace(run_paths, instance_id=task.instance_id, trace=trace_payload)
    elapsed_seconds = round(perf_counter() - started_at, 3)
    logger.warning(
        "task complete",
        instance_id=task.instance_id,
        status="failed",
        run_root=str(run_paths.root),
        attempts=len(attempts),
        best_stage=best_attempt["stage"] if best_attempt is not None else None,
        best_score=best_attempt["score"] if best_attempt is not None else None,
        row_count=best_attempt["execution_result"]["row_count"] if best_attempt is not None else 0,
        elapsed_seconds=elapsed_seconds,
    )
    return FinalAnswer(
        instance_id=task.instance_id,
        status="failed",
        sql=best_attempt["sql"] if best_attempt is not None else None,
        csv_path=None,
        trace_path=str(task_trace_path),
    )


def run_task(
    task: Task,
    *,
    run_paths: RunPaths,
    config: RuntimeConfig,
    llm_client: StructuredLLM | None = None,
    force: bool = False,
    skip_failed: bool = False,
    initial_candidates: int = 3,
    max_attempts: int = 4,
    semantic_repairs: int = 1,
) -> FinalAnswer:
    """Run one task from retrieval through final trace writing."""

    started_at = perf_counter()
    live_logging_enabled = llm_client is None
    task_llm_log_path = llm_call_log_path_for(run_paths, instance_id=task.instance_id)
    client = llm_client or _llm_client_class()(
        config,
        call_logger=LLMCallLogger(task_llm_log_path),
    )
    task_trace_path = trace_path_for(run_paths, instance_id=task.instance_id)

    skipped = _check_skip(task, run_paths, task_trace_path, force=force, skip_failed=skip_failed)
    if skipped is not None:
        return skipped

    prompt_hashes: dict[str, str] = {}
    attempts: list[dict[str, Any]] = []

    ctx = _build_context(task, client, prompt_hashes, run_paths)

    best_attempt = _generate_initial_candidates(
        ctx, attempts, initial_candidates=initial_candidates, max_attempts=max_attempts
    )
    best_attempt = _repair_failed_execution(ctx, attempts, best_attempt, max_attempts=max_attempts)
    best_attempt, candidate_comparison_payload = _compare_candidates(ctx, attempts, best_attempt)
    best_attempt, aggregate_verification_payload = _verify_aggregates(
        ctx, attempts, best_attempt, max_attempts=max_attempts
    )
    best_attempt = _apply_critic_repair(
        ctx, attempts, best_attempt, max_attempts=max_attempts, semantic_repairs=semantic_repairs
    )

    return _write_task_output(
        ctx,
        attempts,
        best_attempt,
        run_paths,
        task_trace_path,
        task_llm_log_path,
        candidate_comparison_payload=candidate_comparison_payload,
        aggregate_verification_payload=aggregate_verification_payload,
        live_logging_enabled=live_logging_enabled,
        started_at=started_at,
    )


def _run_prompt(
    client: StructuredLLM,
    *,
    prompt_hashes: dict[str, str],
    prompt_name: str,
    output_type: type[Any],
    user_prompt: str,
) -> Any:
    """Load one prompt hash and run one structured LLM call."""

    prompt = client.load_prompt(prompt_name)
    prompt_hashes.setdefault(prompt_name, prompt.sha256)
    return client.run_structured_with_prompt(
        user_prompt,
        prompt=prompt,
        output_type=output_type,
    )


def _trace_attempt(attempt: dict[str, Any]) -> dict[str, Any]:
    """Drop non-serializable internal fields before writing the trace."""

    return {key: value for key, value in attempt.items() if not key.startswith("_")}
