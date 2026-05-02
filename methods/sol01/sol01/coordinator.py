"""Coordinate retrieval, LLM calls, validation, execution, and trace writing."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from numbers import Number
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from sol01.config import RuntimeConfig
from sol01.index import CACHE_PATH
from sol01.llm_logging import LLMCallLogger
from sol01.logging import get_logger
from sol01.models import (
    AggregateGrainReport,
    CandidateComparisonReport,
    ConfidenceReport,
    ExecutionResult,
    FilterGroundingReport,
    FinalAnswer,
    Intent,
    OutputShapeReport,
    SchemaSelection,
    SQLCandidate,
    TableSchema,
    Task,
    ValidationReport,
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
from sol01.retrieval import load_db_index, retrieve_schema
from sol01.validation import validate_sql

logger = get_logger(__name__)
LLMClient: Any | None = None


def load_document_text(file_name: str) -> str:
    """Load task-linked document text without importing the docs module at startup."""

    from sol01.docs import load_document_text as _load_document_text

    return _load_document_text(file_name)


def fetch_query_dataframe(sql: str, *, db: str):
    """Run one query without importing the Snowflake stack at startup."""

    from sol01.snowflake_runner import fetch_query_dataframe as _fetch_query_dataframe

    return _fetch_query_dataframe(sql, db=db)


def _dataframe_records(dataframe):
    """Convert one DataFrame slice without importing the Snowflake stack at startup."""

    from sol01.snowflake_runner import _dataframe_records as _dataframe_records_impl

    return _dataframe_records_impl(dataframe)


def profile_dataframe(dataframe):
    """Profile one DataFrame without importing pandas at startup."""

    from sol01.profiling import profile_dataframe as _profile_dataframe

    return _profile_dataframe(dataframe)


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

    if not force and should_skip_task(
        run_paths,
        instance_id=task.instance_id,
        skip_failed=skip_failed,
    ):
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

    prompt_hashes: dict[str, str] = {}
    attempts: list[dict[str, Any]] = []
    critic_repairs_used = 0

    logger.info(
        "task start",
        instance_id=task.instance_id,
        db=task.db,
        question_preview=_question_preview(task.question),
        question_length=len(task.question),
        run_root=str(run_paths.root),
    )
    schema = retrieve_schema(
        task.question,
        task.db,
        llm_client=client,
    )
    logger.info(
        "schema selected",
        instance_id=task.instance_id,
        selected_count=len(schema.selected_tables),
        expanded_count=len(schema.expanded_tables),
        selected_tables=schema.selected_tables,
        expanded_tables=schema.expanded_tables,
        confidence=schema.confidence,
    )
    if task.external_knowledge:
        docs_context = load_document_text(task.external_knowledge)
    else:
        docs_context = "No task-linked document context."

    table_schemas = _table_schemas_for_selection(schema)
    intent = _run_prompt(
        client,
        prompt_hashes=prompt_hashes,
        prompt_name="intent",
        output_type=Intent,
        user_prompt=_intent_user_prompt(task, schema, docs_context, table_schemas),
    )
    intent = _augment_intent_with_value_groundings(
        intent,
        task=task,
        schema=schema,
        table_schemas=table_schemas,
    )
    logger.info(
        "intent extracted",
        instance_id=task.instance_id,
        metrics=intent.metrics,
        filters=intent.filters,
        time_constraints=intent.time_constraints,
    )

    aggregate_grain_guidance = _aggregate_grain_guidance(task, intent, schema, table_schemas)
    metric_source_guidance = _metric_source_guidance(task, intent, table_schemas)
    sql_reference_context = _sql_reference_context(schema, table_schemas)

    logger.info(
        "generating candidates",
        instance_id=task.instance_id,
        initial_candidates=initial_candidates,
        max_attempts=max_attempts,
    )
    for candidate_index in range(initial_candidates):
        if len(attempts) >= max_attempts:
            break
        stage = f"initial_{candidate_index + 1}"
        logger.info(
            "candidate request",
            instance_id=task.instance_id,
            stage=stage,
            prompt_name="sql_generation",
        )
        candidate = _run_prompt(
            client,
            prompt_hashes=prompt_hashes,
            prompt_name="sql_generation",
            output_type=SQLCandidate,
            user_prompt=_sql_generation_prompt(
                task,
                intent,
                sql_reference_context,
                docs_context,
                aggregate_grain_guidance=aggregate_grain_guidance,
                metric_source_guidance=metric_source_guidance,
            ),
        )
        attempt = _evaluate_candidate(
            task=task,
            candidate=candidate,
            intent=intent,
            schema=schema,
            table_schemas=table_schemas,
            stage=stage,
        )
        attempts.append(attempt)
        logger.info(
            "candidate evaluated",
            instance_id=task.instance_id,
            stage=stage,
            validation_ok=attempt["validation"]["ok"],
            execution_ok=attempt["execution_result"]["ok"],
            score=attempt["score"],
            elapsed_seconds=attempt["elapsed_seconds"],
            row_count=attempt["execution_result"]["row_count"],
        )

    best_attempt = _best_attempt(attempts)

    if (
        best_attempt is not None
        and not best_attempt["execution_result"]["ok"]
        and len(attempts) < max_attempts
    ):
        logger.info(
            "repair requested",
            instance_id=task.instance_id,
            stage="repair",
            best_stage=best_attempt["stage"],
        )
        repaired_candidate = _run_prompt(
            client,
            prompt_hashes=prompt_hashes,
            prompt_name="sql_repair",
            output_type=SQLCandidate,
            user_prompt=_sql_repair_prompt(
                task,
                intent,
                best_attempt,
                sql_reference_context,
                docs_context,
                aggregate_grain_guidance=aggregate_grain_guidance,
                metric_source_guidance=metric_source_guidance,
            ),
        )
        attempt = _evaluate_candidate(
            task=task,
            candidate=repaired_candidate,
            intent=intent,
            schema=schema,
            table_schemas=table_schemas,
            stage="repair",
        )
        attempts.append(attempt)
        logger.info(
            "candidate evaluated",
            instance_id=task.instance_id,
            stage="repair",
            validation_ok=attempt["validation"]["ok"],
            execution_ok=attempt["execution_result"]["ok"],
            score=attempt["score"],
            elapsed_seconds=attempt["elapsed_seconds"],
            row_count=attempt["execution_result"]["row_count"],
        )
        best_attempt = _best_attempt(attempts)

    candidate_comparison_payload: dict[str, Any] | None = None
    aggregate_verification_payload: dict[str, Any] | None = None
    executable_attempts = [attempt for attempt in attempts if attempt["execution_result"]["ok"]]
    if len(executable_attempts) > 1:
        logger.info(
            "candidate comparison requested",
            instance_id=task.instance_id,
            baseline_stage=best_attempt["stage"] if best_attempt is not None else None,
            executable_attempts=len(executable_attempts),
        )
        comparison = _run_prompt(
            client,
            prompt_hashes=prompt_hashes,
            prompt_name="result_comparison",
            output_type=CandidateComparisonReport,
            user_prompt=_candidate_comparison_prompt(
                task,
                intent,
                executable_attempts,
                sql_reference_context,
                docs_context,
                aggregate_grain_guidance=aggregate_grain_guidance,
                metric_source_guidance=metric_source_guidance,
                baseline_stage=best_attempt["stage"] if best_attempt is not None else None,
            ),
        )
        candidate_comparison_payload = {
            **comparison.model_dump(mode="json"),
            "candidates": [_comparison_attempt_summary(attempt) for attempt in executable_attempts],
        }
        logger.info(
            "candidate comparison reviewed",
            instance_id=task.instance_id,
            baseline_stage=comparison.baseline_stage,
            preferred_stage=comparison.preferred_stage,
            compared_stages=comparison.compared_stages,
            reasons=comparison.reasons,
        )
        if comparison.preferred_stage:
            preferred_attempt = next(
                (
                    attempt
                    for attempt in executable_attempts
                    if attempt["stage"] == comparison.preferred_stage
                ),
                None,
            )
            if preferred_attempt is not None:
                best_attempt = preferred_attempt

    if (
        best_attempt is not None
        and best_attempt["execution_result"]["ok"]
        and len(attempts) < max_attempts
    ):
        verification_reason = _aggregate_verification_reason(best_attempt)
        if verification_reason is not None:
            logger.info(
                "aggregate verification requested",
                instance_id=task.instance_id,
                stage=best_attempt["stage"],
                reason=verification_reason,
            )
            aggregate_verification = _run_prompt(
                client,
                prompt_hashes=prompt_hashes,
                prompt_name="aggregate_verification",
                output_type=ConfidenceReport,
                user_prompt=_aggregate_verification_prompt(
                    task,
                    best_attempt,
                    sql_reference_context,
                    docs_context,
                    reason=verification_reason,
                ),
            )
            aggregate_verification_payload = {
                "reason": verification_reason,
                **aggregate_verification.model_dump(mode="json"),
            }
            best_attempt["aggregate_verification"] = aggregate_verification_payload
            logger.info(
                "aggregate verification reviewed",
                instance_id=task.instance_id,
                should_repair=aggregate_verification.should_repair,
                confidence=aggregate_verification.confidence,
                issues=aggregate_verification.issues,
                repair_focus=aggregate_verification.repair_focus,
            )
            if aggregate_verification.should_repair:
                logger.info(
                    "aggregate repair requested",
                    instance_id=task.instance_id,
                    stage="aggregate_repair",
                    focus=aggregate_verification.repair_focus,
                )
                repaired_candidate = _run_prompt(
                    client,
                    prompt_hashes=prompt_hashes,
                    prompt_name="sql_repair",
                    output_type=SQLCandidate,
                    user_prompt=_aggregate_repair_prompt(
                        task,
                        best_attempt,
                        aggregate_verification,
                        sql_reference_context,
                        docs_context,
                        intent=intent,
                    ),
                )
                attempt = _evaluate_candidate(
                    task=task,
                    candidate=repaired_candidate,
                    intent=intent,
                    schema=schema,
                    table_schemas=table_schemas,
                    stage="aggregate_repair",
                )
                attempts.append(attempt)
                logger.info(
                    "candidate evaluated",
                    instance_id=task.instance_id,
                    stage="aggregate_repair",
                    validation_ok=attempt["validation"]["ok"],
                    execution_ok=attempt["execution_result"]["ok"],
                    score=attempt["score"],
                    elapsed_seconds=attempt["elapsed_seconds"],
                    row_count=attempt["execution_result"]["row_count"],
                )
                if attempt["execution_result"]["ok"]:
                    best_attempt = attempt
                else:
                    best_attempt = _best_attempt(attempts)

    if (
        best_attempt is not None
        and best_attempt["execution_result"]["ok"]
        and critic_repairs_used < semantic_repairs
        and len(attempts) < max_attempts
    ):
        logger.info(
            "critic requested",
            instance_id=task.instance_id,
            stage="critic",
            best_stage=best_attempt["stage"],
        )
        critic = _run_prompt(
            client,
            prompt_hashes=prompt_hashes,
            prompt_name="result_critic",
            output_type=ConfidenceReport,
            user_prompt=_critic_prompt(
                task,
                intent,
                best_attempt,
                sql_reference_context,
                docs_context,
                metric_source_guidance=metric_source_guidance,
            ),
        )
        best_attempt["critic"] = critic.model_dump(mode="json")
        logger.info(
            "critic reviewed",
            instance_id=task.instance_id,
            should_repair=critic.should_repair,
            confidence=critic.confidence,
            issues=critic.issues,
        )
        if critic.should_repair:
            critic_repairs_used += 1
            logger.info(
                "semantic repair requested",
                instance_id=task.instance_id,
                stage="critic_repair",
                focus=critic.repair_focus,
            )
            repaired_candidate = _run_prompt(
                client,
                prompt_hashes=prompt_hashes,
                prompt_name="sql_repair",
                output_type=SQLCandidate,
                user_prompt=_semantic_repair_prompt(
                    task,
                    intent,
                    best_attempt,
                    critic,
                    sql_reference_context,
                    docs_context,
                    metric_source_guidance=metric_source_guidance,
                ),
            )
            attempt = _evaluate_candidate(
                task=task,
                candidate=repaired_candidate,
                intent=intent,
                schema=schema,
                table_schemas=table_schemas,
                stage="critic_repair",
            )
            attempts.append(attempt)
            logger.info(
                "candidate evaluated",
                instance_id=task.instance_id,
                stage="critic_repair",
                validation_ok=attempt["validation"]["ok"],
                execution_ok=attempt["execution_result"]["ok"],
                score=attempt["score"],
                elapsed_seconds=attempt["elapsed_seconds"],
                row_count=attempt["execution_result"]["row_count"],
            )
            if attempt["execution_result"]["ok"]:
                best_attempt = attempt
            else:
                best_attempt = _best_attempt(attempts)

    trace_payload = {
        "instance_id": task.instance_id,
        "db": task.db,
        "question": task.question,
        "retrieval_mode": schema.retrieval_mode,
        "schema_selection": schema.model_dump(mode="json"),
        "intent": intent.model_dump(mode="json"),
        "prompt_hashes": prompt_hashes,
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
        trace_payload["attempts"] = [_trace_attempt(attempt) for attempt in attempts]
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
    trace_payload["attempts"] = [_trace_attempt(attempt) for attempt in attempts]
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


def _evaluate_candidate(
    *,
    task: Task,
    candidate: SQLCandidate,
    intent: Intent,
    schema: SchemaSelection,
    table_schemas: dict[str, TableSchema] | None = None,
    stage: str,
) -> dict[str, Any]:
    """Validate and execute one candidate, then return a trace-ready attempt record."""

    started_at = perf_counter()
    validation = validate_sql(
        candidate.sql,
        allowed_tables=schema.expanded_tables,
        table_schemas=table_schemas,
    )
    if validation.ok:
        try:
            dataframe = fetch_query_dataframe(candidate.sql, db=task.db)
            execution = ExecutionResult(
                ok=True,
                row_count=len(dataframe),
                columns=[str(column) for column in dataframe.columns],
                sample_rows=_dataframe_records(dataframe.head(3)),
                csv_path=None,
                error=None,
            )
        except Exception as exc:
            dataframe = None
            execution = ExecutionResult(
                ok=False,
                row_count=0,
                columns=[],
                sample_rows=[],
                csv_path=None,
                error=str(exc),
            )
    else:
        dataframe = None
        execution = ExecutionResult(
            ok=False,
            row_count=0,
            columns=[],
            sample_rows=[],
            csv_path=None,
            error="Validation failed before execution.",
        )

    aggregate_grain = _infer_aggregate_grain(
        task=task,
        candidate=candidate,
        schema=schema,
        table_schemas=table_schemas or {},
        validation=validation,
        execution=execution,
    )
    result_profile = profile_dataframe(dataframe) if execution.ok else None
    shape_report = _infer_output_shape_report(
        intent=intent,
        candidate=candidate,
        execution=execution,
        result_profile=result_profile,
    )
    filter_grounding_report = _infer_filter_grounding_report(
        task=task,
        candidate=candidate,
        schema=schema,
        table_schemas=table_schemas or {},
        validation=validation,
        execution=execution,
    )
    logger.debug(
        "candidate processed",
        stage=stage,
        validation_ok=validation.ok,
        execution_ok=execution.ok,
        row_count=execution.row_count,
        error=execution.error,
    )
    attempt: dict[str, Any] = {
        "stage": stage,
        "sql": candidate.sql,
        "explanation": candidate.explanation,
        "assumptions": candidate.assumptions,
        "constraint_ledger": candidate.constraint_ledger,
        "unsupported_assumptions": candidate.unsupported_assumptions,
        "candidate_confidence": candidate.confidence,
        "validation": validation.model_dump(mode="json"),
        "execution_result": execution.model_dump(mode="json"),
        "filter_grounding_report": (
            filter_grounding_report.model_dump(mode="json")
            if filter_grounding_report is not None
            else None
        ),
        "shape_report": shape_report.model_dump(mode="json") if shape_report is not None else None,
        "score_breakdown": _attempt_score_breakdown(
            intent=intent,
            candidate=candidate,
            validation=validation,
            execution=execution,
            aggregate_grain=aggregate_grain,
            result_profile=result_profile,
            shape_report=shape_report,
            filter_grounding_report=filter_grounding_report,
        ),
    }
    attempt["score"] = sum(attempt["score_breakdown"].values())

    if result_profile is not None:
        attempt["result_profile"] = result_profile
        attempt["_dataframe"] = dataframe
    if aggregate_grain is not None:
        attempt["aggregate_grain"] = aggregate_grain.model_dump(mode="json")
    attempt["elapsed_seconds"] = round(perf_counter() - started_at, 3)

    return attempt


def _attempt_score(
    *,
    candidate: SQLCandidate,
    intent: Intent | None = None,
    validation: ValidationReport,
    execution: ExecutionResult,
    aggregate_grain: AggregateGrainReport | None = None,
    result_profile: dict[str, Any] | None = None,
    shape_report: OutputShapeReport | None = None,
    filter_grounding_report: FilterGroundingReport | None = None,
) -> float:
    """Rank candidates by verification evidence first and confidence last."""

    return sum(
        _attempt_score_breakdown(
            intent=intent,
            candidate=candidate,
            validation=validation,
            execution=execution,
            aggregate_grain=aggregate_grain,
            result_profile=result_profile,
            shape_report=shape_report,
            filter_grounding_report=filter_grounding_report,
        ).values()
    )


def _attempt_score_breakdown(
    *,
    intent: Intent | None,
    candidate: SQLCandidate,
    validation: ValidationReport,
    execution: ExecutionResult,
    aggregate_grain: AggregateGrainReport | None = None,
    result_profile: dict[str, Any] | None = None,
    shape_report: OutputShapeReport | None = None,
    filter_grounding_report: FilterGroundingReport | None = None,
) -> dict[str, float]:
    """Return the score contribution for each verification check."""

    return {
        "execution_status": _execution_status_adjustment(execution),
        "validation": _validation_adjustment(validation),
        "shape": _output_shape_adjustment(intent, execution, result_profile, shape_report),
        "filter_grounding": _filter_grounding_adjustment(
            intent,
            validation,
            execution,
            filter_grounding_report,
        ),
        "aggregate_grain": _aggregate_grain_adjustment(aggregate_grain),
        "cardinality": _cardinality_plausibility_adjustment(execution, result_profile),
        "confidence_tiebreaker": candidate.confidence * 0.01,
    }


def _best_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the highest-scoring attempt so far."""

    if not attempts:
        return None
    return max(attempts, key=lambda attempt: float(attempt["score"]))


def _comparison_attempt_summary(attempt: dict[str, Any]) -> dict[str, Any]:
    """Render one attempt in a compact, comparison-friendly format."""

    return {
        "stage": attempt["stage"],
        "sql": attempt["sql"],
        "assumptions": attempt.get("assumptions", []),
        "constraint_ledger": attempt.get("constraint_ledger", []),
        "unsupported_assumptions": attempt.get("unsupported_assumptions", []),
        "candidate_confidence": attempt["candidate_confidence"],
        "score": attempt["score"],
        "score_breakdown": attempt.get("score_breakdown", {}),
        "validation": attempt["validation"],
        "execution_result": attempt["execution_result"],
        "filter_grounding_report": attempt.get("filter_grounding_report"),
        "shape_report": attempt.get("shape_report"),
        "result_profile": attempt.get("result_profile", {}),
    }


def _infer_aggregate_grain(
    *,
    task: Task,
    candidate: SQLCandidate,
    schema: SchemaSelection,
    table_schemas: dict[str, TableSchema],
    validation: Any,
    execution: Any,
) -> AggregateGrainReport | None:
    """Infer the intended aggregate grain for one candidate."""

    sql = str(candidate.sql)
    normalized_sql = " ".join(sql.lower().split())
    if not _looks_aggregate_query(normalized_sql):
        return None

    uses_distinct = "count(distinct" in normalized_sql
    has_joins = " join " in normalized_sql
    selected_tables = [
        table_schemas.get(table_name) or TableSchema(name=table_name, ddl="", searchable_text="")
        for table_name in schema.expanded_tables
    ]
    entity_like_tables = [table for table in selected_tables if _table_is_entity_like(table)]
    question_text = task.question.lower()
    wants_unique = _question_requests_unique_entities(question_text)
    target = _count_distinct_target(sql)

    if uses_distinct:
        if entity_like_tables and not has_joins and not wants_unique:
            table_name = entity_like_tables[0].full_name or entity_like_tables[0].name
            reason = (
                f"Single entity table {table_name} has no joins, so DISTINCT is likely redundant."
            )
            distinct_reason = (
                f"DISTINCT on {target or 'the counted column'} is probably unnecessary."
            )
            return AggregateGrainReport(
                inferred_grain="row_count",
                reason=reason,
                distinct_reason=distinct_reason,
                uses_distinct=True,
                has_joins=has_joins,
                selected_tables=[table.full_name or table.name for table in selected_tables],
            )

        if has_joins or wants_unique:
            reason = "Joins or unique-count wording suggest deduping entity rows."
            distinct_reason = f"DISTINCT on {target or 'the counted column'} is justified here."
            return AggregateGrainReport(
                inferred_grain="distinct_entity_count",
                reason=reason,
                distinct_reason=distinct_reason,
                uses_distinct=True,
                has_joins=has_joins,
                selected_tables=[table.full_name or table.name for table in selected_tables],
            )

        reason = "DISTINCT appears to be a cautious choice, but the target grain is unclear."
        distinct_reason = f"DISTINCT on {target or 'the counted column'} may be unnecessary."
        return AggregateGrainReport(
            inferred_grain="unknown",
            reason=reason,
            distinct_reason=distinct_reason,
            uses_distinct=True,
            has_joins=has_joins,
            selected_tables=[table.full_name or table.name for table in selected_tables],
        )

    if entity_like_tables and not has_joins:
        table_name = entity_like_tables[0].full_name or entity_like_tables[0].name
        return AggregateGrainReport(
            inferred_grain="row_count",
            reason=f"Single entity table {table_name} with no joins usually counts rows.",
            uses_distinct=False,
            has_joins=has_joins,
            selected_tables=[table.full_name or table.name for table in selected_tables],
        )

    if has_joins:
        return AggregateGrainReport(
            inferred_grain="value_count",
            reason=(
                "Join multiplicity suggests the result may count joined rows rather than entities."
            ),
            uses_distinct=False,
            has_joins=has_joins,
            selected_tables=[table.full_name or table.name for table in selected_tables],
        )

    return AggregateGrainReport(
        inferred_grain="unknown",
        reason="Aggregate query does not clearly indicate a unique entity grain.",
        uses_distinct=False,
        has_joins=has_joins,
        selected_tables=[table.full_name or table.name for table in selected_tables],
    )


def _aggregate_grain_adjustment(report: AggregateGrainReport | None) -> float:
    """Score candidate aggregates according to the inferred grain."""

    if report is None:
        return 0.0
    if report.inferred_grain == "row_count":
        return -6.0 if report.uses_distinct else 2.0
    if report.inferred_grain == "distinct_entity_count":
        return 2.0 if report.uses_distinct else -4.0
    if report.inferred_grain == "value_count":
        return 0.5
    return 0.0


def _execution_status_adjustment(execution: ExecutionResult) -> float:
    """Make execution success dominate every other signal."""

    return 1000.0 if execution.ok else -1000.0


def _validation_adjustment(validation: ValidationReport) -> float:
    """Reward SQL that validates cleanly and penalize noisy validation."""

    score = 120.0 if validation.ok else -180.0
    score -= 15.0 * len(validation.errors)
    score -= 5.0 * len(validation.warnings)
    return score


def _output_shape_adjustment(
    intent: Intent | None,
    execution: ExecutionResult,
    result_profile: dict[str, Any] | None,
    shape_report: OutputShapeReport | None = None,
) -> float:
    """Prefer result shapes that match the task's stated output contract."""

    if not execution.ok:
        return 0.0

    score = 0.0
    if shape_report is not None:
        score += _shape_report_adjustment(shape_report)

    profile = result_profile or {
        "row_count": execution.row_count,
        "columns": execution.columns,
    }
    row_count = int(profile.get("row_count") or 0)
    column_count = len(profile.get("columns") or [])
    expectation = _normalized_output_expectation(intent)

    expected_columns = _expected_output_columns(expectation)
    if expected_columns is not None and shape_report is None:
        if column_count == expected_columns:
            score += 45.0
        elif abs(column_count - expected_columns) == 1:
            score += 10.0
        else:
            score -= 25.0

    if shape_report is None:
        if _expects_scalar_output(expectation):
            if row_count == 1 and column_count == 1:
                score += 35.0
            elif row_count == 1 or column_count == 1:
                score += 12.0
            else:
                score -= 18.0

        if _expects_tabular_output(expectation):
            if row_count > 0 and column_count >= 2:
                score += 18.0
            elif row_count > 0:
                score += 6.0
            else:
                score -= 12.0

        if row_count > 0 and column_count > 0:
            score += 4.0
        else:
            score -= 4.0

    return score


def _shape_report_adjustment(report: OutputShapeReport | None) -> float:
    """Score a candidate by how well its observed columns match the inferred contract."""

    if report is None:
        return 0.0

    score = 0.0
    if report.expected_columns:
        if report.observed_columns == report.expected_columns:
            score += 30.0
        elif not report.missing_columns:
            score += 12.0
        else:
            score -= 28.0 * len(report.missing_columns)

    if report.violations:
        score -= 20.0 * len(report.violations)
        if any("grouped key" in violation for violation in report.violations):
            score -= 15.0
    return score


def _projection_columns(statement: exp.Expression) -> list[str]:
    """Return the projected output column names from one query."""

    columns: list[str] = []
    for expression in getattr(statement, "expressions", []):
        name = _expression_output_name(expression)
        if name is not None:
            columns.append(name)
    return columns


def _grouped_columns(statement: exp.Expression) -> list[str]:
    """Return simple grouped column names from one query."""

    group = statement.args.get("group")
    if group is None:
        return []

    columns: list[str] = []
    for expression in group.expressions:
        if isinstance(expression, exp.Column):
            columns.append(_normalized_column_name(expression))
    return columns


def _expression_output_name(expression: exp.Expression) -> str | None:
    """Return the visible output name for one select expression."""

    if _is_aggregate_projection(expression):
        alias = getattr(expression, "alias_or_name", None)
        if alias:
            return str(alias)
        return None

    if isinstance(expression, exp.Alias):
        alias = expression.alias_or_name
        return str(alias) if alias else None
    if isinstance(expression, exp.Column):
        return _normalized_column_name(expression)

    alias = getattr(expression, "alias_or_name", None)
    if alias:
        return str(alias)

    if expression.is_star:
        return "*"
    return None


def _is_aggregate_projection(expression: exp.Expression) -> bool:
    """Return True when one select expression contains an aggregate."""

    return expression.find(exp.AggFunc) is not None


def _normalized_column_name(expression: exp.Column) -> str:
    """Return the visible column name for one AST column node."""

    return str(expression.alias_or_name or expression.name)


def _is_identifier_like_column_name(column_name: str) -> bool:
    """Return True for likely identifier columns that should stay visible."""

    normalized = column_name.replace('"', "").replace("`", "")
    return bool(
        re.search(r"(?:^|[_\W])id$", normalized, flags=re.IGNORECASE)
        or re.search(r"[A-Z]ID$", column_name)
        or normalized.lower().endswith("_id")
        or normalized.lower().endswith("id")
    )


def _filter_grounding_adjustment(
    intent: Intent | None,
    validation: ValidationReport,
    execution: ExecutionResult,
    report: FilterGroundingReport | None,
) -> float:
    """Prefer candidates whose filters appear grounded in observed values."""

    if intent is None or not intent.filters:
        return 0.0
    if not execution.ok:
        return -30.0
    if report is not None and report.zero_like_result:
        if report is not None and report.value_rewrites:
            return 16.0
        return -22.0
    score = 14.0
    if validation.errors:
        score -= 8.0
    if validation.warnings:
        score -= 3.0 * len(validation.warnings)
    if report is not None and report.value_rewrites:
        score += 6.0
    return score


def _cardinality_plausibility_adjustment(
    execution: ExecutionResult,
    result_profile: dict[str, Any] | None,
) -> float:
    """Reward results whose size looks plausible for the query type."""

    if not execution.ok:
        return 0.0

    profile = result_profile or {
        "row_count": execution.row_count,
        "sample_rows": execution.sample_rows,
    }
    row_count = int(profile.get("row_count") or 0)
    if row_count == 0:
        return -16.0

    sample_rows = profile.get("sample_rows") or []
    numeric_values = [
        value
        for value in (_coerce_number(item) for row in sample_rows for item in row.values())
        if value is not None
    ]
    if not numeric_values:
        return 0.0

    max_value = max(numeric_values)
    if row_count == 1 and max_value <= 1:
        return -14.0
    if row_count <= 2 and max_value <= 2:
        return -8.0
    if row_count > 0:
        return 3.0
    return 0.0


def _infer_output_shape_report(
    *,
    intent: Intent,
    candidate: SQLCandidate,
    execution: ExecutionResult,
    result_profile: dict[str, Any] | None,
) -> OutputShapeReport | None:
    """Infer the answer contract from the SQL, intent, and executed result."""

    if not execution.ok:
        return None

    profile = result_profile or {
        "columns": execution.columns,
        "sample_rows": execution.sample_rows,
    }
    observed_columns = [str(column) for column in profile.get("columns") or execution.columns]

    try:
        statement = sqlglot.parse_one(candidate.sql, read="snowflake")
    except ParseError:
        return OutputShapeReport(
            expected_columns=observed_columns,
            observed_columns=observed_columns,
            violations=["SQL could not be parsed for shape analysis."],
        )

    projected_columns = _projection_columns(statement)
    grouped_columns = _grouped_columns(statement)
    expected_columns = list(dict.fromkeys(projected_columns))
    violations: list[str] = []
    has_non_aggregate_projection = any(
        not _is_aggregate_projection(expression) for expression in statement.expressions
    )

    for grouped_column in grouped_columns:
        if (
            _is_identifier_like_column_name(grouped_column)
            and grouped_column not in expected_columns
            and has_non_aggregate_projection
        ):
            expected_columns.append(grouped_column)
            violations.append(f"missing grouped key {grouped_column}")

    missing_columns = [column for column in expected_columns if column not in observed_columns]
    if missing_columns:
        violations.extend(f"missing expected column {column}" for column in missing_columns)

    return OutputShapeReport(
        expected_columns=expected_columns,
        observed_columns=observed_columns,
        projected_columns=projected_columns,
        grouped_columns=grouped_columns,
        missing_columns=missing_columns,
        violations=list(dict.fromkeys(violations)),
    )


def _infer_filter_grounding_report(
    *,
    task: Task,
    candidate: SQLCandidate,
    schema: SchemaSelection,
    table_schemas: dict[str, TableSchema],
    validation: ValidationReport,
    execution: ExecutionResult,
) -> FilterGroundingReport | None:
    """Probe exact string filters when the candidate returns no rows."""

    if not _execution_is_zero_like(candidate, execution, result_profile=None):
        return None

    try:
        statement = sqlglot.parse_one(candidate.sql, read="snowflake")
    except ParseError:
        return FilterGroundingReport(
            exact_filters=[],
            probes=[],
            value_rewrites=[],
            reason="SQL could not be parsed for filter grounding.",
        )

    exact_filters = _extract_exact_string_filters(statement)
    if not exact_filters:
        return None

    selected_tables = validation.referenced_tables or schema.expanded_tables
    probes: list[dict[str, object]] = []
    value_rewrites: list[dict[str, object]] = []
    seen_probe_keys: set[tuple[str, str, str]] = set()

    for filter_match in exact_filters:
        targets = _filter_probe_targets(
            filter_match,
            selected_tables=selected_tables,
            table_schemas=table_schemas,
        )
        for target in targets:
            probe_key = (target["table"], target["column"], target["literal"])
            if probe_key in seen_probe_keys:
                continue
            seen_probe_keys.add(probe_key)

            probe_sql = _string_filter_probe_sql(
                table_name=target["table"],
                column_name=target["column"],
                literal=target["literal"],
            )
            try:
                probe_frame = fetch_query_dataframe(probe_sql, db=task.db)
            except Exception as exc:
                probes.append(
                    {
                        **target,
                        "probe_sql": probe_sql,
                        "error": str(exc),
                    }
                )
                continue

            matched_values = [
                str(value)
                for value in (
                    row.get("MATCHED_VALUE")
                    or row.get("matched_value")
                    or row.get(target["column"])
                    for row in probe_frame.to_dict(orient="records")
                )
                if value not in {None, "", target["literal"]}
            ]
            probe_record: dict[str, object] = {
                **target,
                "probe_sql": probe_sql,
                "matched_values": matched_values,
            }
            probes.append(probe_record)
            if matched_values:
                value_rewrites.append(
                    {
                        "filter": target["raw_filter"],
                        "rewrite": matched_values[0],
                        "table": target["table"],
                        "column": target["column"],
                        "probe_sql": probe_sql,
                    }
                )
                break
        if value_rewrites:
            break

    if not probes:
        return FilterGroundingReport(
            exact_filters=[filter_match["raw_filter"] for filter_match in exact_filters],
            probes=[],
            value_rewrites=[],
            zero_like_result=True,
            reason="No probe targets were available for the empty result.",
        )

    if value_rewrites:
        reason = "Empty result but probe values suggest a stored label variant."
    else:
        reason = "Empty result and probes did not find a grounded label rewrite."

    return FilterGroundingReport(
        exact_filters=[filter_match["raw_filter"] for filter_match in exact_filters],
        probes=probes,
        value_rewrites=value_rewrites,
        zero_like_result=True,
        reason=reason,
    )


def _execution_is_zero_like(
    candidate: SQLCandidate,
    execution: ExecutionResult,
    *,
    result_profile: dict[str, Any] | None,
) -> bool:
    """Return True when one result row still represents an empty aggregate result."""

    if not execution.ok:
        return False
    if execution.row_count == 0:
        return True
    if not _looks_aggregate_query(candidate.sql):
        return False

    profile = result_profile or {
        "sample_rows": execution.sample_rows,
    }
    sample_rows = profile.get("sample_rows") or execution.sample_rows
    numeric_values = [
        value
        for value in (_coerce_number(item) for row in sample_rows for item in row.values())
        if value is not None
    ]
    return bool(numeric_values) and max(numeric_values) == 0


def _extract_exact_string_filters(statement: exp.Expression) -> list[dict[str, str]]:
    """Return exact string equality filters from one parsed query."""

    filters: list[dict[str, str]] = []
    for node in statement.walk():
        if isinstance(node, exp.EQ):
            comparison = _extract_string_equality(node)
            if comparison is not None:
                filters.append(comparison)
        elif isinstance(node, exp.In):
            comparison = _extract_string_membership(node)
            if comparison is not None:
                filters.extend(comparison)
    return filters


def _extract_string_equality(node: exp.EQ) -> dict[str, str] | None:
    """Return one column-to-literal equality comparison when present."""

    left = node.left
    right = node.right
    column, literal = _column_and_string_literal(left, right)
    if column is None or literal is None:
        return None
    return {
        "raw_filter": node.sql(dialect="snowflake"),
        "column": column.sql(dialect="snowflake"),
        "literal": literal,
    }


def _extract_string_membership(node: exp.In) -> list[dict[str, str]] | None:
    """Return one column-to-literal IN comparison when present."""

    column = node.this if isinstance(node.this, exp.Column) else None
    if column is None:
        return None

    filters: list[dict[str, str]] = []
    for expression in node.expressions:
        if not isinstance(expression, exp.Literal) or not expression.is_string:
            return None
        filters.append(
            {
                "raw_filter": node.sql(dialect="snowflake"),
                "column": column.sql(dialect="snowflake"),
                "literal": expression.this,
            }
        )
    return filters


def _column_and_string_literal(
    left: exp.Expression,
    right: exp.Expression,
) -> tuple[exp.Column | None, str | None]:
    """Return a column and a string literal from one equality comparison."""

    if isinstance(left, exp.Column) and isinstance(right, exp.Literal) and right.is_string:
        return left, right.this
    if isinstance(right, exp.Column) and isinstance(left, exp.Literal) and left.is_string:
        return right, left.this
    return None, None


def _filter_probe_targets(
    filter_match: dict[str, str],
    *,
    selected_tables: list[str],
    table_schemas: dict[str, TableSchema],
) -> list[dict[str, str]]:
    """Return low-cost probe targets for one empty exact filter."""

    column_name = filter_match["column"].split(".")[-1].strip('"')
    literal = filter_match["literal"]
    schema_lookup = {
        table_name.lower(): table_schema for table_name, table_schema in table_schemas.items()
    }
    tables = [table_name for table_name in selected_tables if table_name.lower() in schema_lookup]
    tables.sort(key=lambda name: _table_probe_rank(name, column_name))

    targets: list[dict[str, str]] = []
    for table_name in tables:
        schema = schema_lookup[table_name.lower()]
        column_candidates = _probe_columns_for_table(schema, column_name, literal)
        for probe_column in column_candidates:
            targets.append(
                {
                    "raw_filter": filter_match["raw_filter"],
                    "table": table_name,
                    "column": probe_column,
                    "literal": literal,
                }
            )
            if len(targets) >= 4:
                return targets
    return targets


def _probe_columns_for_table(
    table_schema: TableSchema,
    filter_column: str,
    literal: str,
) -> list[str]:
    """Return likely string columns to probe within one table."""

    string_columns = [
        column.name for column in table_schema.columns if _column_looks_string_like(column.type)
    ]
    if not string_columns:
        return []

    exact_matches = [
        column.name
        for column in table_schema.columns
        if column.name.lower() == filter_column.lower()
    ]
    if exact_matches:
        return exact_matches

    if _table_looks_like_lookup(table_schema) or _literal_looks_label_like(literal):
        preferred = [
            column
            for column in string_columns
            if _column_looks_label_like(column) or _column_looks_key_like(column)
        ]
        if preferred:
            return preferred

    preferred = [column for column in string_columns if filter_column.lower() in column.lower()]
    if preferred:
        return preferred
    return string_columns[:2]


def _table_probe_rank(table_name: str, filter_column: str) -> tuple[int, str]:
    """Rank tables so likely lookup tables are probed first."""

    lowered = table_name.lower()
    if filter_column.lower() in lowered:
        return 0, lowered
    if _table_looks_like_lookup_name(lowered):
        return 1, lowered
    return 2, lowered


def _string_filter_probe_sql(*, table_name: str, column_name: str, literal: str) -> str:
    """Build one low-cost LIKE probe for a string filter."""

    escaped = literal.replace("'", "''")
    return (
        f'SELECT DISTINCT "{column_name}" AS MATCHED_VALUE '
        f"FROM {table_name} "
        f"WHERE LOWER(CAST(\"{column_name}\" AS VARCHAR)) LIKE LOWER('%{escaped}%') "
        "LIMIT 5"
    )


def _column_looks_string_like(column_type: str | None) -> bool:
    """Return True when a schema column looks like a text field."""

    if column_type is None:
        return True
    lowered = column_type.lower()
    return any(token in lowered for token in ("char", "text", "string", "varchar", "variant"))


def _table_looks_like_lookup(table_schema: TableSchema) -> bool:
    """Return True when a table name suggests a lookup or code table."""

    return _table_looks_like_lookup_name((table_schema.full_name or table_schema.name).lower())


def _table_looks_like_lookup_name(table_name: str) -> bool:
    """Return True when a table name suggests a lookup or code table."""

    return any(
        token in table_name
        for token in (
            "summary",
            "lookup",
            "ref_",
            "_ref",
            "dim_",
            "_dim",
            "map",
            "code",
            "label",
            "country",
        )
    )


def _column_looks_label_like(column_name: str) -> bool:
    """Return True when a column name looks like a human-readable label."""

    lowered = column_name.lower()
    return any(token in lowered for token in ("name", "label", "display", "title", "desc"))


def _column_looks_key_like(column_name: str) -> bool:
    """Return True when a column name looks like a stored key."""

    lowered = column_name.lower()
    return any(token in lowered for token in ("key", "code", "id"))


def _literal_looks_label_like(literal: str) -> bool:
    """Return True when a filter literal looks like a display label."""

    return (any(part.isalpha() for part in literal) and " " in literal) or literal[:1].isupper()


def _normalized_output_expectation(intent: Intent | None) -> str:
    """Normalize the intent's output expectation for shape heuristics."""

    if intent is None:
        return ""
    return " ".join(intent.output_expectation.lower().split())


def _expected_output_columns(expectation: str) -> int | None:
    """Infer how many columns the answer should expose from the intent text."""

    if not expectation:
        return None
    if any(token in expectation for token in (" and ", " columns", " per ", " by ")):
        return 2
    if any(token in expectation for token in ("count", "how many", "one count", "single count")):
        return 1
    if any(token in expectation for token in ("one column", "single column")):
        return 1
    return None


def _expects_scalar_output(expectation: str) -> bool:
    """Return True when the answer is likely a single value."""

    return any(token in expectation for token in ("count", "how many", "one count", "single count"))


def _expects_tabular_output(expectation: str) -> bool:
    """Return True when the answer is likely a multi-column result set."""

    return any(
        token in expectation for token in (" and ", " columns", " per ", " by ", " rows", " list ")
    )


def _table_is_entity_like(table: TableSchema) -> bool:
    """Return True when a table looks like a master/entity table."""

    table_name = (table.full_name or table.name).split(".")[-1].lower()
    return (
        table_name.startswith(("mst_", "dim_", "ref_", "lkp_"))
        or table_name.endswith(("_users", "_user"))
        or "master" in table_name
        or "entity" in table_name
    )


def _question_requests_unique_entities(question_text: str) -> bool:
    """Return True when the wording asks for unique or deduplicated entities."""

    normalized = question_text.lower()
    return any(
        keyword in normalized
        for keyword in ("unique", "distinct", "dedupe", "dedup", "without duplicates")
    )


def _aggregate_grain_guidance(
    task: Task,
    intent: Intent,
    schema: SchemaSelection,
    table_schemas: dict[str, TableSchema],
) -> str | None:
    """Return prompt guidance when a task looks like a plain count over one entity table."""

    normalized_question = " ".join(
        [
            task.question.lower(),
            intent.summary.lower(),
            intent.output_expectation.lower(),
            " ".join(intent.metrics).lower(),
            " ".join(intent.filters).lower(),
            " ".join(intent.assumptions).lower(),
        ]
    )
    if not (
        _expects_scalar_output(_normalized_output_expectation(intent))
        or any(token in normalized_question for token in ("how many", "count", "number of"))
    ):
        return None

    question_text = " ".join(
        [
            task.question,
            intent.summary,
            intent.output_expectation,
            " ".join(intent.metrics),
            " ".join(intent.filters),
            " ".join(intent.assumptions),
        ]
    )
    selected_tables = [
        table_schemas.get(table_name) or TableSchema(name=table_name, ddl="", searchable_text="")
        for table_name in schema.expanded_tables
    ]
    entity_like_tables = [table for table in selected_tables if _table_is_entity_like(table)]
    if len(selected_tables) != 1 or len(entity_like_tables) != 1:
        return None

    table_name = entity_like_tables[0].full_name or entity_like_tables[0].name
    if _question_requests_unique_entities(question_text):
        return (
            f"Grain guidance: {table_name} looks like a single entity table, and the question "
            "asks for unique or deduplicated entities. Use COUNT(DISTINCT ...) only if the "
            "intent explicitly requires uniqueness."
        )

    return (
        f"Grain guidance: {table_name} looks like a single entity table. Treat this as a "
        "row-count style aggregation and default to COUNT(*) per group. Do not switch to "
        "COUNT(DISTINCT ...) unless the question explicitly asks for unique or deduplicated "
        "entities."
    )


def _metric_source_guidance(
    task: Task,
    intent: Intent,
    table_schemas: dict[str, TableSchema],
) -> str | None:
    """Return task-aware guidance for choosing metric source columns."""

    if not table_schemas:
        return None

    task_text = " ".join(
        [
            task.question,
            intent.summary,
            intent.output_expectation,
            " ".join(intent.metrics),
            " ".join(intent.filters),
            " ".join(intent.assumptions),
        ]
    )
    if not _looks_metric_source_sensitive(task_text):
        return None

    table_lines: list[str] = []
    for table_name in sorted(table_schemas):
        table = table_schemas[table_name]
        metric_columns = [column for column in table.columns if _column_looks_native_metric(column)]
        if not metric_columns:
            continue
        metric_columns = sorted(
            metric_columns,
            key=lambda column: _metric_column_sort_key(column, task_text),
        )
        grain_columns = [
            column.name for column in table.columns if _column_looks_grain_or_filter_column(column)
        ]
        table_identity = table.full_name or table_name
        metrics_preview = ", ".join(column.name for column in metric_columns[:6])
        if grain_columns:
            grain_preview = ", ".join(grain_columns[:8])
            table_lines.append(
                f"- {table_identity}: native metrics [{metrics_preview}]; "
                f"grain/filter/time columns [{grain_preview}]"
            )
        else:
            table_lines.append(f"- {table_identity}: native metrics [{metrics_preview}]")
        if len(table_lines) >= 8:
            break

    if not table_lines:
        return None

    return (
        "Choose the metric source at the requested answer grain. Prefer a native metric "
        "column on a table that already has the needed grouping keys, time key, and "
        "filters when its semantics match the question. Join lower-grain detail tables "
        "only when the task requires detail-level filters, grouping, output columns, an "
        "explicit formula, or no suitable native metric exists.\n"
        "When several native metric columns exist, choose by column-name semantics from "
        "the question; do not treat subtotal, total due, tax, freight, or line-item "
        "formulas as interchangeable. Candidate metric columns below are ordered by "
        "semantic fit to the task text.\n"
        "Native metric candidates in selected tables:\n" + "\n".join(table_lines)
    )


def _looks_metric_source_sensitive(text: str) -> bool:
    """Return True when the task likely depends on choosing the right metric source."""

    normalized = text.lower()
    return any(
        token in normalized
        for token in (
            "amount",
            "annual",
            "average",
            "balance",
            "compare",
            "cost",
            "difference",
            "fee",
            "freight",
            "metric",
            "payment",
            "price",
            "quota",
            "rate",
            "revenue",
            "sales",
            "sum",
            "tax",
            "total",
            "value",
        )
    )


def _column_looks_native_metric(column: Any) -> bool:
    """Return True for columns that look like stored business metrics."""

    if not _column_looks_numeric_like(column.type):
        return False
    if _column_looks_key_like(column.name):
        return False
    normalized = column.name.lower().replace("_", "")
    return any(
        token in normalized
        for token in (
            "amount",
            "amt",
            "balance",
            "cost",
            "due",
            "fare",
            "fee",
            "freight",
            "price",
            "quota",
            "rate",
            "revenue",
            "sales",
            "subtotal",
            "tax",
            "total",
            "value",
        )
    )


def _metric_column_sort_key(column: Any, task_text: str) -> tuple[int, str]:
    """Rank metric columns so prompt candidates lead with better semantic matches."""

    normalized_text = task_text.lower()
    normalized_name = column.name.lower().replace("_", "")
    score = 0

    if "quota" in normalized_text and "quota" in normalized_name:
        score += 8
    if "sales" in normalized_text and "sales" in normalized_name:
        score += 6
    if "revenue" in normalized_text and "revenue" in normalized_name:
        score += 6
    if "total" in normalized_text and "total" in normalized_name:
        score += 5
    if "amount" in normalized_text and ("amount" in normalized_name or "amt" in normalized_name):
        score += 5
    if "due" in normalized_text and "due" in normalized_name:
        score += 5
    if "tax" in normalized_text and "tax" in normalized_name:
        score += 5
    if (
        any(token in normalized_text for token in ("freight", "shipping"))
        and "freight" in normalized_name
    ):
        score += 5

    if "subtotal" in normalized_name and not any(
        token in normalized_text for token in ("subtotal", "sub total", "net", "pre-tax", "pretax")
    ):
        score -= 3
    if any(token in normalized_name for token in ("tax", "freight")) and not any(
        token in normalized_text for token in ("tax", "freight", "shipping")
    ):
        score -= 4

    return (-score, normalized_name)


def _column_looks_numeric_like(column_type: str | None) -> bool:
    """Return True when a schema type appears numeric."""

    if column_type is None:
        return False
    lowered = column_type.lower()
    return any(
        token in lowered
        for token in ("number", "numeric", "decimal", "int", "float", "double", "real")
    )


def _column_looks_grain_or_filter_column(column: Any) -> bool:
    """Return True for columns likely to define grouping, time, joins, or filters."""

    normalized = column.name.lower()
    return any(
        token in normalized
        for token in (
            "date",
            "time",
            "year",
            "month",
            "day",
            "week",
            "quarter",
            "period",
            "id",
            "key",
            "code",
            "name",
            "type",
            "status",
            "category",
            "group",
        )
    )


def _grain_guidance_block(guidance: str | None) -> str:
    """Render an optional grain hint as a prompt section."""

    if not guidance:
        return ""
    return f"Grain guidance:\n{guidance}\n\n"


def _metric_source_guidance_block(guidance: str | None) -> str:
    """Render optional metric source guidance as a prompt section."""

    if not guidance:
        return ""
    return f"Metric source guidance:\n{guidance}\n\n"


def _count_distinct_target(sql: str) -> str | None:
    """Extract the DISTINCT target column from a COUNT aggregate."""

    match = re.search(r"count\s*\(\s*distinct\s+([^)]+)\)", sql, flags=re.IGNORECASE)
    if match is None:
        return None
    target = match.group(1).strip()
    target = re.sub(r"\s+as\s+\w+$", "", target, flags=re.IGNORECASE)
    return target.strip(" \"'`")


def _aggregate_verification_reason(attempt: dict[str, Any]) -> str | None:
    """Return a reason when an aggregate result looks suspiciously small."""

    execution = attempt["execution_result"]
    if not execution["ok"]:
        return None
    sql = str(attempt["sql"]).lower()
    if not _looks_aggregate_query(sql):
        return None

    sample_rows = execution.get("sample_rows") or []
    if not sample_rows:
        return None

    first_row = sample_rows[0]
    numeric_values = [
        value
        for value in (_coerce_number(item) for item in first_row.values())
        if value is not None
    ]
    if not numeric_values:
        return None

    row_count = int(execution.get("row_count") or 0)
    max_value = max(numeric_values)
    if row_count == 1 and max_value <= 1:
        return "Aggregate query returned a single very small numeric result."
    if row_count <= 2 and max_value <= 2:
        return "Aggregate query returned only tiny numeric results."
    return None


def _looks_aggregate_query(sql: str) -> bool:
    """Heuristically detect aggregate queries that deserve extra scrutiny."""

    normalized = " ".join(sql.lower().split())
    return any(
        keyword in normalized
        for keyword in (" count(", " sum(", " avg(", " min(", " max(", " group by ", " having ")
    )


def _coerce_number(value: Any) -> float | None:
    """Convert one value to a number when it looks numeric."""

    if isinstance(value, bool):
        return None
    if isinstance(value, Number):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _trace_attempt(attempt: dict[str, Any]) -> dict[str, Any]:
    """Drop non-serializable internal fields before writing the trace."""

    return {key: value for key, value in attempt.items() if not key.startswith("_")}


def _augment_intent_with_value_groundings(
    intent: Intent,
    *,
    task: Task,
    schema: SchemaSelection,
    table_schemas: dict[str, TableSchema],
) -> Intent:
    """Attach native value matches from selected schemas to the answer contract."""

    native_value_terms = _infer_native_value_terms(task, schema, table_schemas)
    if not native_value_terms:
        return intent

    merged_terms = list(dict.fromkeys([*intent.native_value_terms, *native_value_terms]))
    return intent.model_copy(update={"native_value_terms": merged_terms})


def _infer_native_value_terms(
    task: Task,
    schema: SchemaSelection,
    table_schemas: dict[str, TableSchema],
) -> list[str]:
    """Return exact sample-value matches that should stay tied to native columns."""

    question_text = _normalized_text(task.question)
    terms: list[str] = []
    for table_name in schema.expanded_tables:
        table = table_schemas.get(table_name)
        if table is None:
            continue
        table_identity = table.full_name or table.name
        for column in table.columns:
            if not _column_looks_string_like(column.type):
                continue
            for sample_value in column.sample_values:
                if not _question_mentions_literal(question_text, sample_value):
                    continue
                term = f"{table_identity}.{column.name}={sample_value}"
                if term not in terms:
                    terms.append(term)
    return terms


def _normalized_text(value: str) -> str:
    """Lower-case and collapse whitespace in one text fragment."""

    return " ".join(value.lower().split())


def _question_mentions_literal(question_text: str, literal: str) -> bool:
    """Return True when a question appears to name one stored literal value."""

    normalized_literal = _normalized_text(literal)
    if not normalized_literal:
        return False

    literal_tokens = re.findall(r"[a-z0-9]+", normalized_literal)
    if not literal_tokens:
        return False
    question_tokens = set(re.findall(r"[a-z0-9]+", question_text))
    if len(normalized_literal) < 3 or any(len(token) < 3 for token in literal_tokens):
        return all(token in question_tokens for token in literal_tokens)

    if normalized_literal in question_text:
        return True

    return all(token in question_tokens for token in literal_tokens)


def _schema_context(schema: SchemaSelection) -> str:
    """Render a compact schema summary for prompt inputs."""

    return (
        f"DB: {schema.db}\n"
        f"Selected tables: {', '.join(schema.selected_tables)}\n"
        f"Expanded tables: {', '.join(schema.expanded_tables)}\n"
        f"Rationale: {schema.rationale}"
    )


def _table_schemas_for_selection(
    schema: SchemaSelection,
    *,
    cache_path: Path = CACHE_PATH,
) -> dict[str, TableSchema]:
    """Return indexed schemas for the selected tables, falling back safely in tests."""

    try:
        db_index = load_db_index(schema.db, cache_path=cache_path)
    except (FileNotFoundError, NotADirectoryError):
        logger.warning("schema index unavailable", db=schema.db, cache_path=str(cache_path))
        return {}

    selected: dict[str, TableSchema] = {}
    for table_name in schema.expanded_tables:
        table = db_index.get(table_name)
        if table is not None:
            selected[table_name] = table
    return selected


def _sql_reference_context(
    schema: SchemaSelection,
    table_schemas: dict[str, TableSchema],
) -> str:
    """Render deterministic selected-table context for cache-friendly SQL prompts."""

    lines = [
        "SQL reference context:",
        f"Database: {schema.db}",
        "Selected tables:",
    ]
    for table_name in sorted(schema.expanded_tables):
        lines.append(f"- {table_name}")

    if not table_schemas:
        return "\n".join(lines)

    lines.append("")
    lines.append("Selected table details:")
    for table_name in sorted(table_schemas):
        table = table_schemas[table_name]
        lines.append(f"Table: {table.full_name or table_name}")
        if table.ddl.strip():
            lines.append("DDL:")
            lines.append("```sql")
            lines.append(table.ddl.strip())
            lines.append("```")
        if table.columns:
            lines.append("Columns:")
            for column in table.columns:
                lines.append(f"- {_column_context_line(column)}")
        if table.sample_rows:
            lines.append("Sample rows:")
            for row in table.sample_rows[:3]:
                lines.append(json.dumps(row, sort_keys=True))
        lines.append("")
    return "\n".join(lines).rstrip()


def _column_context_line(column: Any) -> str:
    """Render one compact column line with exact name, type, docs, and samples."""

    line = column.name
    if column.type:
        line += f" [{column.type}]"
    if column.description:
        line += f" - {column.description}"
    if column.sample_values:
        preview = ", ".join(column.sample_values[:3])
        line += f" - sample values: {preview}"
    return line


def _question_preview(question: str, *, max_length: int = 120) -> str:
    """Shorten long questions so task logs stay readable."""

    normalized = " ".join(question.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1].rstrip() + "…"


def _intent_user_prompt(
    task: Task,
    schema: SchemaSelection,
    docs_context: str,
    table_schemas: dict[str, TableSchema] | None = None,
) -> str:
    """Build the user prompt for intent extraction."""

    grounded_literals = _grounded_literal_context(task, schema, table_schemas or {})
    prompt = (
        f"Question: {task.question}\n\n"
        f"Document context:\n{docs_context}\n\n"
        f"Schema context:\n{_schema_context(schema)}"
    )
    if grounded_literals:
        prompt += f"\n\n{grounded_literals}"
    return prompt


def _grounded_literal_context(
    task: Task,
    schema: SchemaSelection,
    table_schemas: dict[str, TableSchema],
) -> str | None:
    """Render native sample-value matches for intent extraction and repair prompts."""

    native_value_terms = _infer_native_value_terms(task, schema, table_schemas)
    if not native_value_terms:
        return None
    lines = ["Grounded literal values:"]
    for term in native_value_terms:
        lines.append(f"- {term}")
    lines.append(
        "Use these as native column values. Do not recast them as behavioral definitions "
        "unless the question explicitly asks for that."
    )
    return "\n".join(lines)


def _grounded_literal_context_from_intent(intent: Intent) -> str | None:
    """Render grounded literals already attached to the answer contract."""

    if not intent.native_value_terms:
        return None
    lines = ["Grounded literal values:"]
    for term in intent.native_value_terms:
        lines.append(f"- {term}")
    lines.append(
        "Use these as native column values. Do not recast them as behavioral definitions "
        "unless the question explicitly asks for that."
    )
    return "\n".join(lines)


def _sql_generation_prompt(
    task: Task,
    intent: Intent,
    sql_reference_context: str,
    docs_context: str,
    *,
    aggregate_grain_guidance: str | None = None,
    metric_source_guidance: str | None = None,
) -> str:
    """Build the SQL-generation prompt body."""

    grounded_literals = _grounded_literal_context_from_intent(intent)
    grounded_literal_block = f"{grounded_literals}\n\n" if grounded_literals else ""
    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context}\n\n"
        f"Question: {task.question}\n\n"
        f"Intent:\n{intent.model_dump_json(indent=2)}\n\n"
        f"{grounded_literal_block}"
        f"{_grain_guidance_block(aggregate_grain_guidance)}"
        f"{_metric_source_guidance_block(metric_source_guidance)}"
        "Write the SQL using only the reference context above."
    )


def _sql_repair_prompt(
    task: Task,
    intent: Intent | None,
    attempt: dict[str, Any],
    sql_reference_context: str,
    docs_context: str,
    *,
    aggregate_grain_guidance: str | None = None,
    metric_source_guidance: str | None = None,
) -> str:
    """Build a repair prompt using validation or execution feedback."""

    grounded_literals = (
        _grounded_literal_context_from_intent(intent) if intent is not None else None
    )
    grounded_literal_block = f"{grounded_literals}\n\n" if grounded_literals else ""

    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context}\n\n"
        f"Question: {task.question}\n\n"
        f"Failed SQL:\n{attempt['sql']}\n\n"
        f"Validation:\n{json.dumps(attempt['validation'], indent=2, sort_keys=True)}\n\n"
        f"Execution:\n{json.dumps(attempt['execution_result'], indent=2, sort_keys=True)}\n\n"
        f"{grounded_literal_block}"
        f"{_grain_guidance_block(aggregate_grain_guidance)}"
        f"{_metric_source_guidance_block(metric_source_guidance)}"
    )


def _critic_prompt(
    task: Task,
    intent: Intent,
    attempt: dict[str, Any],
    sql_reference_context: str,
    docs_context: str,
    *,
    metric_source_guidance: str | None = None,
) -> str:
    """Build the critic prompt using the current best SQL and result profile."""

    grounded_literals = _grounded_literal_context_from_intent(intent)
    grounded_literal_block = f"{grounded_literals}\n\n" if grounded_literals else ""
    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context}\n\n"
        f"Question: {task.question}\n\n"
        f"Answer contract:\n{intent.model_dump_json(indent=2)}\n\n"
        f"{grounded_literal_block}"
        f"{_metric_source_guidance_block(metric_source_guidance)}"
        f"SQL:\n{attempt['sql']}\n\n"
        "Candidate assumptions:\n"
        f"{json.dumps(attempt.get('assumptions', []), indent=2, sort_keys=True)}\n\n"
        "Candidate constraint ledger:\n"
        f"{json.dumps(attempt.get('constraint_ledger', []), indent=2, sort_keys=True)}\n\n"
        "Candidate unsupported assumptions:\n"
        f"{json.dumps(attempt.get('unsupported_assumptions', []), indent=2, sort_keys=True)}\n\n"
        "Execution result:\n"
        f"{json.dumps(attempt.get('execution_result', {}), indent=2, sort_keys=True)}\n\n"
        "Result profile:\n"
        f"{json.dumps(attempt.get('result_profile', {}), indent=2, sort_keys=True)}"
    )


def _aggregate_verification_prompt(
    task: Task,
    attempt: dict[str, Any],
    sql_reference_context: str,
    docs_context: str,
    *,
    reason: str,
) -> str:
    """Build the verification prompt for suspicious aggregate outputs."""

    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context}\n\n"
        f"Question: {task.question}\n\n"
        f"Suspicion: {reason}\n\n"
        f"SQL:\n{attempt['sql']}\n\n"
        "Execution result:\n"
        f"{json.dumps(attempt['execution_result'], indent=2, sort_keys=True)}\n\n"
        "Result profile:\n"
        f"{json.dumps(attempt.get('result_profile', {}), indent=2, sort_keys=True)}\n\n"
        "Check whether the aggregate output is plausible.\n"
        "If the result looks too small, inspect nearby value variants, filter selectivity, "
        "and the grain of the aggregation.\n"
        "Recommend repair only when the result is not trustworthy."
    )


def _candidate_comparison_prompt(
    task: Task,
    intent: Intent,
    attempts: list[dict[str, Any]],
    sql_reference_context: str,
    docs_context: str,
    *,
    aggregate_grain_guidance: str | None = None,
    metric_source_guidance: str | None = None,
    baseline_stage: str | None,
) -> str:
    """Build the comparison prompt for executable candidates."""

    comparison_candidates = [_comparison_attempt_summary(attempt) for attempt in attempts]
    grounded_literals = _grounded_literal_context_from_intent(intent)
    grounded_literal_block = f"{grounded_literals}\n\n" if grounded_literals else ""
    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context}\n\n"
        f"Question: {task.question}\n\n"
        f"Intent:\n{intent.model_dump_json(indent=2)}\n\n"
        f"{grounded_literal_block}"
        f"{_grain_guidance_block(aggregate_grain_guidance)}"
        f"{_metric_source_guidance_block(metric_source_guidance)}"
        f"Baseline stage: {baseline_stage or 'unknown'}\n\n"
        "Executable candidates:\n"
        f"{json.dumps(comparison_candidates, indent=2, sort_keys=True)}\n\n"
        "Compare every executable candidate above and choose the one "
        "that best fits the answer contract."
    )


def _aggregate_repair_prompt(
    task: Task,
    attempt: dict[str, Any],
    verification: ConfidenceReport,
    sql_reference_context: str,
    docs_context: str,
    intent: Intent | None = None,
) -> str:
    """Build the repair prompt after aggregate verification fails."""

    verification_json = json.dumps(verification.model_dump(mode="json"), indent=2, sort_keys=True)
    execution_json = json.dumps(attempt["execution_result"], indent=2, sort_keys=True)
    profile_json = json.dumps(attempt.get("result_profile", {}), indent=2, sort_keys=True)
    grounded_literals = (
        _grounded_literal_context_from_intent(intent) if intent is not None else None
    )
    grounded_literal_block = f"{grounded_literals}\n\n" if grounded_literals else ""
    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context}\n\n"
        f"Question: {task.question}\n\n"
        f"Current SQL:\n{attempt['sql']}\n\n"
        f"Verification:\n{verification_json}\n\n"
        f"Execution result:\n{execution_json}\n\n"
        f"Result profile:\n{profile_json}\n\n"
        f"{grounded_literal_block}"
    )


def _semantic_repair_prompt(
    task: Task,
    intent: Intent,
    attempt: dict[str, Any],
    critic: ConfidenceReport,
    sql_reference_context: str,
    docs_context: str,
    *,
    metric_source_guidance: str | None = None,
) -> str:
    """Build the repair prompt for one critic-triggered retry."""

    grounded_literals = _grounded_literal_context_from_intent(intent)
    grounded_literal_block = f"{grounded_literals}\n\n" if grounded_literals else ""
    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context}\n\n"
        f"Question: {task.question}\n\n"
        f"Current answer contract:\n{intent.model_dump_json(indent=2)}\n\n"
        f"{grounded_literal_block}"
        f"{_metric_source_guidance_block(metric_source_guidance)}"
        f"Current SQL:\n{attempt['sql']}\n\n"
        "Candidate assumptions:\n"
        f"{json.dumps(attempt.get('assumptions', []), indent=2, sort_keys=True)}\n\n"
        "Candidate constraint ledger:\n"
        f"{json.dumps(attempt.get('constraint_ledger', []), indent=2, sort_keys=True)}\n\n"
        "Candidate unsupported assumptions:\n"
        f"{json.dumps(attempt.get('unsupported_assumptions', []), indent=2, sort_keys=True)}\n\n"
        f"Critic issues:\n{json.dumps(critic.model_dump(mode='json'), indent=2, sort_keys=True)}"
    )
