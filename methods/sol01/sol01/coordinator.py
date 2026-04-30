"""Coordinate retrieval, LLM calls, validation, execution, and trace writing."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from numbers import Number
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from sol01.config import RuntimeConfig
from sol01.docs import load_document_text
from sol01.index import CACHE_PATH
from sol01.llm import LLMClient
from sol01.llm_logging import LLMCallLogger
from sol01.logging import get_logger
from sol01.models import (
    AggregateGrainReport,
    CandidateComparisonReport,
    ConfidenceReport,
    ExecutionResult,
    FinalAnswer,
    Intent,
    SchemaSelection,
    SQLCandidate,
    TableSchema,
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
from sol01.profiling import profile_dataframe
from sol01.retrieval import load_db_index, retrieve_schema
from sol01.snowflake_runner import _dataframe_records, fetch_query_dataframe
from sol01.validation import validate_sql

logger = get_logger(__name__)


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
    client = llm_client or LLMClient(
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
    intent = _run_prompt(
        client,
        prompt_hashes=prompt_hashes,
        prompt_name="intent",
        output_type=Intent,
        user_prompt=_intent_user_prompt(task, schema),
    )
    logger.info(
        "intent extracted",
        instance_id=task.instance_id,
        metrics=intent.metrics,
        filters=intent.filters,
        time_constraints=intent.time_constraints,
    )

    table_schemas = _table_schemas_for_selection(schema)
    sql_reference_context = _sql_reference_context(schema, table_schemas)
    if task.external_knowledge:
        docs_context = load_document_text(task.external_knowledge)
    else:
        docs_context = "No task-linked document context."

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
            ),
        )
        attempt = _evaluate_candidate(
            task=task,
            candidate=candidate,
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
            user_prompt=_sql_repair_prompt(task, best_attempt, sql_reference_context, docs_context),
        )
        attempt = _evaluate_candidate(
            task=task,
            candidate=repaired_candidate,
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
                    ),
                )
                attempt = _evaluate_candidate(
                    task=task,
                    candidate=repaired_candidate,
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
            user_prompt=_critic_prompt(task, best_attempt, sql_reference_context),
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
                    best_attempt,
                    critic,
                    sql_reference_context,
                    docs_context,
                ),
            )
            attempt = _evaluate_candidate(
                task=task,
                candidate=repaired_candidate,
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
        "candidate_confidence": candidate.confidence,
        "validation": validation.model_dump(mode="json"),
        "execution_result": execution.model_dump(mode="json"),
        "score": _attempt_score(
            candidate=candidate,
            validation=validation,
            execution=execution,
            aggregate_grain=aggregate_grain,
        ),
    }

    if execution.ok:
        attempt["result_profile"] = profile_dataframe(dataframe)
        attempt["_dataframe"] = dataframe
    if aggregate_grain is not None:
        attempt["aggregate_grain"] = aggregate_grain.model_dump(mode="json")
    attempt["elapsed_seconds"] = round(perf_counter() - started_at, 3)

    return attempt


def _attempt_score(
    *,
    candidate: SQLCandidate,
    validation: Any,
    execution: Any,
    aggregate_grain: AggregateGrainReport | None = None,
) -> float:
    """Prefer successfully executed candidates, then valid ones, then candidate confidence."""

    score = candidate.confidence
    if validation.ok:
        score += 10.0
    if execution.ok:
        score += 100.0
        if execution.columns:
            score += 2.0
        if execution.row_count > 0:
            score += 1.0
    score += _aggregate_grain_adjustment(aggregate_grain)
    return score


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
        "candidate_confidence": attempt["candidate_confidence"],
        "score": attempt["score"],
        "validation": attempt["validation"],
        "execution_result": attempt["execution_result"],
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
        table_schemas.get(table_name)
        or TableSchema(name=table_name, ddl="", searchable_text="")
        for table_name in schema.expanded_tables
    ]
    entity_like_tables = [table for table in selected_tables if _table_is_entity_like(table)]
    question_text = task.question.lower()
    wants_unique = any(
        keyword in question_text
        for keyword in ("unique", "distinct", "dedupe", "dedup", "without duplicates")
    )
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
        return -2.5 if report.uses_distinct else 0.75
    if report.inferred_grain == "distinct_entity_count":
        return 0.75 if report.uses_distinct else -1.5
    if report.inferred_grain == "value_count":
        return 0.25
    return 0.0


def _table_is_entity_like(table: TableSchema) -> bool:
    """Return True when a table looks like a master/entity table."""

    table_name = (table.full_name or table.name).split(".")[-1].lower()
    return (
        table_name.startswith(("mst_", "dim_", "ref_", "lkp_"))
        or table_name.endswith(("_users", "_user"))
        or "master" in table_name
        or "entity" in table_name
    )


def _count_distinct_target(sql: str) -> str | None:
    """Extract the DISTINCT target column from a COUNT aggregate."""

    match = re.search(r"count\s*\(\s*distinct\s+([^)]+)\)", sql, flags=re.IGNORECASE)
    if match is None:
        return None
    target = match.group(1).strip()
    target = re.sub(r"\s+as\s+\w+$", "", target, flags=re.IGNORECASE)
    return target.strip(' "\'`')


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

    normalized = " ".join(sql.split())
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


def _intent_user_prompt(task: Task, schema: SchemaSelection) -> str:
    """Build the user prompt for intent extraction."""

    return f"Question: {task.question}\n\nSchema context:\n{_schema_context(schema)}"


def _sql_generation_prompt(
    task: Task,
    intent: Intent,
    sql_reference_context: str,
    docs_context: str,
) -> str:
    """Build the SQL-generation prompt body."""

    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context}\n\n"
        f"Question: {task.question}\n\n"
        f"Intent:\n{intent.model_dump_json(indent=2)}\n\n"
        "Write the SQL using only the reference context above."
    )


def _sql_repair_prompt(
    task: Task,
    attempt: dict[str, Any],
    sql_reference_context: str,
    docs_context: str,
) -> str:
    """Build a repair prompt using validation or execution feedback."""

    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context}\n\n"
        f"Question: {task.question}\n\n"
        f"Failed SQL:\n{attempt['sql']}\n\n"
        f"Validation:\n{json.dumps(attempt['validation'], indent=2, sort_keys=True)}\n\n"
        f"Execution:\n{json.dumps(attempt['execution_result'], indent=2, sort_keys=True)}"
    )


def _critic_prompt(task: Task, attempt: dict[str, Any], sql_reference_context: str) -> str:
    """Build the critic prompt using the current best SQL and result profile."""

    return (
        f"{sql_reference_context}\n\n"
        f"Question: {task.question}\n\n"
        f"SQL:\n{attempt['sql']}\n\n"
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
    baseline_stage: str | None,
) -> str:
    """Build the comparison prompt for executable candidates."""

    comparison_candidates = [_comparison_attempt_summary(attempt) for attempt in attempts]
    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context}\n\n"
        f"Question: {task.question}\n\n"
        f"Intent:\n{intent.model_dump_json(indent=2)}\n\n"
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
) -> str:
    """Build the repair prompt after aggregate verification fails."""

    verification_json = json.dumps(verification.model_dump(mode="json"), indent=2, sort_keys=True)
    execution_json = json.dumps(attempt["execution_result"], indent=2, sort_keys=True)
    profile_json = json.dumps(attempt.get("result_profile", {}), indent=2, sort_keys=True)
    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context}\n\n"
        f"Question: {task.question}\n\n"
        f"Current SQL:\n{attempt['sql']}\n\n"
        f"Verification:\n{verification_json}\n\n"
        f"Execution result:\n{execution_json}\n\n"
        f"Result profile:\n{profile_json}"
    )


def _semantic_repair_prompt(
    task: Task,
    attempt: dict[str, Any],
    critic: ConfidenceReport,
    sql_reference_context: str,
    docs_context: str,
) -> str:
    """Build the repair prompt for one critic-triggered retry."""

    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context}\n\n"
        f"Question: {task.question}\n\n"
        f"Current SQL:\n{attempt['sql']}\n\n"
        f"Critic issues:\n{json.dumps(critic.model_dump(mode='json'), indent=2, sort_keys=True)}"
    )
