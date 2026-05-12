"""Coordinate retrieval, LLM calls, validation, execution, and trace writing."""

from __future__ import annotations

import dataclasses
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from sol01.candidates.evaluator import _dataframe_records, evaluate_candidate
from sol01.candidates.scoring import _best_attempt
from sol01.candidates.verification import (
    _augment_intent_with_value_groundings,
    _table_schemas_for_selection,
)
from sol01.infra.config import DEFAULT_DOTENV_PATH, RuntimeConfig, SchemaRetrievalConfig
from sol01.infra.logging import get_logger
from sol01.llm.llm_logging import LLMCallLogger
from sol01.llm.prompt_builders import (
    _candidate_review_prompt,
    _question_preview,
    _retrieval_planning_user_prompt,
    _semantic_repair_prompt,
    _sql_generation_batch_prompt,
    _sql_reference_context,
    _sql_repair_prompt,
    enforce_prompt_budget,
    sanitize_hybrid_planning_decision,
    schema_expansion_trigger,
)
from sol01.models import (
    CandidateReviewReport,
    ConfidenceReport,
    ExecutionResult,
    FinalAnswer,
    HybridPlanningDecision,
    Intent,
    SchemaSelection,
    SelectedSchemaObject,
    SQLCandidate,
    SQLCandidateBatch,
    Task,
)
from sol01.output.output import (
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
from sol01.schema.hybrid_retrieval import retrieve_schema_objects
from sol01.schema.index import CACHE_PATH
from sol01.schema.resolver import resolve_schema_context
from sol01.schema.retrieval import load_db_index
from sol01.schema.retrieval_index import build_retrieval_index

logger = get_logger(__name__)
LLMClient: Any | None = None

__all__ = ["run_task", "run_tasks"]


def load_document_text(file_name: str) -> str:
    """Load task-linked document text without importing the docs module at startup."""

    from sol01.loading.docs import load_document_text as _load_document_text  # noqa: PLC0415

    return _load_document_text(file_name)


def _llm_client_class() -> Any:
    """Return the live LLM client class, importing it only when needed."""

    global LLMClient
    if LLMClient is None:
        from sol01.llm.client import LLMClient as _LLMClient  # noqa: PLC0415

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

    schema_retrieval_config = SchemaRetrievalConfig.from_env(dotenv_path=DEFAULT_DOTENV_PATH)
    _prewarm_schema_indexes(tasks, schema_retrieval_config=schema_retrieval_config)
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


def _prewarm_schema_indexes(
    tasks: list[Task],
    *,
    cache_path: Path = CACHE_PATH,
    schema_retrieval_config: SchemaRetrievalConfig,
) -> None:
    """Build each selected database retrieval index before workers start."""

    seen: set[str] = set()
    for task in tasks:
        if task.db in seen:
            continue
        seen.add(task.db)
        db_index = load_db_index(task.db, cache_path=cache_path)
        build_retrieval_index(
            task.db,
            db_index=db_index,
            config=schema_retrieval_config,
        )


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
    prompt_hashes: dict[str, str]
    schema_retrieval_version: str
    schema_retrieval_config: SchemaRetrievalConfig
    schema_retrieval: dict[str, Any]


def _check_skip(
    task: Task,
    run_paths: RunPaths,
    task_trace_path: Path,
    *,
    force: bool,
    skip_failed: bool,
    expected_schema_retrieval_version: str,
) -> FinalAnswer | None:
    """Return an existing FinalAnswer if the task should be skipped, else None."""

    if force or not should_skip_task(
        run_paths, instance_id=task.instance_id, skip_failed=skip_failed
    ):
        return None
    existing_trace = json.loads(task_trace_path.read_text(encoding="utf-8"))
    if existing_trace.get("schema_retrieval_version") != expected_schema_retrieval_version:
        logger.info(
            "task rerun: schema retrieval version changed",
            instance_id=task.instance_id,
            db=task.db,
            expected_schema_retrieval_version=expected_schema_retrieval_version,
            existing_schema_retrieval_version=existing_trace.get("schema_retrieval_version"),
        )
        return None
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
    schema_retrieval_config: SchemaRetrievalConfig,
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
    docs_context = (
        load_document_text(task.external_knowledge)
        if task.external_knowledge
        else "No task-linked document context."
    )
    schema, intent, table_schemas, sql_reference_context, schema_retrieval = _run_planning(
        task,
        client,
        prompt_hashes,
        docs_context,
        schema_retrieval_config=schema_retrieval_config,
    )
    intent = _augment_intent_with_value_groundings(
        intent, task=task, schema=schema, table_schemas=table_schemas
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
        sql_reference_context=sql_reference_context,
        docs_context=docs_context,
        prompt_hashes=prompt_hashes,
        schema_retrieval_version=schema_retrieval_config.schema_retrieval_version,
        schema_retrieval_config=schema_retrieval_config,
        schema_retrieval=schema_retrieval,
    )


def _run_planning(
    task: Task,
    client: StructuredLLM,
    prompt_hashes: dict[str, str],
    docs_context: str,
    *,
    schema_retrieval_config: SchemaRetrievalConfig,
) -> tuple[SchemaSelection, Intent, dict[str, Any], str, dict[str, Any]]:
    """Retrieve schema objects and run the retrieval-scoped planning call."""

    db_index = load_db_index(task.db)
    retrieval_index = build_retrieval_index(
        task.db,
        db_index=db_index,
        config=schema_retrieval_config,
    )
    linked_docs = [] if docs_context == "No task-linked document context." else [docs_context]
    retrieved_objects, retrieval_diagnostics = retrieve_schema_objects(
        retrieval_index,
        task.question,
        linked_docs=linked_docs,
        config=schema_retrieval_config,
    )
    planning_prompt = _build_planning_prompt(
        task,
        docs_context,
        retrieved_objects,
        schema_retrieval_config=schema_retrieval_config,
    )
    decision = _run_prompt(
        client,
        prompt_hashes=prompt_hashes,
        prompt_name="planning",
        output_type=HybridPlanningDecision,
        user_prompt=planning_prompt,
    )
    sanitized_decision, planner_diagnostics = sanitize_hybrid_planning_decision(
        decision,
        retrieved_objects,
    )
    resolved = resolve_schema_context(
        db=task.db,
        selected_objects=sanitized_decision.selected_objects,
        canonical_schema_objects=retrieval_index.objects,
        db_index=db_index,
        question=task.question,
        retrieval_evidence=retrieved_objects,
        constraints=sanitized_decision.constraints,
    )
    sql_reference_context = _checked_schema_prompt(
        "sql_reference_context",
        resolved.prompt_context,
        schema_retrieval_config,
    )
    prompt_budget = _prompt_budget_diagnostics(
        planning_prompt=planning_prompt,
        sql_reference_context=sql_reference_context,
        schema_retrieval_config=schema_retrieval_config,
    )

    schema = SchemaSelection(
        db=task.db,
        selected_object_ids=[
            selected.object_id for selected in sanitized_decision.selected_objects
        ],
        selected_tables=list(resolved.resolved_tables),
        expanded_tables=list(resolved.allowed_tables),
        allowed_tables=list(resolved.allowed_tables),
        rationale=sanitized_decision.rationale,
        confidence=sanitized_decision.confidence,
        diagnostics={
            "selection_prompt_chars": len(planning_prompt),
            "planning_prompt_chars": len(planning_prompt),
            "sql_reference_context_chars": len(sql_reference_context),
            "max_schema_prompt_chars": schema_retrieval_config.max_schema_prompt_chars,
            "prompt_budget": prompt_budget,
            "retrieved_object_count": len(retrieved_objects),
            "selected_objects": [
                selected.model_dump(mode="json") for selected in sanitized_decision.selected_objects
            ],
            "resolved_tables": list(resolved.resolved_tables),
            "allowed_tables": list(resolved.allowed_tables),
            "planner": planner_diagnostics,
            "resolver": resolved.diagnostics,
        },
    )
    schema_retrieval = {
        "index": {
            "db": retrieval_index.db,
            "cache_key": retrieval_index.cache_key,
            "object_count": len(retrieval_index.objects),
            "chunk_count": len(retrieval_index.chunks),
        },
        "retrieval": retrieval_diagnostics,
        "retrieved_objects": [
            {
                "object_id": item.schema_object.object_id,
                "object_type": item.schema_object.object_type,
                "name": item.schema_object.name,
                "table_name": item.schema_object.table_name,
                "rank": item.rank,
                "score": item.score,
            }
            for item in retrieved_objects
        ],
        "planner": planner_diagnostics,
        "resolver": resolved.diagnostics,
        "prompt_budget": prompt_budget,
    }
    return (
        schema,
        sanitized_decision.intent,
        resolved.table_schemas,
        sql_reference_context,
        schema_retrieval,
    )


def _build_planning_prompt(
    task: Task,
    docs_context: str,
    retrieved_objects: list[Any],
    *,
    schema_retrieval_config: SchemaRetrievalConfig,
) -> str:
    """Build a planning prompt that fits the configured schema prompt budget."""

    return _retrieval_planning_user_prompt(
        task,
        task.db,
        docs_context,
        retrieved_objects,
        max_docs_chars=schema_retrieval_config.max_linked_doc_chars,
        max_total_chars=schema_retrieval_config.max_schema_prompt_chars,
    )


def _checked_schema_prompt(
    prompt_name: str,
    prompt: str,
    schema_retrieval_config: SchemaRetrievalConfig,
) -> str:
    """Enforce the shared schema prompt budget for generated schema contexts."""

    return enforce_prompt_budget(
        prompt_name,
        prompt,
        schema_retrieval_config.max_schema_prompt_chars,
    )


def _prompt_budget_diagnostics(
    *,
    planning_prompt: str | None = None,
    sql_reference_context: str | None = None,
    schema_retrieval_config: SchemaRetrievalConfig,
) -> dict[str, object]:
    """Return trace fields for prompt budget enforcement."""

    diagnostics: dict[str, object] = {
        "max_schema_prompt_chars": schema_retrieval_config.max_schema_prompt_chars,
    }
    if planning_prompt is not None:
        planning_chars = len(planning_prompt)
        diagnostics.update(
            {
                "planning_prompt_chars": planning_chars,
                "planning_prompt_within_budget": (
                    planning_chars <= schema_retrieval_config.max_schema_prompt_chars
                ),
            }
        )
    if sql_reference_context is not None:
        context_chars = len(sql_reference_context)
        diagnostics.update(
            {
                "sql_reference_context_chars": context_chars,
                "sql_reference_context_within_budget": (
                    context_chars <= schema_retrieval_config.max_schema_prompt_chars
                ),
            }
        )
    return diagnostics


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
    candidate_limit = min(initial_candidates, max_attempts - len(attempts))
    if candidate_limit <= 0:
        return _best_attempt(attempts)

    batch = _run_prompt(
        ctx.client,
        prompt_hashes=ctx.prompt_hashes,
        prompt_name="sql_generation_batch",
        output_type=SQLCandidateBatch,
        user_prompt=_sql_generation_batch_prompt(
            ctx.task,
            ctx.intent,
            ctx.sql_reference_context,
            ctx.docs_context,
            candidate_count=candidate_limit,
        ),
    )
    for candidate_index, candidate in enumerate(batch.candidates[:candidate_limit]):
        stage = f"initial_{candidate_index + 1}"
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


def _review_and_repair(
    ctx: _TaskCtx,
    attempts: list[dict[str, Any]],
    best_attempt: dict[str, Any] | None,
    *,
    max_attempts: int,
    semantic_repairs: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Ask the model to adjudicate executable attempts using local observations."""

    if best_attempt is None or not best_attempt["execution_result"]["ok"]:
        return best_attempt, None

    executable_attempts = [attempt for attempt in attempts if attempt["execution_result"]["ok"]]
    review_reason = _candidate_review_reason(executable_attempts)

    logger.info(
        "candidate review requested",
        instance_id=ctx.task.instance_id,
        best_stage=best_attempt["stage"],
        executable_attempts=len(executable_attempts),
        reason=review_reason,
    )
    review = _run_prompt(
        ctx.client,
        prompt_hashes=ctx.prompt_hashes,
        prompt_name="candidate_review",
        output_type=CandidateReviewReport,
        user_prompt=_candidate_review_prompt(
            ctx.task,
            ctx.intent,
            executable_attempts,
            ctx.sql_reference_context,
            ctx.docs_context,
            baseline_stage=best_attempt["stage"],
            review_reason=review_reason,
        ),
    )
    review_payload = {
        "review_reason": review_reason,
        **review.model_dump(mode="json"),
    }
    if review.preferred_stage:
        preferred_attempt = next(
            (
                attempt
                for attempt in executable_attempts
                if attempt["stage"] == review.preferred_stage
            ),
            None,
        )
        if preferred_attempt is not None:
            best_attempt = preferred_attempt

    best_attempt["critic"] = {
        "confidence": review.confidence,
        "issues": review.issues,
        "should_repair": review.should_repair,
        "repair_focus": review.repair_focus,
    }
    best_attempt["candidate_review"] = review_payload
    logger.info(
        "candidate review complete",
        instance_id=ctx.task.instance_id,
        preferred_stage=review.preferred_stage,
        should_repair=review.should_repair,
        confidence=review.confidence,
        issues=review.issues,
    )

    if not review.should_repair or semantic_repairs < 1:
        return best_attempt, review_payload

    budget_exhausted = len(attempts) >= max_attempts
    if budget_exhausted:
        best_attempt["repair_skipped_reason"] = "attempt budget exhausted"
        logger.info(
            "critic repair skipped",
            instance_id=ctx.task.instance_id,
            reason="attempt budget exhausted",
        )
        return best_attempt, review_payload

    logger.info(
        "semantic repair requested",
        instance_id=ctx.task.instance_id,
        stage="critic_repair",
        focus=review.repair_focus,
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
            ConfidenceReport(
                confidence=review.confidence,
                issues=review.issues,
                should_repair=review.should_repair,
                repair_focus=review.repair_focus,
            ),
            ctx.sql_reference_context,
            ctx.docs_context,
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

    new_best = attempt if attempt["execution_result"]["ok"] else _best_attempt(attempts)
    return new_best, review_payload


def _candidate_review_reason(executable_attempts: list[dict[str, Any]]) -> str:
    """Return the trace reason for model-led final candidate adjudication."""

    if len(executable_attempts) == 1:
        return "final adjudication of the only executable candidate"
    return "final adjudication across executable candidates using local observations"


def _rebuild_context_for_expansion(
    ctx: _TaskCtx,
    expanded_schema: SchemaSelection,
    *,
    table_schemas: dict[str, Any] | None = None,
    sql_reference_context: str | None = None,
    schema_retrieval: dict[str, Any] | None = None,
) -> _TaskCtx:
    """Rebuild selected-schema context while preserving the original answer contract."""

    new_table_schemas = table_schemas or _table_schemas_for_selection(expanded_schema)
    reference_context = sql_reference_context or _sql_reference_context(
        expanded_schema,
        new_table_schemas,
    )
    reference_context = _checked_schema_prompt(
        "schema_expansion_sql_reference_context",
        reference_context,
        ctx.schema_retrieval_config,
    )
    intent = _augment_intent_with_value_groundings(
        ctx.intent, task=ctx.task, schema=expanded_schema, table_schemas=new_table_schemas
    )
    return _TaskCtx(
        task=ctx.task,
        client=ctx.client,
        intent=intent,
        schema=expanded_schema,
        table_schemas=new_table_schemas,
        sql_reference_context=reference_context,
        docs_context=ctx.docs_context,
        prompt_hashes=ctx.prompt_hashes,
        schema_retrieval_version=ctx.schema_retrieval_version,
        schema_retrieval_config=ctx.schema_retrieval_config,
        schema_retrieval=schema_retrieval or ctx.schema_retrieval,
    )


def _attempt_schema_expansion(
    ctx: _TaskCtx,
    attempts: list[dict[str, Any]],
    best_attempt: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, _TaskCtx | None]:
    """Run one schema-expansion recovery attempt when evidence warrants it.

    Returns (updated_best_attempt, schema_expansion_payload, expanded_ctx).
    expanded_ctx is non-None only when new tables were actually added.  The
    expansion attempt is outside the normal max_attempts budget.
    """

    if best_attempt is None:
        return best_attempt, None, None

    trigger = schema_expansion_trigger(best_attempt)
    if trigger is None:
        return best_attempt, None, None

    logger.info(
        "schema expansion triggered",
        instance_id=ctx.task.instance_id,
        trigger=trigger,
    )

    try:
        db_index = load_db_index(ctx.task.db)
    except Exception:
        logger.warning(
            "schema expansion skipped: db_index unavailable",
            instance_id=ctx.task.instance_id,
            exc_info=True,
        )
        return best_attempt, None, None

    expansion_payload: dict[str, Any] = {
        "trigger": trigger,
        "expansion_query": _schema_expansion_query(ctx, best_attempt, trigger),
        "decision": None,
        "added_tables": [],
        "outcome": "no_new_tables",
    }
    retrieval_index = build_retrieval_index(
        ctx.task.db,
        db_index=db_index,
        config=ctx.schema_retrieval_config,
    )

    deterministic_tables = _deterministic_expansion_tables(best_attempt, ctx.schema, db_index)

    if deterministic_tables:
        expansion_payload["decision"] = {
            "source": "exact_name",
            "additional_tables": deterministic_tables,
        }
        selected_additions = [
            SelectedSchemaObject(
                object_id=f"table:{table_name}",
                role="primary",
                reason="unambiguous table name in schema error",
            )
            for table_name in deterministic_tables
        ]
        retrieved_objects: list[Any] = []
        planner_diagnostics: dict[str, object] = {}
    else:
        selected_additions, retrieved_objects, planner_diagnostics = _retrieve_expansion_objects(
            ctx,
            best_attempt,
            trigger,
            retrieval_index=retrieval_index,
            expansion_query=expansion_payload["expansion_query"],
        )
        expansion_payload["decision"] = {
            "source": "retrieval",
            "selected_object_ids": [selected.object_id for selected in selected_additions],
        }
        logger.info(
            "schema expansion retrieval complete",
            instance_id=ctx.task.instance_id,
            retrieved_objects=len(retrieved_objects),
            selected_additions=len(selected_additions),
        )

    expanded_schema, resolved, added_tables = _resolve_expanded_schema(
        ctx,
        selected_additions,
        retrieval_index=retrieval_index,
        db_index=db_index,
        retrieval_evidence=retrieved_objects,
        expansion_query=expansion_payload["expansion_query"],
    )
    expansion_payload.update(
        {
            "retrieved_objects": _retrieved_object_trace(retrieved_objects),
            "selected_additions": [
                selected.model_dump(mode="json") for selected in selected_additions
            ],
            "added_tables": added_tables,
            "planner": planner_diagnostics,
            "resolver": resolved.diagnostics,
            "prompt_budget": _prompt_budget_diagnostics(
                sql_reference_context=resolved.prompt_context,
                schema_retrieval_config=ctx.schema_retrieval_config,
            ),
        }
    )
    if not added_tables:
        expansion_payload["outcome"] = "no_new_tables"
        logger.info(
            "schema expansion skipped: no new valid tables",
            instance_id=ctx.task.instance_id,
            requested=[selected.object_id for selected in selected_additions],
        )
        return best_attempt, expansion_payload, None

    logger.info(
        "schema expanding",
        instance_id=ctx.task.instance_id,
        added_tables=added_tables,
    )

    expanded_ctx = _rebuild_context_for_expansion(
        ctx,
        expanded_schema,
        table_schemas=resolved.table_schemas,
        sql_reference_context=resolved.prompt_context,
    )

    expansion_stage = "schema_expansion"
    expansion_payload["expansion_attempt_stage"] = expansion_stage
    logger.info(
        "candidate request",
        instance_id=ctx.task.instance_id,
        stage=expansion_stage,
        prompt_name="sql_generation_batch",
    )
    batch = _run_prompt(
        expanded_ctx.client,
        prompt_hashes=expanded_ctx.prompt_hashes,
        prompt_name="sql_generation_batch",
        output_type=SQLCandidateBatch,
        user_prompt=_sql_generation_batch_prompt(
            expanded_ctx.task,
            expanded_ctx.intent,
            expanded_ctx.sql_reference_context,
            expanded_ctx.docs_context,
            candidate_count=1,
        ),
    )
    if not batch.candidates:
        expansion_payload["outcome"] = "expanded_no_candidate"
        expanded_ctx.schema_retrieval = _schema_retrieval_with_expansion(
            ctx.schema_retrieval,
            expansion_payload,
        )
        return _best_attempt(attempts), expansion_payload, expanded_ctx
    expansion_attempt = evaluate_candidate(
        task=expanded_ctx.task,
        candidate=batch.candidates[0],
        intent=expanded_ctx.intent,
        schema=expanded_ctx.schema,
        table_schemas=expanded_ctx.table_schemas,
        stage=expansion_stage,
    )
    attempts.append(expansion_attempt)
    _log_candidate(ctx.task.instance_id, expansion_attempt)

    if expansion_attempt["execution_result"]["ok"]:
        expansion_payload["outcome"] = "expanded"
    else:
        expansion_payload["outcome"] = "expanded_failed"
    expanded_ctx.schema_retrieval = _schema_retrieval_with_expansion(
        ctx.schema_retrieval,
        expansion_payload,
    )

    return _best_attempt(attempts), expansion_payload, expanded_ctx


def _retrieve_expansion_objects(
    ctx: _TaskCtx,
    attempt: dict[str, Any],
    trigger: str,
    *,
    retrieval_index: Any,
    expansion_query: str,
) -> tuple[list[SelectedSchemaObject], list[Any], dict[str, object]]:
    """Retrieve and sanitize schema objects for one expansion attempt."""

    linked_docs = (
        [] if ctx.docs_context == "No task-linked document context." else [ctx.docs_context]
    )
    retrieved_objects, retrieval_diagnostics = retrieve_schema_objects(
        retrieval_index,
        expansion_query,
        linked_docs=linked_docs,
        config=ctx.schema_retrieval_config,
    )
    expansion_task = ctx.task.model_copy(
        update={
            "question": (
                f"{ctx.task.question}\n\n"
                "Schema expansion evidence:\n"
                f"{_schema_expansion_query(ctx, attempt, trigger)}"
            )
        }
    )
    planning_prompt = _build_planning_prompt(
        expansion_task,
        ctx.docs_context,
        retrieved_objects,
        schema_retrieval_config=ctx.schema_retrieval_config,
    )
    decision = _run_prompt(
        ctx.client,
        prompt_hashes=ctx.prompt_hashes,
        prompt_name="planning",
        output_type=HybridPlanningDecision,
        user_prompt=planning_prompt,
    )
    sanitized_decision, planner_diagnostics = sanitize_hybrid_planning_decision(
        decision,
        retrieved_objects,
    )
    current_object_ids = set(ctx.schema.selected_object_ids)
    selected_additions = [
        selected
        for selected in sanitized_decision.selected_objects
        if selected.object_id not in current_object_ids
    ]
    diagnostics = {
        "retrieval": retrieval_diagnostics,
        "planner": planner_diagnostics,
        "prompt_budget": _prompt_budget_diagnostics(
            planning_prompt=planning_prompt,
            schema_retrieval_config=ctx.schema_retrieval_config,
        ),
        "rationale": sanitized_decision.rationale,
        "confidence": sanitized_decision.confidence,
    }
    return selected_additions, retrieved_objects, diagnostics


def _resolve_expanded_schema(
    ctx: _TaskCtx,
    selected_additions: list[SelectedSchemaObject],
    *,
    retrieval_index: Any,
    db_index: dict[str, Any],
    retrieval_evidence: list[Any],
    expansion_query: str,
) -> tuple[SchemaSelection, Any, list[str]]:
    """Resolve current and newly selected objects into one compact schema context."""

    object_ids = {schema_object.object_id for schema_object in retrieval_index.objects}
    selected_objects: list[SelectedSchemaObject] = []
    seen: set[str] = set()
    for object_id in ctx.schema.selected_object_ids:
        if object_id in object_ids and object_id not in seen:
            selected_objects.append(SelectedSchemaObject(object_id=object_id, role="unknown"))
            seen.add(object_id)
    for selected in selected_additions:
        if selected.object_id in object_ids and selected.object_id not in seen:
            selected_objects.append(selected)
            seen.add(selected.object_id)

    resolved = resolve_schema_context(
        db=ctx.task.db,
        selected_objects=selected_objects,
        canonical_schema_objects=retrieval_index.objects,
        db_index=db_index,
        question=ctx.task.question,
        retrieval_evidence=retrieval_evidence,
    )
    current_tables = set(ctx.schema.expanded_tables)
    added_tables = [table for table in resolved.allowed_tables if table not in current_tables]
    expanded_schema = ctx.schema.model_copy(
        update={
            "selected_object_ids": [selected.object_id for selected in selected_objects],
            "selected_tables": list(resolved.resolved_tables),
            "expanded_tables": list(resolved.allowed_tables),
            "allowed_tables": list(resolved.allowed_tables),
            "diagnostics": {
                **ctx.schema.diagnostics,
                "schema_expansion": {
                    "expansion_query": expansion_query,
                    "selected_additions": [
                        selected.model_dump(mode="json") for selected in selected_additions
                    ],
                    "added_tables": added_tables,
                    "resolver_warnings": resolved.diagnostics.get("warnings", []),
                },
            },
        }
    )
    return expanded_schema, resolved, added_tables


def _schema_expansion_query(
    ctx: _TaskCtx,
    attempt: dict[str, Any],
    trigger: str,
) -> str:
    """Build the augmented retrieval query used to find expansion candidates."""

    validation = attempt.get("validation", {})
    execution = attempt.get("execution_result", {})
    parts = [
        f"Original question: {ctx.task.question}",
        f"Schema expansion trigger: {trigger}",
        f"Failed SQL: {attempt.get('sql') or ''}",
        f"Validation errors: {'; '.join(validation.get('errors', [])) or 'none'}",
        f"Validation warnings: {'; '.join(validation.get('warnings', [])) or 'none'}",
        f"Execution error: {execution.get('error') or 'none'}",
        "Current selected object ids: " + ", ".join(ctx.schema.selected_object_ids),
        "Current allowed tables: " + ", ".join(ctx.schema.expanded_tables),
    ]
    return "\n".join(parts)


def _deterministic_expansion_tables(
    attempt: dict[str, Any],
    schema: SchemaSelection,
    db_index: dict[str, Any],
) -> list[str]:
    """Return tables named unambiguously by validation or execution errors."""

    named_tables = _table_names_from_schema_errors(attempt)
    current = set(schema.expanded_tables)
    selected: list[str] = []
    for name in named_tables:
        table_name = _unambiguous_table_name(name, db_index)
        if table_name is None or table_name in current or table_name in selected:
            continue
        selected.append(table_name)
    return selected


def _table_names_from_schema_errors(attempt: dict[str, Any]) -> list[str]:
    """Extract table-like names explicitly reported by validation or execution."""

    names: list[str] = []
    validation = attempt.get("validation", {})
    for error in validation.get("errors", []):
        match = re.search(r"Unknown table referenced:\s*([A-Za-z0-9_.$\"]+)", error)
        if match:
            names.append(match.group(1).strip('".'))

    execution_error = str(attempt.get("execution_result", {}).get("error") or "")
    if execution_error:
        quoted_names = re.findall(r"['\"]([A-Za-z0-9_.$]+)['\"]", execution_error)
        names.extend(match.strip('"') for match in quoted_names)
        dotted_names = re.findall(
            r"\b[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+){1,2}\b",
            execution_error,
        )
        names.extend(dotted_names)
    return list(dict.fromkeys(names))


def _unambiguous_table_name(raw_name: str, db_index: dict[str, Any]) -> str | None:
    """Map one raw table name to a canonical db_index table only when unique."""

    normalized = raw_name.strip().strip('"').lower()
    if not normalized:
        return None
    matches = {
        table_name for table_name in db_index if normalized in _table_name_aliases(table_name)
    }
    if len(matches) != 1:
        return None
    return next(iter(matches))


def _table_name_aliases(table_name: str) -> list[str]:
    """Return exact normalized aliases for a fully qualified table name."""

    parts = [part for part in table_name.lower().split(".") if part]
    aliases = {parts[-1]} if parts else set()
    for start in range(len(parts)):
        aliases.add(".".join(parts[start:]))
    return sorted(aliases, key=len, reverse=True)


def _retrieved_object_trace(retrieved_objects: list[Any]) -> list[dict[str, object]]:
    """Render expansion retrieval hits in the task trace."""

    return [
        {
            "object_id": item.schema_object.object_id,
            "object_type": item.schema_object.object_type,
            "name": item.schema_object.name,
            "table_name": item.schema_object.table_name,
            "rank": item.rank,
            "score": item.score,
        }
        for item in retrieved_objects
    ]


def _schema_retrieval_with_expansion(
    schema_retrieval: dict[str, Any],
    expansion_payload: dict[str, Any],
) -> dict[str, Any]:
    """Attach expansion diagnostics to the schema retrieval trace block."""

    updated = dict(schema_retrieval)
    expansions = list(updated.get("expansions", []))
    expansions.append(
        {
            "expansion_query": expansion_payload.get("expansion_query"),
            "retrieved_objects": expansion_payload.get("retrieved_objects", []),
            "selected_additions": expansion_payload.get("selected_additions", []),
            "added_tables": expansion_payload.get("added_tables", []),
            "planner": expansion_payload.get("planner", {}),
            "resolver": expansion_payload.get("resolver", {}),
            "prompt_budget": expansion_payload.get("prompt_budget", {}),
            "outcome": expansion_payload.get("outcome"),
        }
    )
    updated["expansions"] = expansions
    return updated


def _write_task_output(
    ctx: _TaskCtx,
    attempts: list[dict[str, Any]],
    best_attempt: dict[str, Any] | None,
    run_paths: RunPaths,
    task_trace_path: Path,
    task_llm_log_path: Path,
    *,
    candidate_review_payload: dict[str, Any] | None,
    schema_expansion_payload: dict[str, Any] | None,
    expanded_ctx: _TaskCtx | None,
    live_logging_enabled: bool,
    started_at: float,
) -> FinalAnswer:
    """Write final SQL, CSV, and trace; return the FinalAnswer."""

    task = ctx.task
    final_ctx = expanded_ctx if expanded_ctx is not None else ctx
    trace_payload: dict[str, Any] = {
        "instance_id": task.instance_id,
        "db": task.db,
        "question": task.question,
        "schema_selection": final_ctx.schema.model_dump(mode="json"),
        "schema_retrieval_version": final_ctx.schema_retrieval_version,
        "schema_retrieval": final_ctx.schema_retrieval,
        "intent": final_ctx.intent.model_dump(mode="json"),
        "prompt_hashes": ctx.prompt_hashes,
        "attempts": attempts,
    }
    if candidate_review_payload is not None:
        trace_payload["candidate_review"] = candidate_review_payload
    if schema_expansion_payload is not None:
        trace_payload["schema_expansion"] = schema_expansion_payload
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
    schema_retrieval_config = SchemaRetrievalConfig.from_env(dotenv_path=DEFAULT_DOTENV_PATH)

    skipped = _check_skip(
        task,
        run_paths,
        task_trace_path,
        force=force,
        skip_failed=skip_failed,
        expected_schema_retrieval_version=schema_retrieval_config.schema_retrieval_version,
    )
    if skipped is not None:
        return skipped

    prompt_hashes: dict[str, str] = {}
    attempts: list[dict[str, Any]] = []

    ctx = _build_context(
        task,
        client,
        prompt_hashes,
        run_paths,
        schema_retrieval_config,
    )

    best_attempt = _generate_initial_candidates(
        ctx, attempts, initial_candidates=initial_candidates, max_attempts=max_attempts
    )
    best_attempt = _repair_failed_execution(ctx, attempts, best_attempt, max_attempts=max_attempts)
    best_attempt, candidate_review_payload = _review_and_repair(
        ctx, attempts, best_attempt, max_attempts=max_attempts, semantic_repairs=semantic_repairs
    )
    best_attempt, schema_expansion_payload, expanded_ctx = _attempt_schema_expansion(
        ctx, attempts, best_attempt
    )

    return _write_task_output(
        ctx,
        attempts,
        best_attempt,
        run_paths,
        task_trace_path,
        task_llm_log_path,
        candidate_review_payload=candidate_review_payload,
        schema_expansion_payload=schema_expansion_payload,
        expanded_ctx=expanded_ctx,
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
