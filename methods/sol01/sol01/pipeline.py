"""Per-task solver pipeline stages used by coordinator.run_task."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel

from sol01.candidates.evaluator import evaluate_candidate
from sol01.candidates.selection import final_winner_reason, select_winner
from sol01.candidates.verification import (
    augment_intent_with_value_groundings,
    table_schemas_for_selection,
)
from sol01.execution.snowflake_runner import dataframe_records
from sol01.infra.config import DEFAULT_SCHEMA_CONTEXT_VERSION, SchemaContextConfig
from sol01.infra.logging import get_logger
from sol01.infra.strings import question_preview
from sol01.llm.client import LLMClient
from sol01.llm.prompt_builders import (
    candidate_review_prompt,
    enforce_prompt_budget,
    sanitize_schema_planning_decision,
    schema_context_planning_user_prompt,
    schema_expansion_trigger,
    semantic_repair_prompt,
    sql_generation_batch_prompt,
    sql_reference_context,
    sql_repair_prompt,
)
from sol01.loading.docs import load_document_text
from sol01.models import (
    AttemptRecord,
    CandidateReviewReport,
    ConfidenceReport,
    ExecutionResult,
    FinalAnswer,
    Intent,
    SchemaPlanningDecision,
    SchemaSelection,
    SelectedSchemaObject,
    SQLCandidate,
    SQLCandidateBatch,
    Task,
)
from sol01.output.output import (
    RunPaths,
    csv_path_for,
    should_skip_task,
    write_sql,
    write_trace,
)
from sol01.schema.db_index import load_db_index
from sol01.schema.expansion import (
    deterministic_expansion_tables,
    resolve_expanded_schema,
    schema_context_object_trace,
    schema_context_with_expansion,
    schema_expansion_query,
)
from sol01.schema.resolver import resolve_schema_context
from sol01.schema.schema_context import select_schema_context_objects
from sol01.schema.schema_context_cache import build_schema_context_cache

logger = get_logger(__name__)


def _winner_attempt(
    attempts: list[AttemptRecord],
    *,
    preferred_stage: str | None = None,
) -> AttemptRecord | None:
    sel = select_winner(attempts, preferred_stage=preferred_stage)
    return sel.attempt if sel is not None else None


# ---------------------------------------------------------------------------
# Per-task pipeline stages
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TaskContext:
    """Shared context threaded through each pipeline stage of run_task."""

    task: Task
    client: LLMClient
    intent: Intent
    schema: SchemaSelection
    table_schemas: dict[str, Any]
    sql_reference_context: str
    docs_context: str | None
    prompt_hashes: dict[str, str]
    schema_context_version: str
    schema_context_config: SchemaContextConfig
    schema_context: dict[str, Any]


def check_skip(
    task: Task,
    run_paths: RunPaths,
    task_trace_path: Path,
    *,
    force: bool,
    skip_failed: bool,
    expected_schema_context_version: str,
) -> FinalAnswer | None:
    """Return an existing FinalAnswer if the task should be skipped, else None."""

    if force or not should_skip_task(
        run_paths, instance_id=task.instance_id, skip_failed=skip_failed
    ):
        return None
    existing_trace = json.loads(task_trace_path.read_text(encoding="utf-8"))
    if existing_trace.get("schema_context_version") != expected_schema_context_version:
        logger.info(
            "task rerun: schema context version changed",
            instance_id=task.instance_id,
            db=task.db,
            expected_schema_context_version=expected_schema_context_version,
            existing_schema_context_version=existing_trace.get("schema_context_version"),
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


def build_context(
    task: Task,
    client: LLMClient,
    prompt_hashes: dict[str, str],
    run_paths: RunPaths,
    schema_context_config: SchemaContextConfig,
) -> TaskContext:
    """Build schema context, extract intent, and assemble the shared pipeline context."""

    logger.info(
        "task start",
        instance_id=task.instance_id,
        db=task.db,
        question_preview=question_preview(task.question),
        question_length=len(task.question),
        run_root=str(run_paths.root),
    )
    docs_context = load_document_text(task.external_knowledge) if task.external_knowledge else None
    schema, intent, table_schemas, sql_reference_context, schema_context = run_planning(
        task,
        client,
        prompt_hashes,
        docs_context,
        schema_context_config=schema_context_config,
    )
    intent = augment_intent_with_value_groundings(
        intent, task=task, schema=schema, table_schemas=table_schemas
    )
    logger.info(
        "schema selected",
        instance_id=task.instance_id,
        expanded_count=len(schema.expanded_tables),
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
    return TaskContext(
        task=task,
        client=client,
        intent=intent,
        schema=schema,
        table_schemas=table_schemas,
        sql_reference_context=sql_reference_context,
        docs_context=docs_context,
        prompt_hashes=prompt_hashes,
        schema_context_version=DEFAULT_SCHEMA_CONTEXT_VERSION,
        schema_context_config=schema_context_config,
        schema_context=schema_context,
    )


def run_planning(
    task: Task,
    client: LLMClient,
    prompt_hashes: dict[str, str],
    docs_context: str | None,
    *,
    schema_context_config: SchemaContextConfig,
) -> tuple[SchemaSelection, Intent, dict[str, Any], str, dict[str, Any]]:
    """Select schema metadata objects and run the schema-scoped planning call."""

    db_index = load_db_index(task.db)
    schema_context_cache = build_schema_context_cache(
        task.db,
        db_index=db_index,
        config=schema_context_config,
    )
    linked_docs = [] if docs_context is None else [docs_context]
    schema_context_objects, context_diagnostics = select_schema_context_objects(
        schema_context_cache,
        task.question,
        linked_docs=linked_docs,
        config=schema_context_config,
    )
    planning_prompt = build_planning_prompt(
        task,
        docs_context,
        schema_context_objects,
        schema_context_config=schema_context_config,
    )
    decision = run_prompt(
        client,
        prompt_hashes=prompt_hashes,
        prompt_name="planning",
        output_type=SchemaPlanningDecision,
        user_prompt=planning_prompt,
    )
    sanitized_decision, planner_diagnostics = sanitize_schema_planning_decision(
        decision,
        schema_context_objects,
    )
    resolved = resolve_schema_context(
        db=task.db,
        selected_objects=sanitized_decision.selected_objects,
        canonical_schema_objects=schema_context_cache.objects,
        db_index=db_index,
        question=task.question,
        schema_context_evidence=schema_context_objects,
        constraints=sanitized_decision.constraints,
    )
    sql_reference_context = checked_schema_prompt(
        "sql_reference_context",
        resolved.prompt_context,
        schema_context_config,
    )
    prompt_budget = prompt_budget_diagnostics(
        planning_prompt=planning_prompt,
        sql_reference_context=sql_reference_context,
        schema_context_config=schema_context_config,
    )

    schema = SchemaSelection(
        db=task.db,
        selected_object_ids=[
            selected.object_id for selected in sanitized_decision.selected_objects
        ],
        expanded_tables=list(resolved.resolved_tables),
        rationale=sanitized_decision.rationale,
        confidence=sanitized_decision.confidence,
        diagnostics={
            "selection_prompt_chars": len(planning_prompt),
            "planning_prompt_chars": len(planning_prompt),
            "sql_reference_context_chars": len(sql_reference_context),
            "max_schema_prompt_chars": schema_context_config.max_schema_prompt_chars,
            "prompt_budget": prompt_budget,
            "schema_context_object_count": len(schema_context_objects),
            "selected_objects": [
                selected.model_dump(mode="json") for selected in sanitized_decision.selected_objects
            ],
            "resolved_tables": list(resolved.resolved_tables),
            "planner": planner_diagnostics,
            "resolver": resolved.diagnostics,
        },
    )
    schema_context = {
        "cache": {
            "db": schema_context_cache.db,
            "cache_key": schema_context_cache.cache_key,
            "object_count": len(schema_context_cache.objects),
            "chunk_count": len(schema_context_cache.chunks),
        },
        "selection": context_diagnostics,
        "schema_context_objects": [
            {
                "object_id": item.schema_object.object_id,
                "object_type": item.schema_object.object_type,
                "name": item.schema_object.name,
                "table_name": item.schema_object.table_name,
                "rank": item.rank,
                "score": item.score,
            }
            for item in schema_context_objects
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
        schema_context,
    )


def build_planning_prompt(
    task: Task,
    docs_context: str | None,
    schema_context_objects: list[Any],
    *,
    schema_context_config: SchemaContextConfig,
) -> str:
    """Build a planning prompt that fits the configured schema prompt budget."""

    return schema_context_planning_user_prompt(
        task,
        task.db,
        docs_context,
        schema_context_objects,
        max_docs_chars=schema_context_config.max_linked_doc_chars,
        max_total_chars=schema_context_config.max_schema_prompt_chars,
    )


def checked_schema_prompt(
    prompt_name: str,
    prompt: str,
    schema_context_config: SchemaContextConfig,
) -> str:
    """Enforce the shared schema prompt budget for generated schema contexts."""

    return enforce_prompt_budget(
        prompt_name,
        prompt,
        schema_context_config.max_schema_prompt_chars,
    )


def prompt_budget_diagnostics(
    *,
    planning_prompt: str | None = None,
    sql_reference_context: str | None = None,
    schema_context_config: SchemaContextConfig,
) -> dict[str, object]:
    """Return trace fields for prompt budget enforcement."""

    diagnostics: dict[str, object] = {
        "max_schema_prompt_chars": schema_context_config.max_schema_prompt_chars,
    }
    if planning_prompt is not None:
        planning_chars = len(planning_prompt)
        diagnostics.update(
            {
                "planning_prompt_chars": planning_chars,
                "planning_prompt_within_budget": (
                    planning_chars <= schema_context_config.max_schema_prompt_chars
                ),
            }
        )
    if sql_reference_context is not None:
        context_chars = len(sql_reference_context)
        diagnostics.update(
            {
                "sql_reference_context_chars": context_chars,
                "sql_reference_context_within_budget": (
                    context_chars <= schema_context_config.max_schema_prompt_chars
                ),
            }
        )
    return diagnostics


def log_candidate(instance_id: str, attempt: AttemptRecord) -> None:
    """Log a standard candidate-evaluated event."""

    logger.info(
        "candidate evaluated",
        instance_id=instance_id,
        stage=attempt.stage,
        validation_ok=attempt.validation.ok,
        execution_ok=attempt.execution_result.ok,
        score=attempt.score,
        elapsed_seconds=attempt.elapsed_seconds,
        row_count=attempt.execution_result.row_count,
    )


def generate_initial_candidates(
    ctx: TaskContext,
    attempts: list[AttemptRecord],
    *,
    initial_candidates: int,
    max_attempts: int,
) -> AttemptRecord | None:
    """Generate up to initial_candidates SQL candidates; appends to attempts in place."""

    logger.info(
        "generating candidates",
        instance_id=ctx.task.instance_id,
        initial_candidates=initial_candidates,
        max_attempts=max_attempts,
    )
    candidate_limit = min(initial_candidates, max_attempts - len(attempts))
    if candidate_limit <= 0:
        return _winner_attempt(attempts)

    batch = run_prompt(
        ctx.client,
        prompt_hashes=ctx.prompt_hashes,
        prompt_name="sql_generation_batch",
        output_type=SQLCandidateBatch,
        user_prompt=sql_generation_batch_prompt(
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
        log_candidate(ctx.task.instance_id, attempt)
    return _winner_attempt(attempts)


def repair_failed_execution(
    ctx: TaskContext,
    attempts: list[AttemptRecord],
    current_best: AttemptRecord | None,
    *,
    max_attempts: int,
) -> AttemptRecord | None:
    """Repair the best attempt if it failed execution; appends to attempts in place."""

    if current_best is None or current_best.execution_result.ok or len(attempts) >= max_attempts:
        return current_best

    logger.info(
        "repair requested",
        instance_id=ctx.task.instance_id,
        stage="repair",
        best_stage=current_best.stage,
    )
    repaired_candidate = run_prompt(
        ctx.client,
        prompt_hashes=ctx.prompt_hashes,
        prompt_name="sql_repair",
        output_type=SQLCandidate,
        user_prompt=sql_repair_prompt(
            ctx.task,
            ctx.intent,
            current_best,
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
    log_candidate(ctx.task.instance_id, attempt)
    return _winner_attempt(attempts)


def review_and_repair(
    ctx: TaskContext,
    attempts: list[AttemptRecord],
    current_best: AttemptRecord | None,
    *,
    max_attempts: int,
    semantic_repairs: int,
) -> tuple[AttemptRecord | None, dict[str, Any] | None]:
    """Ask the model to adjudicate executable attempts using local observations."""

    if current_best is None or not current_best.execution_result.ok:
        return current_best, None

    executable_attempts = [attempt for attempt in attempts if attempt.execution_result.ok]
    review_reason = candidate_review_reason(executable_attempts)

    logger.info(
        "candidate review requested",
        instance_id=ctx.task.instance_id,
        best_stage=current_best.stage,
        executable_attempts=len(executable_attempts),
        reason=review_reason,
    )
    review = run_prompt(
        ctx.client,
        prompt_hashes=ctx.prompt_hashes,
        prompt_name="candidate_review",
        output_type=CandidateReviewReport,
        user_prompt=candidate_review_prompt(
            ctx.task,
            ctx.intent,
            executable_attempts,
            ctx.sql_reference_context,
            ctx.docs_context,
            baseline_stage=current_best.stage,
            review_reason=review_reason,
        ),
    )
    review_payload = {
        "review_reason": review_reason,
        **review.model_dump(mode="json"),
    }
    current_best = _winner_attempt(attempts, preferred_stage=review.preferred_stage) or current_best

    current_best.critic = {
        "confidence": review.confidence,
        "issues": review.issues,
        "should_repair": review.should_repair,
        "repair_focus": review.repair_focus,
    }
    current_best.candidate_review = review_payload
    logger.info(
        "candidate review complete",
        instance_id=ctx.task.instance_id,
        preferred_stage=review.preferred_stage,
        should_repair=review.should_repair,
        confidence=review.confidence,
        issues=review.issues,
    )

    if not review.should_repair or semantic_repairs < 1:
        return current_best, review_payload

    budget_exhausted = len(attempts) >= max_attempts
    if budget_exhausted:
        current_best.repair_skipped_reason = "attempt budget exhausted"
        logger.info(
            "critic repair skipped",
            instance_id=ctx.task.instance_id,
            reason="attempt budget exhausted",
        )
        return current_best, review_payload

    logger.info(
        "semantic repair requested",
        instance_id=ctx.task.instance_id,
        stage="critic_repair",
        focus=review.repair_focus,
    )
    repaired_candidate = run_prompt(
        ctx.client,
        prompt_hashes=ctx.prompt_hashes,
        prompt_name="sql_repair",
        output_type=SQLCandidate,
        user_prompt=semantic_repair_prompt(
            ctx.task,
            ctx.intent,
            current_best,
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
    log_candidate(ctx.task.instance_id, attempt)

    new_best = _winner_attempt(attempts)
    return new_best, review_payload


def candidate_review_reason(executable_attempts: list[AttemptRecord]) -> str:
    """Return the trace reason for model-led final candidate adjudication."""

    if len(executable_attempts) == 1:
        return "final adjudication of the only executable candidate"
    return "final adjudication across executable candidates using local observations"


def rebuild_context_for_expansion(
    ctx: TaskContext,
    expanded_schema: SchemaSelection,
    *,
    table_schemas: dict[str, Any] | None = None,
    prebuilt_reference: str | None = None,
    schema_context: dict[str, Any] | None = None,
) -> TaskContext:
    """Rebuild selected-schema context while preserving the original answer contract."""

    new_table_schemas = table_schemas or table_schemas_for_selection(expanded_schema)
    reference_context = prebuilt_reference or sql_reference_context(
        expanded_schema,
        new_table_schemas,
    )
    reference_context = checked_schema_prompt(
        "schema_expansion_sql_reference_context",
        reference_context,
        ctx.schema_context_config,
    )
    intent = augment_intent_with_value_groundings(
        ctx.intent, task=ctx.task, schema=expanded_schema, table_schemas=new_table_schemas
    )
    return TaskContext(
        task=ctx.task,
        client=ctx.client,
        intent=intent,
        schema=expanded_schema,
        table_schemas=new_table_schemas,
        sql_reference_context=reference_context,
        docs_context=ctx.docs_context,
        prompt_hashes=ctx.prompt_hashes,
        schema_context_version=ctx.schema_context_version,
        schema_context_config=ctx.schema_context_config,
        schema_context=schema_context or ctx.schema_context,
    )


def write_task_output(
    ctx: TaskContext,
    attempts: list[AttemptRecord],
    current_best: AttemptRecord | None,
    run_paths: RunPaths,
    task_trace_path: Path,
    task_llm_log_path: Path,
    *,
    candidate_review_payload: dict[str, Any] | None,
    schema_expansion_payload: dict[str, Any] | None,
    expanded_ctx: TaskContext | None,
    live_logging_enabled: bool,
    started_at: float,
) -> FinalAnswer:
    """Write final SQL, CSV, and trace; return the FinalAnswer."""

    task = ctx.task
    final_ctx = expanded_ctx if expanded_ctx is not None else ctx
    final_selection = select_winner(attempts) if current_best is not None else None
    final_attempt_index = final_selection.index if final_selection is not None else None
    trace_payload: dict[str, Any] = {
        "instance_id": task.instance_id,
        "db": task.db,
        "question": task.question,
        "schema_selection": final_ctx.schema.model_dump(mode="json"),
        "schema_context_version": final_ctx.schema_context_version,
        "schema_context": final_ctx.schema_context,
        "intent": final_ctx.intent.model_dump(mode="json"),
        "prompt_hashes": ctx.prompt_hashes,
        "final_attempt_index": final_attempt_index,
        "final_attempt_reason": final_winner_reason(
            current_best,
            candidate_review_payload=candidate_review_payload,
        ),
        "attempts": attempts,
    }
    if candidate_review_payload is not None:
        trace_payload["candidate_review"] = candidate_review_payload
    if schema_expansion_payload is not None:
        trace_payload["schema_expansion"] = schema_expansion_payload
    if live_logging_enabled:
        trace_payload["llm_call_log_path"] = str(task_llm_log_path)

    if current_best is not None and current_best.execution_result.ok:
        sql_path = write_sql(run_paths, instance_id=task.instance_id, sql=current_best.sql)
        csv_path = csv_path_for(run_paths, instance_id=task.instance_id)
        best_dataframe = current_best._dataframe
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        best_dataframe.to_csv(csv_path, index=False)
        final_execution = ExecutionResult(
            ok=True,
            row_count=len(best_dataframe),
            columns=[str(column) for column in best_dataframe.columns],
            sample_rows=dataframe_records(best_dataframe.head(3)),
            csv_path=str(csv_path),
            error=None,
        )
        trace_payload.update(
            {
                "status": "success",
                "final_sql": current_best.sql,
                "sql_path": str(sql_path),
                "csv_path": str(csv_path),
                "final_execution": final_execution.model_dump(mode="json"),
            }
        )
        trace_payload["attempts"] = [trace_attempt(a) for a in attempts]
        write_trace(run_paths, instance_id=task.instance_id, trace=trace_payload)
        elapsed_seconds = round(perf_counter() - started_at, 3)
        logger.info(
            "task complete",
            instance_id=task.instance_id,
            status="success",
            run_root=str(run_paths.root),
            attempts=len(attempts),
            best_stage=current_best.stage,
            best_score=current_best.score,
            row_count=len(best_dataframe),
            columns=[str(column) for column in best_dataframe.columns],
            elapsed_seconds=elapsed_seconds,
            sql_path=str(sql_path),
            csv_path=str(csv_path),
        )
        return FinalAnswer(
            instance_id=task.instance_id,
            status="success",
            sql=current_best.sql,
            csv_path=str(csv_path),
            trace_path=str(task_trace_path),
        )

    trace_payload.update(
        {
            "status": "failed",
            "final_sql": current_best.sql if current_best is not None else None,
            "csv_path": None,
        }
    )
    trace_payload["attempts"] = [trace_attempt(a) for a in attempts]
    write_trace(run_paths, instance_id=task.instance_id, trace=trace_payload)
    elapsed_seconds = round(perf_counter() - started_at, 3)
    logger.warning(
        "task complete",
        instance_id=task.instance_id,
        status="failed",
        run_root=str(run_paths.root),
        attempts=len(attempts),
        best_stage=current_best.stage if current_best is not None else None,
        best_score=current_best.score if current_best is not None else None,
        row_count=current_best.execution_result.row_count if current_best is not None else 0,
        elapsed_seconds=elapsed_seconds,
    )
    return FinalAnswer(
        instance_id=task.instance_id,
        status="failed",
        sql=current_best.sql if current_best is not None else None,
        csv_path=None,
        trace_path=str(task_trace_path),
    )


def attempt_schema_expansion(
    ctx: TaskContext,
    attempts: list[AttemptRecord],
    current_best: AttemptRecord | None,
) -> tuple[AttemptRecord | None, dict[str, Any] | None, TaskContext | None]:
    """Run one schema-expansion recovery attempt when evidence warrants it."""

    if current_best is None:
        return current_best, None, None

    trigger = schema_expansion_trigger(current_best)
    if trigger is None:
        return current_best, None, None

    try:
        db_index = load_db_index(ctx.task.db)
    except Exception:
        return current_best, None, None

    expansion_query = schema_expansion_query(
        ctx.task.question,
        current_best,
        trigger,
        ctx.schema.selected_object_ids,
        ctx.schema.expanded_tables,
    )
    expansion_payload: dict[str, Any] = {
        "trigger": trigger,
        "expansion_query": expansion_query,
        "decision": None,
        "added_tables": [],
        "outcome": "no_new_tables",
    }
    schema_context_cache = build_schema_context_cache(
        ctx.task.db,
        db_index=db_index,
        config=ctx.schema_context_config,
    )

    det_tables = deterministic_expansion_tables(current_best, ctx.schema, db_index)
    if det_tables:
        expansion_payload["decision"] = {
            "source": "exact_name",
            "additional_tables": det_tables,
        }
        selected_additions: list[SelectedSchemaObject] = [
            SelectedSchemaObject(
                object_id=f"table:{table_name}",
                role="primary",
                reason="unambiguous table name in schema error",
            )
            for table_name in det_tables
        ]
        schema_context_objects: list[Any] = []
        planner_diagnostics: dict[str, object] = {}
    else:
        selected_additions, schema_context_objects, planner_diagnostics = _select_expansion_objects(
            ctx,
            expansion_query=expansion_query,
            schema_context_cache=schema_context_cache,
        )
        expansion_payload["decision"] = {
            "source": "schema_context",
            "selected_object_ids": [selected.object_id for selected in selected_additions],
        }

    expanded_schema, resolved, added_tables = resolve_expanded_schema(
        ctx.task.db,
        ctx.task.question,
        ctx.schema,
        selected_additions,
        schema_context_cache=schema_context_cache,
        db_index=db_index,
        schema_context_evidence=schema_context_objects,
        expansion_query=expansion_query,
    )
    expansion_payload.update(
        {
            "schema_context_objects": schema_context_object_trace(schema_context_objects),
            "selected_additions": [
                selected.model_dump(mode="json") for selected in selected_additions
            ],
            "added_tables": added_tables,
            "planner": planner_diagnostics,
            "resolver": resolved.diagnostics,
            "prompt_budget": prompt_budget_diagnostics(
                sql_reference_context=resolved.prompt_context,
                schema_context_config=ctx.schema_context_config,
            ),
        }
    )
    if not added_tables:
        expansion_payload["outcome"] = "no_new_tables"
        return current_best, expansion_payload, None

    expanded_ctx = rebuild_context_for_expansion(
        ctx,
        expanded_schema,
        table_schemas=resolved.table_schemas,
        prebuilt_reference=resolved.prompt_context,
    )

    expansion_stage = "schema_expansion"
    expansion_payload["expansion_attempt_stage"] = expansion_stage
    batch = run_prompt(
        expanded_ctx.client,
        prompt_hashes=expanded_ctx.prompt_hashes,
        prompt_name="sql_generation_batch",
        output_type=SQLCandidateBatch,
        user_prompt=sql_generation_batch_prompt(
            expanded_ctx.task,
            expanded_ctx.intent,
            expanded_ctx.sql_reference_context,
            expanded_ctx.docs_context,
            candidate_count=1,
        ),
    )
    if not batch.candidates:
        expansion_payload["outcome"] = "expanded_no_candidate"
        expanded_ctx.schema_context = schema_context_with_expansion(
            ctx.schema_context,
            expansion_payload,
        )
        return _winner_attempt(attempts), expansion_payload, expanded_ctx

    expansion_attempt = evaluate_candidate(
        task=expanded_ctx.task,
        candidate=batch.candidates[0],
        intent=expanded_ctx.intent,
        schema=expanded_ctx.schema,
        table_schemas=expanded_ctx.table_schemas,
        stage=expansion_stage,
    )
    attempts.append(expansion_attempt)
    log_candidate(ctx.task.instance_id, expansion_attempt)

    expansion_payload["outcome"] = (
        "expanded" if expansion_attempt.execution_result.ok else "expanded_failed"
    )
    expanded_ctx.schema_context = schema_context_with_expansion(
        ctx.schema_context,
        expansion_payload,
    )
    return _winner_attempt(attempts), expansion_payload, expanded_ctx


def _select_expansion_objects(
    ctx: TaskContext,
    *,
    expansion_query: str,
    schema_context_cache: Any,
) -> tuple[list[SelectedSchemaObject], list[Any], dict[str, object]]:
    """Select and sanitize schema objects for one expansion attempt."""

    linked_docs = [] if ctx.docs_context is None else [ctx.docs_context]
    schema_context_objects, context_diagnostics = select_schema_context_objects(
        schema_context_cache,
        expansion_query,
        linked_docs=linked_docs,
        config=ctx.schema_context_config,
    )
    expansion_task = ctx.task.model_copy(
        update={
            "question": (f"{ctx.task.question}\n\nSchema expansion evidence:\n{expansion_query}")
        }
    )
    planning_prompt = build_planning_prompt(
        expansion_task,
        ctx.docs_context,
        schema_context_objects,
        schema_context_config=ctx.schema_context_config,
    )
    decision = run_prompt(
        ctx.client,
        prompt_hashes=ctx.prompt_hashes,
        prompt_name="planning",
        output_type=SchemaPlanningDecision,
        user_prompt=planning_prompt,
    )
    sanitized_decision, planner_diagnostics = sanitize_schema_planning_decision(
        decision,
        schema_context_objects,
    )
    current_object_ids = set(ctx.schema.selected_object_ids)
    selected_additions = [
        selected
        for selected in sanitized_decision.selected_objects
        if selected.object_id not in current_object_ids
    ]
    diagnostics: dict[str, object] = {
        "selection": context_diagnostics,
        "planner": planner_diagnostics,
        "prompt_budget": prompt_budget_diagnostics(
            planning_prompt=planning_prompt,
            schema_context_config=ctx.schema_context_config,
        ),
        "rationale": sanitized_decision.rationale,
        "confidence": sanitized_decision.confidence,
    }
    return selected_additions, schema_context_objects, diagnostics


def run_prompt(
    client: LLMClient,
    *,
    prompt_hashes: dict[str, str],
    prompt_name: str,
    output_type: type[BaseModel],
    user_prompt: str,
) -> BaseModel:
    """Load one prompt hash and run one structured LLM call."""

    prompt = client.load_prompt(prompt_name)
    prompt_hashes.setdefault(prompt_name, prompt.sha256)
    return client.run_structured_with_prompt(
        user_prompt,
        prompt=prompt,
        output_type=output_type,
    )


def trace_attempt(attempt: AttemptRecord) -> dict[str, Any]:
    """Drop non-serializable internal fields before writing the trace."""

    return attempt.model_dump(mode="json")
