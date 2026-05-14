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
from sol01.schema.schema_context import build_available_schema_context
from sol01.schema.schema_context_cache import build_schema_context_cache

logger = get_logger(__name__)


@dataclasses.dataclass
class TaskRun:
    """All state for one task as it flows through the solver pipeline stages."""

    task: Task
    client: LLMClient
    schema_context_config: SchemaContextConfig
    schema_context_version: str = DEFAULT_SCHEMA_CONTEXT_VERSION

    # Set during plan_schema
    intent: Intent | None = None
    schema: SchemaSelection | None = None
    table_schemas: dict[str, Any] = dataclasses.field(default_factory=dict)
    sql_reference_context: str = ""
    docs_context: str | None = None
    schema_context: dict[str, Any] = dataclasses.field(default_factory=dict)

    # Accumulated across stages
    prompt_hashes: dict[str, str] = dataclasses.field(default_factory=dict)
    attempts: list[AttemptRecord] = dataclasses.field(default_factory=list)
    candidate_review_payload: dict[str, Any] | None = None
    recovery_payload: dict[str, Any] | None = None


def _current_best(run: TaskRun, *, preferred_stage: str | None = None) -> AttemptRecord | None:
    sel = select_winner(run.attempts, preferred_stage=preferred_stage)
    return sel.attempt if sel is not None else None


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def check_skip(
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


def plan_schema(run: TaskRun, *, run_paths: RunPaths) -> TaskRun:
    """Load docs, select schema objects, run planning, and populate run."""

    logger.info(
        "task start",
        instance_id=run.task.instance_id,
        db=run.task.db,
        question_preview=question_preview(run.task.question),
        question_length=len(run.task.question),
        run_root=str(run_paths.root),
    )
    run.docs_context = (
        load_document_text(run.task.external_knowledge) if run.task.external_knowledge else None
    )

    db_index = load_db_index(run.task.db)
    schema_context_cache = build_schema_context_cache(
        run.task.db,
        db_index=db_index,
        config=run.schema_context_config,
    )
    linked_docs = [] if run.docs_context is None else [run.docs_context]
    schema_context_objects, context_diagnostics = build_available_schema_context(
        schema_context_cache,
        run.task.question,
        linked_docs=linked_docs,
        config=run.schema_context_config,
    )
    planning_prompt = _build_planning_prompt(
        run.task,
        run.docs_context,
        schema_context_objects,
        schema_context_config=run.schema_context_config,
    )
    decision = _run_prompt(
        run.client,
        prompt_hashes=run.prompt_hashes,
        prompt_name="planning",
        output_type=SchemaPlanningDecision,
        user_prompt=planning_prompt,
    )
    sanitized_decision, planner_diagnostics = sanitize_schema_planning_decision(
        decision, schema_context_objects
    )
    resolved = resolve_schema_context(
        db=run.task.db,
        selected_objects=sanitized_decision.selected_objects,
        canonical_schema_objects=schema_context_cache.objects,
        db_index=db_index,
        question=run.task.question,
        schema_context_evidence=schema_context_objects,
        constraints=sanitized_decision.constraints,
    )
    sql_ref_context = _checked_schema_prompt(
        "sql_reference_context",
        resolved.prompt_context,
        run.schema_context_config,
    )
    prompt_budget = _prompt_budget_diagnostics(
        planning_prompt=planning_prompt,
        sql_reference_context=sql_ref_context,
        schema_context_config=run.schema_context_config,
    )

    run.schema = SchemaSelection(
        db=run.task.db,
        selected_object_ids=[s.object_id for s in sanitized_decision.selected_objects],
        expanded_tables=list(resolved.resolved_tables),
        rationale=sanitized_decision.rationale,
        confidence=sanitized_decision.confidence,
        diagnostics={
            "planning_prompt_chars": len(planning_prompt),
            "sql_reference_context_chars": len(sql_ref_context),
            "max_schema_prompt_chars": run.schema_context_config.max_schema_prompt_chars,
            "prompt_budget": prompt_budget,
            "schema_context_object_count": len(schema_context_objects),
            "selected_objects": [
                s.model_dump(mode="json") for s in sanitized_decision.selected_objects
            ],
            "resolved_tables": list(resolved.resolved_tables),
            "planner": planner_diagnostics,
            "resolver": resolved.diagnostics,
        },
    )
    run.intent = augment_intent_with_value_groundings(
        sanitized_decision.intent,
        task=run.task,
        schema=run.schema,
        table_schemas=resolved.table_schemas,
    )
    run.table_schemas = resolved.table_schemas
    run.sql_reference_context = sql_ref_context
    run.schema_context = {
        "cache": {
            "db": schema_context_cache.db,
            "cache_key": schema_context_cache.cache_key,
            "object_count": len(schema_context_cache.objects),
        },
        "available_context": context_diagnostics,
        "schema_context_objects": [
            {
                "object_id": item.schema_object.object_id,
                "object_type": item.schema_object.object_type,
                "name": item.schema_object.name,
                "table_name": item.schema_object.table_name,
                "position": item.position,
            }
            for item in schema_context_objects
        ],
        "planner": planner_diagnostics,
        "resolver": resolved.diagnostics,
        "prompt_budget": prompt_budget,
    }
    logger.info(
        "schema selected",
        instance_id=run.task.instance_id,
        expanded_count=len(run.schema.expanded_tables),
        expanded_tables=run.schema.expanded_tables,
        confidence=run.schema.confidence,
    )
    logger.info(
        "intent extracted",
        instance_id=run.task.instance_id,
        metrics=run.intent.metrics,
        filters=run.intent.filters,
        time_constraints=run.intent.time_constraints,
    )
    return run


def generate_initial_candidates(run: TaskRun, *, count: int, max_attempts: int) -> TaskRun:
    """Generate up to count SQL candidates and append them to run.attempts."""

    logger.info(
        "generating candidates",
        instance_id=run.task.instance_id,
        initial_candidates=count,
        max_attempts=max_attempts,
    )
    candidate_limit = min(count, max_attempts - len(run.attempts))
    if candidate_limit <= 0:
        return run

    batch = _run_prompt(
        run.client,
        prompt_hashes=run.prompt_hashes,
        prompt_name="sql_generation_batch",
        output_type=SQLCandidateBatch,
        user_prompt=sql_generation_batch_prompt(
            run.task,
            run.intent,
            run.sql_reference_context,
            run.docs_context,
            candidate_count=candidate_limit,
        ),
    )
    for i, candidate in enumerate(batch.candidates[:candidate_limit]):
        attempt = evaluate_candidate(
            task=run.task,
            candidate=candidate,
            intent=run.intent,
            schema=run.schema,
            table_schemas=run.table_schemas,
            stage=f"initial_{i + 1}",
        )
        run.attempts.append(attempt)
        _log_candidate(run.task.instance_id, attempt)
    return run


def run_recovery_stage(
    run: TaskRun,
    *,
    max_attempts: int,
    semantic_repairs: int,
) -> TaskRun:
    """Run the single recovery stage until it is done or the shared budget is spent."""

    actions: list[dict[str, Any]] = []
    remaining_semantic_repairs = semantic_repairs
    review_ran = False
    stop_reason = "no_attempts"

    while True:
        best = _current_best(run)
        if best is None:
            stop_reason = "no_attempts"
            break

        schema_trigger = schema_expansion_trigger(best)
        if schema_trigger is not None:
            if len(run.attempts) >= max_attempts:
                best.repair_skipped_reason = "attempt budget exhausted"
                stop_reason = "attempt_budget_exhausted"
                break

            action = _run_schema_recovery(run, best=best, trigger=schema_trigger)
            actions.append(action)
            if action["outcome"] == "expanded":
                stop_reason = "schema_recovery_complete"
                break
            if action["outcome"] != "expanded_failed":
                stop_reason = action["outcome"]
                break
            continue

        if not best.execution_result.ok:
            if len(run.attempts) >= max_attempts:
                best.repair_skipped_reason = "attempt budget exhausted"
                stop_reason = "attempt_budget_exhausted"
                break

            action = _run_sql_recovery(run, best=best)
            actions.append(action)
            if action["outcome"] == "recovered":
                stop_reason = "sql_recovery_complete"
                break
            continue

        if not review_ran:
            review = _run_candidate_review(run, best=best)
            review_ran = True
            if not review.should_repair:
                stop_reason = "review_complete"
                break
            if remaining_semantic_repairs < 1:
                best = _current_best(run, preferred_stage=review.preferred_stage) or best
                best.repair_skipped_reason = "semantic repair budget exhausted"
                stop_reason = "semantic_repair_budget_exhausted"
                break
            if len(run.attempts) >= max_attempts:
                best = _current_best(run, preferred_stage=review.preferred_stage) or best
                best.repair_skipped_reason = "attempt budget exhausted"
                stop_reason = "attempt_budget_exhausted"
                break

            actions.append(_run_semantic_recovery(run, best=best, review=review))
            remaining_semantic_repairs -= 1
            continue

        stop_reason = "no_recovery_needed"
        break

    run.recovery_payload = {
        "priority_order": ["schema", "sql", "semantic"],
        "attempts_before_recovery": len(
            [attempt for attempt in run.attempts if attempt.stage.startswith("initial_")]
        ),
        "attempts_after_recovery": len(run.attempts),
        "max_attempts": max_attempts,
        "semantic_repairs_allowed": semantic_repairs,
        "semantic_repairs_remaining": remaining_semantic_repairs,
        "actions": actions,
        "stop_reason": stop_reason,
    }
    return run


def write_task_output(
    run: TaskRun,
    *,
    run_paths: RunPaths,
    task_trace_path: Path,
    task_llm_log_path: Path,
    live_logging_enabled: bool,
    started_at: float,
) -> FinalAnswer:
    """Write final SQL, CSV, and trace; return the FinalAnswer."""

    task = run.task
    best = _current_best(run)
    final_selection = select_winner(run.attempts) if best is not None else None
    final_attempt_index = final_selection.index if final_selection is not None else None

    trace_payload: dict[str, Any] = {
        "instance_id": task.instance_id,
        "db": task.db,
        "question": task.question,
        "schema_selection": run.schema.model_dump(mode="json"),
        "schema_context_version": run.schema_context_version,
        "schema_context": run.schema_context,
        "intent": run.intent.model_dump(mode="json"),
        "prompt_hashes": run.prompt_hashes,
        "final_attempt_index": final_attempt_index,
        "final_attempt_reason": final_winner_reason(
            best,
            candidate_review_payload=run.candidate_review_payload,
        ),
        "attempts": [a.model_dump(mode="json") for a in run.attempts],
    }
    if run.candidate_review_payload is not None:
        trace_payload["candidate_review"] = run.candidate_review_payload
    if run.recovery_payload is not None:
        trace_payload["recovery"] = run.recovery_payload
    if live_logging_enabled:
        trace_payload["llm_call_log_path"] = str(task_llm_log_path)

    if best is not None and best.execution_result.ok:
        sql_path = write_sql(run_paths, instance_id=task.instance_id, sql=best.sql)
        csv_path = csv_path_for(run_paths, instance_id=task.instance_id)
        df = best._dataframe
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        final_execution = ExecutionResult(
            ok=True,
            row_count=len(df),
            columns=[str(c) for c in df.columns],
            sample_rows=dataframe_records(df.head(3)),
            csv_path=str(csv_path),
            error=None,
        )
        trace_payload.update(
            {
                "status": "success",
                "final_sql": best.sql,
                "sql_path": str(sql_path),
                "csv_path": str(csv_path),
                "final_execution": final_execution.model_dump(mode="json"),
            }
        )
        write_trace(run_paths, instance_id=task.instance_id, trace=trace_payload)
        elapsed = round(perf_counter() - started_at, 3)
        logger.info(
            "task complete",
            instance_id=task.instance_id,
            status="success",
            run_root=str(run_paths.root),
            attempts=len(run.attempts),
            best_stage=best.stage,
            best_score=best.score,
            row_count=len(df),
            columns=[str(c) for c in df.columns],
            elapsed_seconds=elapsed,
            sql_path=str(sql_path),
            csv_path=str(csv_path),
        )
        return FinalAnswer(
            instance_id=task.instance_id,
            status="success",
            sql=best.sql,
            csv_path=str(csv_path),
            trace_path=str(task_trace_path),
        )

    trace_payload.update(
        {
            "status": "failed",
            "final_sql": best.sql if best is not None else None,
            "csv_path": None,
        }
    )
    write_trace(run_paths, instance_id=task.instance_id, trace=trace_payload)
    elapsed = round(perf_counter() - started_at, 3)
    logger.warning(
        "task complete",
        instance_id=task.instance_id,
        status="failed",
        run_root=str(run_paths.root),
        attempts=len(run.attempts),
        best_stage=best.stage if best is not None else None,
        best_score=best.score if best is not None else None,
        row_count=best.execution_result.row_count if best is not None else 0,
        elapsed_seconds=elapsed,
    )
    return FinalAnswer(
        instance_id=task.instance_id,
        status="failed",
        sql=best.sql if best is not None else None,
        csv_path=None,
        trace_path=str(task_trace_path),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _select_expansion_objects(
    run: TaskRun,
    *,
    expansion_query: str,
    schema_context_cache: Any,
) -> tuple[list[SelectedSchemaObject], list[Any], dict[str, object]]:
    """Select and sanitize schema objects for one expansion attempt."""

    linked_docs = [] if run.docs_context is None else [run.docs_context]
    schema_context_objects, context_diagnostics = build_available_schema_context(
        schema_context_cache,
        expansion_query,
        linked_docs=linked_docs,
        config=run.schema_context_config,
    )
    expansion_task = run.task.model_copy(
        update={"question": f"{run.task.question}\n\nSchema expansion evidence:\n{expansion_query}"}
    )
    planning_prompt = _build_planning_prompt(
        expansion_task,
        run.docs_context,
        schema_context_objects,
        schema_context_config=run.schema_context_config,
    )
    decision = _run_prompt(
        run.client,
        prompt_hashes=run.prompt_hashes,
        prompt_name="planning",
        output_type=SchemaPlanningDecision,
        user_prompt=planning_prompt,
    )
    sanitized_decision, planner_diagnostics = sanitize_schema_planning_decision(
        decision, schema_context_objects
    )
    current_ids = set(run.schema.selected_object_ids)
    selected_additions = [
        s for s in sanitized_decision.selected_objects if s.object_id not in current_ids
    ]
    diagnostics: dict[str, object] = {
        "available_context": context_diagnostics,
        "planner": planner_diagnostics,
        "prompt_budget": _prompt_budget_diagnostics(
            planning_prompt=planning_prompt,
            schema_context_config=run.schema_context_config,
        ),
        "rationale": sanitized_decision.rationale,
        "confidence": sanitized_decision.confidence,
    }
    return selected_additions, schema_context_objects, diagnostics


def _run_sql_recovery(run: TaskRun, *, best: AttemptRecord) -> dict[str, Any]:
    """Attempt one SQL-focused recovery for a non-executable best attempt."""

    logger.info(
        "recovery action requested",
        instance_id=run.task.instance_id,
        action="sql",
        best_stage=best.stage,
    )
    repaired = _run_prompt(
        run.client,
        prompt_hashes=run.prompt_hashes,
        prompt_name="sql_repair",
        output_type=SQLCandidate,
        user_prompt=sql_repair_prompt(
            run.task,
            run.intent,
            best,
            run.sql_reference_context,
            run.docs_context,
        ),
    )
    attempt = evaluate_candidate(
        task=run.task,
        candidate=repaired,
        intent=run.intent,
        schema=run.schema,
        table_schemas=run.table_schemas,
        stage="recovery_sql",
    )
    run.attempts.append(attempt)
    _log_candidate(run.task.instance_id, attempt)
    return {
        "kind": "sql",
        "trigger": "best_attempt_not_executable",
        "source_stage": best.stage,
        "attempt_stage": attempt.stage,
        "outcome": "recovered" if attempt.execution_result.ok else "still_failed",
    }


def _run_candidate_review(run: TaskRun, *, best: AttemptRecord) -> CandidateReviewReport:
    """Review executable attempts and attach the critic report to the preferred baseline."""

    executable = [attempt for attempt in run.attempts if attempt.execution_result.ok]
    review_reason = (
        "final adjudication of the only executable candidate"
        if len(executable) == 1
        else "final adjudication across executable candidates using local observations"
    )

    logger.info(
        "candidate review requested",
        instance_id=run.task.instance_id,
        best_stage=best.stage,
        executable_attempts=len(executable),
        reason=review_reason,
    )
    review = _run_prompt(
        run.client,
        prompt_hashes=run.prompt_hashes,
        prompt_name="candidate_review",
        output_type=CandidateReviewReport,
        user_prompt=candidate_review_prompt(
            run.task,
            run.intent,
            executable,
            run.sql_reference_context,
            run.docs_context,
            baseline_stage=best.stage,
            review_reason=review_reason,
        ),
    )
    run.candidate_review_payload = {
        "review_reason": review_reason,
        **review.model_dump(mode="json"),
    }
    annotated = _current_best(run, preferred_stage=review.preferred_stage) or best
    annotated.critic = {
        "confidence": review.confidence,
        "issues": review.issues,
        "should_repair": review.should_repair,
        "repair_focus": review.repair_focus,
    }
    annotated.candidate_review = run.candidate_review_payload

    logger.info(
        "candidate review complete",
        instance_id=run.task.instance_id,
        preferred_stage=review.preferred_stage,
        should_repair=review.should_repair,
        confidence=review.confidence,
        issues=review.issues,
    )
    return review


def _run_semantic_recovery(
    run: TaskRun,
    *,
    best: AttemptRecord,
    review: CandidateReviewReport,
) -> dict[str, Any]:
    """Attempt one semantic recovery after the critic identifies a concrete issue."""

    logger.info(
        "recovery action requested",
        instance_id=run.task.instance_id,
        action="semantic",
        best_stage=best.stage,
        focus=review.repair_focus,
    )
    repaired = _run_prompt(
        run.client,
        prompt_hashes=run.prompt_hashes,
        prompt_name="sql_repair",
        output_type=SQLCandidate,
        user_prompt=semantic_repair_prompt(
            run.task,
            run.intent,
            best,
            ConfidenceReport(
                confidence=review.confidence,
                issues=review.issues,
                should_repair=review.should_repair,
                repair_focus=review.repair_focus,
            ),
            run.sql_reference_context,
            run.docs_context,
        ),
    )
    attempt = evaluate_candidate(
        task=run.task,
        candidate=repaired,
        intent=run.intent,
        schema=run.schema,
        table_schemas=run.table_schemas,
        stage="recovery_semantic",
    )
    run.attempts.append(attempt)
    _log_candidate(run.task.instance_id, attempt)
    return {
        "kind": "semantic",
        "trigger": review.repair_focus or "critic_requested_repair",
        "source_stage": best.stage,
        "attempt_stage": attempt.stage,
        "outcome": "recovered" if attempt.execution_result.ok else "still_failed",
    }


def _run_schema_recovery(
    run: TaskRun,
    *,
    best: AttemptRecord,
    trigger: str,
) -> dict[str, Any]:
    """Expand schema context and regenerate one candidate when a schema miss is evident."""

    try:
        db_index = load_db_index(run.task.db)
    except Exception:
        return {
            "kind": "schema",
            "trigger": trigger,
            "source_stage": best.stage,
            "attempt_stage": None,
            "decision": None,
            "added_tables": [],
            "outcome": "db_index_unavailable",
        }

    expansion_query = schema_expansion_query(
        run.task.question,
        best,
        trigger,
        run.schema.selected_object_ids,
        run.schema.expanded_tables,
    )
    action: dict[str, Any] = {
        "kind": "schema",
        "trigger": trigger,
        "source_stage": best.stage,
        "expansion_query": expansion_query,
        "decision": None,
        "added_tables": [],
        "outcome": "no_new_tables",
    }
    schema_context_cache = build_schema_context_cache(
        run.task.db,
        db_index=db_index,
        config=run.schema_context_config,
    )

    det_tables = deterministic_expansion_tables(best, run.schema, db_index)
    if det_tables:
        action["decision"] = {
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
            run,
            expansion_query=expansion_query,
            schema_context_cache=schema_context_cache,
        )
        action["decision"] = {
            "source": "schema_context",
            "selected_object_ids": [selected.object_id for selected in selected_additions],
        }

    expanded_schema, resolved, added_tables = resolve_expanded_schema(
        run.task.db,
        run.task.question,
        run.schema,
        selected_additions,
        schema_context_cache=schema_context_cache,
        db_index=db_index,
        schema_context_evidence=schema_context_objects,
        expansion_query=expansion_query,
    )
    action.update(
        {
            "schema_context_objects": schema_context_object_trace(schema_context_objects),
            "selected_additions": [s.model_dump(mode="json") for s in selected_additions],
            "added_tables": added_tables,
            "planner": planner_diagnostics,
            "resolver": resolved.diagnostics,
            "prompt_budget": _prompt_budget_diagnostics(
                sql_reference_context=resolved.prompt_context,
                schema_context_config=run.schema_context_config,
            ),
        }
    )

    if not added_tables:
        run.schema_context = schema_context_with_expansion(run.schema_context, action)
        return action

    new_table_schemas = resolved.table_schemas or table_schemas_for_selection(expanded_schema)
    run.schema = expanded_schema
    run.table_schemas = new_table_schemas
    run.sql_reference_context = _checked_schema_prompt(
        "schema_expansion_sql_reference_context",
        resolved.prompt_context,
        run.schema_context_config,
    )
    run.intent = augment_intent_with_value_groundings(
        run.intent, task=run.task, schema=expanded_schema, table_schemas=new_table_schemas
    )

    batch = _run_prompt(
        run.client,
        prompt_hashes=run.prompt_hashes,
        prompt_name="sql_generation_batch",
        output_type=SQLCandidateBatch,
        user_prompt=sql_generation_batch_prompt(
            run.task,
            run.intent,
            run.sql_reference_context,
            run.docs_context,
            candidate_count=1,
        ),
    )

    if not batch.candidates:
        action["outcome"] = "expanded_no_candidate"
        run.schema_context = schema_context_with_expansion(run.schema_context, action)
        return action

    expansion_attempt = evaluate_candidate(
        task=run.task,
        candidate=batch.candidates[0],
        intent=run.intent,
        schema=run.schema,
        table_schemas=run.table_schemas,
        stage="recovery_schema",
    )
    run.attempts.append(expansion_attempt)
    _log_candidate(run.task.instance_id, expansion_attempt)

    action["attempt_stage"] = expansion_attempt.stage
    action["outcome"] = "expanded" if expansion_attempt.execution_result.ok else "expanded_failed"
    run.schema_context = schema_context_with_expansion(run.schema_context, action)
    return action


def _build_planning_prompt(
    task: Task,
    docs_context: str | None,
    schema_context_objects: list[Any],
    *,
    schema_context_config: SchemaContextConfig,
) -> str:
    return schema_context_planning_user_prompt(
        task,
        task.db,
        docs_context,
        schema_context_objects,
        max_docs_chars=schema_context_config.max_linked_doc_chars,
        max_total_chars=schema_context_config.max_schema_prompt_chars,
    )


def _checked_schema_prompt(
    prompt_name: str,
    prompt: str,
    schema_context_config: SchemaContextConfig,
) -> str:
    return enforce_prompt_budget(prompt_name, prompt, schema_context_config.max_schema_prompt_chars)


def _prompt_budget_diagnostics(
    *,
    planning_prompt: str | None = None,
    sql_reference_context: str | None = None,
    schema_context_config: SchemaContextConfig,
) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "max_schema_prompt_chars": schema_context_config.max_schema_prompt_chars,
    }
    if planning_prompt is not None:
        chars = len(planning_prompt)
        diagnostics.update(
            {
                "planning_prompt_chars": chars,
                "planning_prompt_within_budget": (
                    chars <= schema_context_config.max_schema_prompt_chars
                ),
            }
        )
    if sql_reference_context is not None:
        chars = len(sql_reference_context)
        diagnostics.update(
            {
                "sql_reference_context_chars": chars,
                "sql_reference_context_within_budget": (
                    chars <= schema_context_config.max_schema_prompt_chars
                ),
            }
        )
    return diagnostics


def _log_candidate(instance_id: str, attempt: AttemptRecord) -> None:
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


def _run_prompt(
    client: LLMClient,
    *,
    prompt_hashes: dict[str, str],
    prompt_name: str,
    output_type: type[BaseModel],
    user_prompt: str,
) -> BaseModel:
    prompt = client.load_prompt(prompt_name)
    prompt_hashes.setdefault(prompt_name, prompt.sha256)
    return client.run_structured_with_prompt(
        user_prompt,
        prompt=prompt,
        output_type=output_type,
    )
