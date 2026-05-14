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
from sol01.schema.schema_context import select_schema_context_objects
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
    schema_expansion_payload: dict[str, Any] | None = None


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
    schema_context_objects, context_diagnostics = select_schema_context_objects(
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
            "selection_prompt_chars": len(planning_prompt),
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


def repair_failed_execution(run: TaskRun, *, max_attempts: int) -> TaskRun:
    """Repair the best attempt if execution failed; appends to run.attempts."""

    best = _current_best(run)
    if best is None or best.execution_result.ok or len(run.attempts) >= max_attempts:
        return run

    logger.info(
        "repair requested",
        instance_id=run.task.instance_id,
        stage="repair",
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
        stage="repair",
    )
    run.attempts.append(attempt)
    _log_candidate(run.task.instance_id, attempt)
    return run


def review_and_repair(run: TaskRun, *, max_attempts: int, semantic_repairs: int) -> TaskRun:
    """Review executable attempts and optionally run one semantic repair."""

    best = _current_best(run)
    if best is None or not best.execution_result.ok:
        return run

    executable = [a for a in run.attempts if a.execution_result.ok]
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
    best = _current_best(run, preferred_stage=review.preferred_stage) or best
    best.critic = {
        "confidence": review.confidence,
        "issues": review.issues,
        "should_repair": review.should_repair,
        "repair_focus": review.repair_focus,
    }
    best.candidate_review = run.candidate_review_payload

    logger.info(
        "candidate review complete",
        instance_id=run.task.instance_id,
        preferred_stage=review.preferred_stage,
        should_repair=review.should_repair,
        confidence=review.confidence,
        issues=review.issues,
    )

    if not review.should_repair or semantic_repairs < 1:
        return run

    if len(run.attempts) >= max_attempts:
        best.repair_skipped_reason = "attempt budget exhausted"
        logger.info(
            "critic repair skipped",
            instance_id=run.task.instance_id,
            reason="attempt budget exhausted",
        )
        return run

    logger.info(
        "semantic repair requested",
        instance_id=run.task.instance_id,
        stage="critic_repair",
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
        stage="critic_repair",
    )
    run.attempts.append(attempt)
    _log_candidate(run.task.instance_id, attempt)
    return run


def attempt_schema_expansion(run: TaskRun) -> TaskRun:
    """Run one schema-expansion attempt when evidence warrants it."""

    best = _current_best(run)
    if best is None:
        return run

    trigger = schema_expansion_trigger(best)
    if trigger is None:
        return run

    try:
        db_index = load_db_index(run.task.db)
    except Exception:
        return run

    expansion_query = schema_expansion_query(
        run.task.question,
        best,
        trigger,
        run.schema.selected_object_ids,
        run.schema.expanded_tables,
    )
    expansion_payload: dict[str, Any] = {
        "trigger": trigger,
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
        expansion_payload["decision"] = {
            "source": "exact_name",
            "additional_tables": det_tables,
        }
        selected_additions: list[SelectedSchemaObject] = [
            SelectedSchemaObject(
                object_id=f"table:{t}",
                role="primary",
                reason="unambiguous table name in schema error",
            )
            for t in det_tables
        ]
        schema_context_objects: list[Any] = []
        planner_diagnostics: dict[str, object] = {}
    else:
        selected_additions, schema_context_objects, planner_diagnostics = _select_expansion_objects(
            run,
            expansion_query=expansion_query,
            schema_context_cache=schema_context_cache,
        )
        expansion_payload["decision"] = {
            "source": "schema_context",
            "selected_object_ids": [s.object_id for s in selected_additions],
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
    expansion_payload.update(
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
        expansion_payload["outcome"] = "no_new_tables"
        run.schema_expansion_payload = expansion_payload
        return run

    # Update run context with the expanded schema
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

    expansion_stage = "schema_expansion"
    expansion_payload["expansion_attempt_stage"] = expansion_stage
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
        expansion_payload["outcome"] = "expanded_no_candidate"
        run.schema_context = schema_context_with_expansion(run.schema_context, expansion_payload)
        run.schema_expansion_payload = expansion_payload
        return run

    expansion_attempt = evaluate_candidate(
        task=run.task,
        candidate=batch.candidates[0],
        intent=run.intent,
        schema=run.schema,
        table_schemas=run.table_schemas,
        stage=expansion_stage,
    )
    run.attempts.append(expansion_attempt)
    _log_candidate(run.task.instance_id, expansion_attempt)

    expansion_payload["outcome"] = (
        "expanded" if expansion_attempt.execution_result.ok else "expanded_failed"
    )
    run.schema_context = schema_context_with_expansion(run.schema_context, expansion_payload)
    run.schema_expansion_payload = expansion_payload
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
    if run.schema_expansion_payload is not None:
        trace_payload["schema_expansion"] = run.schema_expansion_payload
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
    schema_context_objects, context_diagnostics = select_schema_context_objects(
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
        "selection": context_diagnostics,
        "planner": planner_diagnostics,
        "prompt_budget": _prompt_budget_diagnostics(
            planning_prompt=planning_prompt,
            schema_context_config=run.schema_context_config,
        ),
        "rationale": sanitized_decision.rationale,
        "confidence": sanitized_decision.confidence,
    }
    return selected_additions, schema_context_objects, diagnostics


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
