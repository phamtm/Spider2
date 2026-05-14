"""Recovery-stage orchestration for one solver task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sol01.candidates.verification import (
    augment_intent_with_value_groundings,
    table_schemas_for_selection,
)
from sol01.infra.logging import get_logger
from sol01.infra.policy import DEFAULT_RECOVERY_SIGNAL_POLICY
from sol01.llm.planning_prompts import sanitize_schema_planning_decision
from sol01.llm.sql_prompts import (
    candidate_review_prompt,
    semantic_repair_prompt,
    sql_generation_batch_prompt,
    sql_repair_prompt,
)
from sol01.models import (
    CandidateReviewReport,
    ConfidenceReport,
    SchemaPlanningDecision,
    SelectedSchemaObject,
    SQLCandidate,
    SQLCandidateBatch,
)
from sol01.pipeline_state import TaskRun, current_best
from sol01.pipeline_support import (
    build_planning_prompt,
    checked_schema_prompt,
    evaluate_and_record_candidate,
    prompt_budget_diagnostics,
    run_prompt,
)
from sol01.recovery_signals import schema_expansion_trigger
from sol01.schema.db_index import load_db_index
from sol01.schema.expansion import (
    deterministic_expansion_tables,
    resolve_expanded_schema,
    schema_context_object_trace,
    schema_context_with_expansion,
    schema_expansion_query,
)
from sol01.schema.schema_context import build_available_schema_context
from sol01.schema.schema_context_cache import build_schema_context_cache

logger = get_logger(__name__)


@dataclass(frozen=True)
class RecoveryAction:
    """Typed recovery action record that can be serialized into traces."""

    kind: str
    trigger: str
    source_stage: str
    outcome: str
    attempt_stage: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_trace_payload(self) -> dict[str, Any]:
        payload = {
            "kind": self.kind,
            "trigger": self.trigger,
            "source_stage": self.source_stage,
            "attempt_stage": self.attempt_stage,
            "outcome": self.outcome,
        }
        payload.update(self.details)
        return payload


def run_recovery_stage(run: TaskRun) -> TaskRun:
    """Run the single recovery stage until it is done or the shared budget is spent."""

    actions: list[RecoveryAction] = []
    remaining_semantic_repairs = run.policy.semantic_repairs
    review_ran = False
    stop_reason = "no_attempts"

    while True:
        best = current_best(run)
        if best is None:
            stop_reason = "no_attempts"
            break

        schema_trigger = schema_expansion_trigger(best)
        if schema_trigger is not None:
            if len(run.attempts) >= run.policy.max_attempts:
                best.repair_skipped_reason = "attempt budget exhausted"
                stop_reason = "attempt_budget_exhausted"
                break

            action = _run_schema_recovery(run, best=best, trigger=schema_trigger)
            actions.append(action)
            if action.outcome == "expanded":
                stop_reason = "schema_recovery_complete"
                break
            if action.outcome != "expanded_failed":
                stop_reason = action.outcome
                break
            continue

        if not best.execution_result.ok:
            if len(run.attempts) >= run.policy.max_attempts:
                best.repair_skipped_reason = "attempt budget exhausted"
                stop_reason = "attempt_budget_exhausted"
                break

            action = _run_sql_recovery(run, best=best)
            actions.append(action)
            if action.outcome == "recovered":
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
                best = current_best(run, preferred_stage=review.preferred_stage) or best
                best.repair_skipped_reason = "semantic repair budget exhausted"
                stop_reason = "semantic_repair_budget_exhausted"
                break
            if len(run.attempts) >= run.policy.max_attempts:
                best = current_best(run, preferred_stage=review.preferred_stage) or best
                best.repair_skipped_reason = "attempt budget exhausted"
                stop_reason = "attempt_budget_exhausted"
                break

            actions.append(_run_semantic_recovery(run, best=best, review=review))
            remaining_semantic_repairs -= 1
            continue

        stop_reason = "no_recovery_needed"
        break

    run.recovery_payload = {
        "priority_order": list(DEFAULT_RECOVERY_SIGNAL_POLICY.priority_order),
        "attempts_before_recovery": len(
            [attempt for attempt in run.attempts if attempt.stage.startswith("initial_")]
        ),
        "attempts_after_recovery": len(run.attempts),
        "max_attempts": run.policy.max_attempts,
        "semantic_repairs_allowed": run.policy.semantic_repairs,
        "semantic_repairs_remaining": remaining_semantic_repairs,
        "actions": [action.as_trace_payload() for action in actions],
        "stop_reason": stop_reason,
    }
    return run


def _select_expansion_objects(
    run: TaskRun,
    *,
    expansion_query: str,
    schema_context_cache: Any,
) -> tuple[list[SelectedSchemaObject], list[Any], dict[str, object]]:
    """Select and sanitize schema objects for one expansion attempt."""

    schema_context_objects, context_diagnostics = build_available_schema_context(
        schema_context_cache
    )
    expansion_task = run.task.model_copy(
        update={"question": f"{run.task.question}\n\nSchema expansion evidence:\n{expansion_query}"}
    )
    planning_prompt = build_planning_prompt(
        expansion_task,
        run.docs_context,
        schema_context_objects,
        schema_context_config=run.schema_context_config,
    )
    decision = run_prompt(
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
        selected
        for selected in sanitized_decision.selected_objects
        if selected.object_id not in current_ids
    ]
    diagnostics: dict[str, object] = {
        "available_context": context_diagnostics,
        "planner": planner_diagnostics,
        "prompt_budget": prompt_budget_diagnostics(
            planning_prompt=planning_prompt,
            schema_context_config=run.schema_context_config,
        ),
        "rationale": sanitized_decision.rationale,
        "confidence": sanitized_decision.confidence,
    }
    return selected_additions, schema_context_objects, diagnostics


def _run_sql_recovery(run: TaskRun, *, best) -> RecoveryAction:
    """Attempt one SQL-focused recovery for a non-executable best attempt."""

    logger.info(
        "recovery action requested",
        instance_id=run.task.instance_id,
        action="sql",
        best_stage=best.stage,
    )
    repaired = run_prompt(
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
    attempt = evaluate_and_record_candidate(run, candidate=repaired, stage="recovery_sql")
    return RecoveryAction(
        kind="sql",
        trigger="best_attempt_not_executable",
        source_stage=best.stage,
        attempt_stage=attempt.stage,
        outcome="recovered" if attempt.execution_result.ok else "still_failed",
    )


def _run_candidate_review(run: TaskRun, *, best) -> CandidateReviewReport:
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
    review = run_prompt(
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
    annotated = current_best(run, preferred_stage=review.preferred_stage) or best
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
    best,
    review: CandidateReviewReport,
) -> RecoveryAction:
    """Attempt one semantic recovery after the critic identifies a concrete issue."""

    logger.info(
        "recovery action requested",
        instance_id=run.task.instance_id,
        action="semantic",
        best_stage=best.stage,
        focus=review.repair_focus,
    )
    repaired = run_prompt(
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
    attempt = evaluate_and_record_candidate(run, candidate=repaired, stage="recovery_semantic")
    return RecoveryAction(
        kind="semantic",
        trigger=review.repair_focus or "critic_requested_repair",
        source_stage=best.stage,
        attempt_stage=attempt.stage,
        outcome="recovered" if attempt.execution_result.ok else "still_failed",
    )


def _run_schema_recovery(
    run: TaskRun,
    *,
    best,
    trigger: str,
) -> RecoveryAction:
    """Expand schema context and regenerate one candidate when a schema miss is evident."""

    try:
        db_index = load_db_index(run.task.db)
    except Exception:
        return RecoveryAction(
            kind="schema",
            trigger=trigger,
            source_stage=best.stage,
            outcome="db_index_unavailable",
            details={
                "decision": None,
                "added_tables": [],
            },
        )

    expansion_query = schema_expansion_query(
        run.task.question,
        best,
        trigger,
        run.schema.selected_object_ids,
        run.schema.expanded_tables,
    )
    details: dict[str, Any] = {
        "expansion_query": expansion_query,
        "decision": None,
        "added_tables": [],
    }
    schema_context_cache = build_schema_context_cache(
        run.task.db,
        db_index=db_index,
        config=run.schema_context_config,
    )

    det_tables = deterministic_expansion_tables(best, run.schema, db_index)
    if det_tables:
        details["decision"] = {
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
        details["decision"] = {
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
    details.update(
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
                schema_context_config=run.schema_context_config,
            ),
        }
    )

    if not added_tables:
        action = RecoveryAction(
            kind="schema",
            trigger=trigger,
            source_stage=best.stage,
            outcome="no_new_tables",
            details=details,
        )
        run.schema_context = schema_context_with_expansion(
            run.schema_context,
            action.as_trace_payload(),
        )
        return action

    new_table_schemas = resolved.table_schemas or table_schemas_for_selection(expanded_schema)
    run.schema = expanded_schema
    run.table_schemas = new_table_schemas
    run.sql_reference_context = checked_schema_prompt(
        "schema_expansion_sql_reference_context",
        resolved.prompt_context,
        run.schema_context_config,
    )
    run.intent = augment_intent_with_value_groundings(
        run.intent,
        task=run.task,
        schema=expanded_schema,
        table_schemas=new_table_schemas,
    )

    batch = run_prompt(
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
        action = RecoveryAction(
            kind="schema",
            trigger=trigger,
            source_stage=best.stage,
            outcome="expanded_no_candidate",
            details=details,
        )
        run.schema_context = schema_context_with_expansion(
            run.schema_context,
            action.as_trace_payload(),
        )
        return action

    attempt = evaluate_and_record_candidate(
        run,
        candidate=batch.candidates[0],
        stage="recovery_schema",
    )
    action = RecoveryAction(
        kind="schema",
        trigger=trigger,
        source_stage=best.stage,
        attempt_stage=attempt.stage,
        outcome="expanded" if attempt.execution_result.ok else "expanded_failed",
        details=details,
    )
    run.schema_context = schema_context_with_expansion(
        run.schema_context,
        action.as_trace_payload(),
    )
    return action
