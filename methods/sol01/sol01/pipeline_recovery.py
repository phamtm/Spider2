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
from sol01.llm.sql_prompts import (
    candidate_review_prompt,
    semantic_repair_prompt,
    sql_generation_batch_prompt,
    sql_repair_prompt,
)
from sol01.models import (
    CandidateReviewReport,
    ConfidenceReport,
    SelectedSchemaObject,
    SQLCandidate,
    SQLCandidateBatch,
)
from sol01.pipeline_state import TaskRun, current_best
from sol01.pipeline_support import (
    checked_schema_prompt,
    evaluate_and_record_candidate,
    prompt_budget_diagnostics,
    run_prompt,
    run_schema_planning,
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
from sol01.schema.schema_context_cache import build_schema_context_cache
from sol01.workflow import (
    RECOVERY_KIND_SCHEMA,
    RECOVERY_KIND_SEMANTIC,
    RECOVERY_KIND_SQL,
    RECOVERY_OUTCOME_DB_INDEX_UNAVAILABLE,
    RECOVERY_OUTCOME_EXPANDED,
    RECOVERY_OUTCOME_EXPANDED_FAILED,
    RECOVERY_OUTCOME_EXPANDED_NO_CANDIDATE,
    RECOVERY_OUTCOME_NO_NEW_TABLES,
    RECOVERY_OUTCOME_RECOVERED,
    RECOVERY_OUTCOME_STILL_FAILED,
    RECOVERY_STAGE_SCHEMA,
    RECOVERY_STAGE_SEMANTIC,
    RECOVERY_STAGE_SQL,
    REPAIR_SKIPPED_REASON_ATTEMPT_BUDGET,
    REPAIR_SKIPPED_REASON_SEMANTIC_BUDGET,
    STOP_REASON_ATTEMPT_BUDGET_EXHAUSTED,
    STOP_REASON_NO_ATTEMPTS,
    STOP_REASON_NO_RECOVERY_NEEDED,
    STOP_REASON_REVIEW_COMPLETE,
    STOP_REASON_SCHEMA_RECOVERY_COMPLETE,
    STOP_REASON_SEMANTIC_REPAIR_BUDGET_EXHAUSTED,
    STOP_REASON_SQL_RECOVERY_COMPLETE,
    CandidateReviewTrace,
    RecoveryKind,
    RecoveryOutcome,
    RecoveryStopReason,
    RecoveryTrace,
    is_initial_attempt_stage,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class RecoveryAction:
    """Typed recovery action record that can be serialized into traces."""

    kind: RecoveryKind
    trigger: str
    source_stage: str
    outcome: RecoveryOutcome
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
    stop_reason: RecoveryStopReason = STOP_REASON_NO_ATTEMPTS

    while True:
        best = current_best(run)
        if best is None:
            stop_reason = STOP_REASON_NO_ATTEMPTS
            break

        schema_trigger = schema_expansion_trigger(best)
        if schema_trigger is not None:
            if len(run.attempts) >= run.policy.max_attempts:
                best.repair_skipped_reason = REPAIR_SKIPPED_REASON_ATTEMPT_BUDGET
                stop_reason = STOP_REASON_ATTEMPT_BUDGET_EXHAUSTED
                break

            action = _run_schema_recovery(run, best=best, trigger=schema_trigger)
            actions.append(action)
            if action.outcome == RECOVERY_OUTCOME_EXPANDED:
                stop_reason = STOP_REASON_SCHEMA_RECOVERY_COMPLETE
                break
            if action.outcome != RECOVERY_OUTCOME_EXPANDED_FAILED:
                stop_reason = action.outcome
                break
            continue

        if not best.execution_result.ok:
            if len(run.attempts) >= run.policy.max_attempts:
                best.repair_skipped_reason = REPAIR_SKIPPED_REASON_ATTEMPT_BUDGET
                stop_reason = STOP_REASON_ATTEMPT_BUDGET_EXHAUSTED
                break

            action = _run_sql_recovery(run, best=best)
            actions.append(action)
            if action.outcome == RECOVERY_OUTCOME_RECOVERED:
                stop_reason = STOP_REASON_SQL_RECOVERY_COMPLETE
                break
            continue

        if not review_ran:
            review = _run_candidate_review(run, best=best)
            review_ran = True
            if not review.should_repair:
                stop_reason = STOP_REASON_REVIEW_COMPLETE
                break
            if remaining_semantic_repairs < 1:
                best = current_best(run, preferred_stage=review.preferred_stage) or best
                best.repair_skipped_reason = REPAIR_SKIPPED_REASON_SEMANTIC_BUDGET
                stop_reason = STOP_REASON_SEMANTIC_REPAIR_BUDGET_EXHAUSTED
                break
            if len(run.attempts) >= run.policy.max_attempts:
                best = current_best(run, preferred_stage=review.preferred_stage) or best
                best.repair_skipped_reason = REPAIR_SKIPPED_REASON_ATTEMPT_BUDGET
                stop_reason = STOP_REASON_ATTEMPT_BUDGET_EXHAUSTED
                break

            actions.append(_run_semantic_recovery(run, best=best, review=review))
            remaining_semantic_repairs -= 1
            continue

        stop_reason = STOP_REASON_NO_RECOVERY_NEEDED
        break

    run.recovery_payload = RecoveryTrace(
        priority_order=list(DEFAULT_RECOVERY_SIGNAL_POLICY.priority_order),
        attempts_before_recovery=len(
            [attempt for attempt in run.attempts if is_initial_attempt_stage(attempt.stage)]
        ),
        attempts_after_recovery=len(run.attempts),
        max_attempts=run.policy.max_attempts,
        semantic_repairs_allowed=run.policy.semantic_repairs,
        semantic_repairs_remaining=remaining_semantic_repairs,
        actions=[action.as_trace_payload() for action in actions],
        stop_reason=stop_reason,
    )
    return run


def _select_expansion_objects(
    run: TaskRun,
    *,
    expansion_query: str,
    schema_context_cache: Any,
) -> tuple[list[SelectedSchemaObject], list[Any], dict[str, object]]:
    """Select and sanitize schema objects for one expansion attempt."""

    expansion_task = run.task.model_copy(
        update={"question": f"{run.task.question}\n\nSchema expansion evidence:\n{expansion_query}"}
    )
    planning = run_schema_planning(
        task=expansion_task,
        docs_context=run.docs_context,
        client=run.client,
        prompt_hashes=run.prompt_hashes,
        schema_context_config=run.schema_context_config,
        schema_context_cache=schema_context_cache,
    )
    current_ids = set(run.schema.selected_object_ids)
    selected_additions = [
        selected
        for selected in planning.decision.selected_objects
        if selected.object_id not in current_ids
    ]
    diagnostics: dict[str, object] = {
        "available_context": planning.context_diagnostics,
        "planner": planning.planner_diagnostics,
        "prompt_budget": planning.prompt_budget,
        "rationale": planning.decision.rationale,
        "confidence": planning.decision.confidence,
    }
    return selected_additions, planning.schema_context_objects, diagnostics


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
    attempt = evaluate_and_record_candidate(run, candidate=repaired, stage=RECOVERY_STAGE_SQL)
    return RecoveryAction(
        kind=RECOVERY_KIND_SQL,
        trigger="best_attempt_not_executable",
        source_stage=best.stage,
        attempt_stage=attempt.stage,
        outcome=RECOVERY_OUTCOME_RECOVERED
        if attempt.execution_result.ok
        else RECOVERY_OUTCOME_STILL_FAILED,
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
    run.candidate_review_payload = CandidateReviewTrace(
        review_reason=review_reason,
        **review.model_dump(mode="json"),
    )
    annotated = current_best(run, preferred_stage=review.preferred_stage) or best
    annotated.critic = {
        "confidence": review.confidence,
        "issues": review.issues,
        "should_repair": review.should_repair,
        "repair_focus": review.repair_focus,
    }
    annotated.candidate_review = run.candidate_review_payload.model_dump(mode="json")

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
    attempt = evaluate_and_record_candidate(run, candidate=repaired, stage=RECOVERY_STAGE_SEMANTIC)
    return RecoveryAction(
        kind=RECOVERY_KIND_SEMANTIC,
        trigger=review.repair_focus or "critic_requested_repair",
        source_stage=best.stage,
        attempt_stage=attempt.stage,
        outcome=RECOVERY_OUTCOME_RECOVERED
        if attempt.execution_result.ok
        else RECOVERY_OUTCOME_STILL_FAILED,
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
            kind=RECOVERY_KIND_SCHEMA,
            trigger=trigger,
            source_stage=best.stage,
            outcome=RECOVERY_OUTCOME_DB_INDEX_UNAVAILABLE,
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
            kind=RECOVERY_KIND_SCHEMA,
            trigger=trigger,
            source_stage=best.stage,
            outcome=RECOVERY_OUTCOME_NO_NEW_TABLES,
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
            kind=RECOVERY_KIND_SCHEMA,
            trigger=trigger,
            source_stage=best.stage,
            outcome=RECOVERY_OUTCOME_EXPANDED_NO_CANDIDATE,
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
        stage=RECOVERY_STAGE_SCHEMA,
    )
    action = RecoveryAction(
        kind=RECOVERY_KIND_SCHEMA,
        trigger=trigger,
        source_stage=best.stage,
        attempt_stage=attempt.stage,
        outcome=RECOVERY_OUTCOME_EXPANDED
        if attempt.execution_result.ok
        else RECOVERY_OUTCOME_EXPANDED_FAILED,
        details=details,
    )
    run.schema_context = schema_context_with_expansion(
        run.schema_context,
        action.as_trace_payload(),
    )
    return action
