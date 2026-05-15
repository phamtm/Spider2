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
from sol01.llm.sql_prompts import sql_generation_batch_prompt, sql_repair_prompt
from sol01.models import SelectedSchemaObject, SQLCandidate, SQLCandidateBatch
from sol01.pipeline_state import TaskRun, current_best
from sol01.pipeline_support import (
    checked_schema_prompt,
    evaluate_and_record_candidate,
    prompt_budget_diagnostics,
    run_prompt,
    run_schema_grounding,
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
    RECOVERY_KIND_SQL,
    RECOVERY_OUTCOME_DB_INDEX_UNAVAILABLE,
    RECOVERY_OUTCOME_EXPANDED,
    RECOVERY_OUTCOME_EXPANDED_FAILED,
    RECOVERY_OUTCOME_EXPANDED_NO_CANDIDATE,
    RECOVERY_OUTCOME_NO_NEW_TABLES,
    RECOVERY_OUTCOME_RECOVERED,
    RECOVERY_OUTCOME_STILL_FAILED,
    RECOVERY_STAGE_SCHEMA,
    RECOVERY_STAGE_SQL,
    REPAIR_SKIPPED_REASON_ATTEMPT_BUDGET,
    STOP_REASON_ATTEMPT_BUDGET_EXHAUSTED,
    STOP_REASON_NO_ATTEMPTS,
    STOP_REASON_NO_RECOVERY_NEEDED,
    STOP_REASON_SCHEMA_RECOVERY_COMPLETE,
    STOP_REASON_SQL_RECOVERY_COMPLETE,
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

        if best.execution_result.ok:
            stop_reason = STOP_REASON_NO_RECOVERY_NEEDED
            break

        if len(run.attempts) >= run.policy.max_attempts:
            best.repair_skipped_reason = REPAIR_SKIPPED_REASON_ATTEMPT_BUDGET
            stop_reason = STOP_REASON_ATTEMPT_BUDGET_EXHAUSTED
            break

        action = _run_sql_recovery(run, best=best)
        actions.append(action)
        if action.outcome == RECOVERY_OUTCOME_RECOVERED:
            stop_reason = STOP_REASON_SQL_RECOVERY_COMPLETE
            break

    run.recovery_payload = RecoveryTrace(
        priority_order=list(DEFAULT_RECOVERY_SIGNAL_POLICY.priority_order),
        attempts_before_recovery=len(
            [attempt for attempt in run.attempts if is_initial_attempt_stage(attempt.stage)]
        ),
        attempts_after_recovery=len(run.attempts),
        max_attempts=run.policy.max_attempts,
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
            run.schema_grounding,
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
                sql_reference_context=resolved.sql_prompt_context,
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
        resolved.sql_prompt_context,
        run.schema_context_config,
    )
    run.intent = augment_intent_with_value_groundings(
        run.intent,
        task=run.task,
        schema=expanded_schema,
        table_schemas=new_table_schemas,
    )
    grounding = run_schema_grounding(
        task=run.task,
        intent=run.intent,
        available_tables=list(new_table_schemas),
        table_schemas=new_table_schemas,
        sql_prompt_context=run.sql_reference_context,
        client=run.client,
        prompt_hashes=run.prompt_hashes,
        schema_context_config=run.schema_context_config,
    )
    run.schema_grounding = grounding.grounding
    run.schema.diagnostics["schema_grounding"] = grounding.diagnostics
    details["schema_grounding"] = grounding.diagnostics

    batch = run_prompt(
        run.client,
        prompt_hashes=run.prompt_hashes,
        prompt_name="sql_generation_batch",
        output_type=SQLCandidateBatch,
        user_prompt=sql_generation_batch_prompt(
            run.task,
            run.intent,
            run.schema_grounding,
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
