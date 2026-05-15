"""Per-task solver pipeline stages used by coordinator.run_task."""

from __future__ import annotations

import json
from pathlib import Path

from sol01.candidates.verification import (
    augment_intent_with_value_groundings,
)
from sol01.infra.logging import get_logger
from sol01.infra.strings import question_preview
from sol01.llm.sql_prompts import sql_generation_batch_prompt
from sol01.loading.docs import load_document_text
from sol01.models import (
    FinalAnswer,
    SchemaSelection,
    SQLCandidateBatch,
    Task,
)
from sol01.output.output import (
    RunPaths,
    should_skip_task,
)
from sol01.pipeline_state import TaskRun
from sol01.pipeline_support import (
    checked_schema_prompt,
    evaluate_and_record_candidate,
    prompt_budget_diagnostics,
    run_prompt,
    run_schema_grounding,
    run_schema_planning,
)
from sol01.schema.db_index import load_db_index
from sol01.schema.resolver import resolve_schema_context
from sol01.workflow import TASK_STATUS_SKIPPED, initial_attempt_stage

logger = get_logger(__name__)


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
        status=TASK_STATUS_SKIPPED,
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
    planning = run_schema_planning(
        task=run.task,
        docs_context=run.docs_context,
        client=run.client,
        prompt_hashes=run.prompt_hashes,
        schema_context_config=run.schema_context_config,
        db_index=db_index,
    )
    resolved = resolve_schema_context(
        db=run.task.db,
        selected_objects=planning.decision.selected_objects,
        canonical_schema_objects=planning.cache.objects,
        db_index=db_index,
        question=run.task.question,
        schema_context_evidence=planning.schema_context_objects,
        constraints=planning.decision.constraints,
    )
    sql_ref_context = checked_schema_prompt(
        "sql_reference_context",
        resolved.sql_prompt_context,
        run.schema_context_config,
    )
    selected_object_ids = [selected.object_id for selected in planning.decision.selected_objects]
    resolved_tables = list(resolved.resolved_tables)
    schema_selection = SchemaSelection(
        db=run.task.db,
        selected_object_ids=selected_object_ids,
        expanded_tables=resolved_tables,
        rationale=planning.decision.rationale,
        confidence=planning.decision.confidence,
    )
    run.intent = augment_intent_with_value_groundings(
        planning.decision.intent,
        task=run.task,
        schema=schema_selection,
        table_schemas=resolved.table_schemas,
    )
    grounding = run_schema_grounding(
        task=run.task,
        intent=run.intent,
        available_tables=list(resolved.table_schemas),
        table_schemas=resolved.table_schemas,
        sql_prompt_context=sql_ref_context,
        client=run.client,
        prompt_hashes=run.prompt_hashes,
        schema_context_config=run.schema_context_config,
    )
    prompt_budget = prompt_budget_diagnostics(
        planning_prompt=planning.planning_prompt,
        sql_reference_context=sql_ref_context,
        schema_context_config=run.schema_context_config,
    )

    run.schema = schema_selection.model_copy(
        update={
            "diagnostics": {
                "planning_prompt_chars": len(planning.planning_prompt),
                "sql_reference_context_chars": len(sql_ref_context),
                "schema_grounding_prompt_chars": grounding.diagnostics["prompt_chars"],
                "max_schema_prompt_chars": run.schema_context_config.max_schema_prompt_chars,
                "prompt_budget": prompt_budget,
                "schema_context_object_count": len(planning.schema_context_objects),
                "selected_objects": [
                    s.model_dump(mode="json") for s in planning.decision.selected_objects
                ],
                "resolved_tables": resolved_tables,
                "planner": planning.planner_diagnostics,
                "resolver": resolved.diagnostics,
                "schema_grounding": grounding.diagnostics,
            },
        }
    )
    run.schema_grounding = grounding.grounding
    run.table_schemas = resolved.table_schemas
    run.sql_reference_context = sql_ref_context
    run.schema_context = {
        "cache": {
            "db": planning.cache.db,
            "cache_key": planning.cache.cache_key,
            "object_count": len(planning.cache.objects),
        },
        "available_context": planning.context_diagnostics,
        "schema_context_objects": [
            {
                "object_id": item.schema_object.object_id,
                "object_type": item.schema_object.object_type,
                "name": item.schema_object.name,
                "table_name": item.schema_object.table_name,
                "position": item.position,
            }
            for item in planning.schema_context_objects
        ],
        "planner": planning.planner_diagnostics,
        "resolver": resolved.diagnostics,
        "schema_grounding": grounding.diagnostics,
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


def generate_initial_candidates(run: TaskRun) -> TaskRun:
    """Generate the configured initial SQL candidates and append them to run.attempts."""

    logger.info(
        "generating candidates",
        instance_id=run.task.instance_id,
        initial_candidates=run.policy.initial_candidates,
        max_attempts=run.policy.max_attempts,
    )
    candidate_limit = min(
        run.policy.initial_candidates, run.policy.max_attempts - len(run.attempts)
    )
    if candidate_limit <= 0:
        return run

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
            candidate_count=candidate_limit,
        ),
    )
    for i, candidate in enumerate(batch.candidates[:candidate_limit]):
        evaluate_and_record_candidate(
            run,
            candidate=candidate,
            stage=initial_attempt_stage(i + 1),
        )
    return run
