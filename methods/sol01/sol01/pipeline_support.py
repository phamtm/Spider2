"""Shared helpers used by multiple pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from sol01.candidates.evaluator import evaluate_candidate
from sol01.infra.config import SchemaContextConfig
from sol01.infra.logging import get_logger
from sol01.llm.client import LLMClient
from sol01.llm.planning_prompts import (
    sanitize_schema_planning_decision,
    schema_context_planning_user_prompt,
)
from sol01.llm.prompt_budget import enforce_prompt_budget
from sol01.models import (
    AttemptRecord,
    SchemaContextObject,
    SchemaPlanningDecision,
    SQLCandidate,
    TableSchema,
    Task,
)
from sol01.pipeline_state import TaskRun
from sol01.schema.schema_context import build_available_schema_context
from sol01.schema.schema_context_cache import SchemaContextCache, build_schema_context_cache

logger = get_logger(__name__)


@dataclass(frozen=True)
class SchemaPlanningRun:
    """Shared planning result used by initial planning and schema recovery."""

    cache: SchemaContextCache
    schema_context_objects: list[SchemaContextObject]
    context_diagnostics: dict[str, object]
    planning_prompt: str
    decision: SchemaPlanningDecision
    planner_diagnostics: dict[str, object]
    prompt_budget: dict[str, object]


def build_planning_prompt(
    task: Task,
    docs_context: str | None,
    schema_context_objects: list[Any],
    *,
    schema_context_config: SchemaContextConfig,
) -> str:
    """Build the planner prompt for one task."""

    return schema_context_planning_user_prompt(
        task,
        task.db,
        docs_context,
        schema_context_objects,
        schema_context_config=schema_context_config,
    )


def checked_schema_prompt(
    prompt_name: str,
    prompt: str,
    schema_context_config: SchemaContextConfig,
) -> str:
    """Enforce the configured schema-context budget for one prompt fragment."""

    return enforce_prompt_budget(prompt_name, prompt, schema_context_config.max_schema_prompt_chars)


def prompt_budget_diagnostics(
    *,
    planning_prompt: str | None = None,
    sql_reference_context: str | None = None,
    schema_context_config: SchemaContextConfig,
) -> dict[str, object]:
    """Capture prompt-size diagnostics for traces and debugging."""

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


def run_schema_planning(
    *,
    task: Task,
    docs_context: str | None,
    client: LLMClient,
    prompt_hashes: dict[str, str],
    schema_context_config: SchemaContextConfig,
    db_index: dict[str, TableSchema] | None = None,
    schema_context_cache: SchemaContextCache | None = None,
) -> SchemaPlanningRun:
    """Run the shared planner flow and return the sanitized decision plus diagnostics."""

    cache = schema_context_cache or build_schema_context_cache(
        task.db,
        db_index=db_index,
        config=schema_context_config,
    )
    schema_context_objects, context_diagnostics = build_available_schema_context(cache)
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
        decision, schema_context_objects
    )
    return SchemaPlanningRun(
        cache=cache,
        schema_context_objects=schema_context_objects,
        context_diagnostics=context_diagnostics,
        planning_prompt=planning_prompt,
        decision=sanitized_decision,
        planner_diagnostics=planner_diagnostics,
        prompt_budget=prompt_budget_diagnostics(
            planning_prompt=planning_prompt,
            schema_context_config=schema_context_config,
        ),
    )


def log_candidate(instance_id: str, attempt: AttemptRecord) -> None:
    """Emit a compact candidate evaluation log event."""

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


def run_prompt(
    client: LLMClient,
    *,
    prompt_hashes: dict[str, str],
    prompt_name: str,
    output_type: type[BaseModel],
    user_prompt: str,
) -> BaseModel:
    """Run one structured prompt and record its stable prompt hash."""

    prompt = client.load_prompt(prompt_name)
    prompt_hashes.setdefault(prompt_name, prompt.sha256)
    return client.run_structured_with_prompt(
        user_prompt,
        prompt=prompt,
        output_type=output_type,
    )


def evaluate_and_record_candidate(
    run: TaskRun,
    *,
    candidate: SQLCandidate,
    stage: str,
) -> AttemptRecord:
    """Evaluate one candidate, append it to the run, and log the result."""

    attempt = evaluate_candidate(
        task=run.task,
        candidate=candidate,
        intent=run.intent,
        schema=run.schema,
        table_schemas=run.table_schemas,
        stage=stage,
    )
    run.attempts.append(attempt)
    log_candidate(run.task.instance_id, attempt)
    return attempt
