"""Recover from incomplete schema selection during candidate evaluation."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from sol01.candidates.evaluator import evaluate_candidate
from sol01.candidates.scoring import best_attempt as choose_best_attempt
from sol01.llm.client import LLMClient
from sol01.llm.prompt_builders import (
    sanitize_schema_planning_decision,
    schema_expansion_trigger,
    sql_generation_batch_prompt,
)
from sol01.models import (
    AttemptRecord,
    SchemaPlanningDecision,
    SchemaSelection,
    SelectedSchemaObject,
    SQLCandidateBatch,
)
from sol01.pipeline import TaskContext
from sol01.schema.db_index import load_db_index
from sol01.schema.resolver import resolve_schema_context
from sol01.schema.schema_context import select_schema_context_objects
from sol01.schema.schema_context_cache import build_schema_context_cache

RunPrompt = Callable[[LLMClient, str, str, type[BaseModel], str], BaseModel]
BuildPlanningPrompt = Callable[..., str]
PromptBudgetDiagnostics = Callable[..., dict[str, object]]
LogCandidate = Callable[[str, AttemptRecord], None]
RebuildContext = Callable[..., TaskContext]


def attempt_schema_expansion(
    ctx: TaskContext,
    attempts: list[AttemptRecord],
    current_best: AttemptRecord | None,
    *,
    run_prompt: RunPrompt,
    build_planning_prompt: BuildPlanningPrompt,
    prompt_budget_diagnostics: PromptBudgetDiagnostics,
    rebuild_context: RebuildContext,
    log_candidate: LogCandidate,
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

    expansion_payload: dict[str, Any] = {
        "trigger": trigger,
        "expansion_query": _schema_expansion_query(ctx, current_best, trigger),
        "decision": None,
        "added_tables": [],
        "outcome": "no_new_tables",
    }
    schema_context_cache = build_schema_context_cache(
        ctx.task.db,
        db_index=db_index,
        config=ctx.schema_context_config,
    )

    deterministic_tables = _deterministic_expansion_tables(current_best, ctx.schema, db_index)
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
        schema_context_objects: list[Any] = []
        planner_diagnostics: dict[str, object] = {}
    else:
        selected_additions, schema_context_objects, planner_diagnostics = _select_expansion_objects(
            ctx,
            current_best,
            trigger,
            schema_context_cache=schema_context_cache,
            expansion_query=expansion_payload["expansion_query"],
            run_prompt=run_prompt,
            build_planning_prompt=build_planning_prompt,
            prompt_budget_diagnostics=prompt_budget_diagnostics,
        )
        expansion_payload["decision"] = {
            "source": "schema_context",
            "selected_object_ids": [selected.object_id for selected in selected_additions],
        }

    expanded_schema, resolved, added_tables = _resolve_expanded_schema(
        ctx,
        selected_additions,
        schema_context_cache=schema_context_cache,
        db_index=db_index,
        schema_context_evidence=schema_context_objects,
        expansion_query=expansion_payload["expansion_query"],
    )
    expansion_payload.update(
        {
            "schema_context_objects": _schema_context_object_trace(schema_context_objects),
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

    expanded_ctx = rebuild_context(
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
        expanded_ctx.schema_context = _schema_context_with_expansion(
            ctx.schema_context,
            expansion_payload,
        )
        return choose_best_attempt(attempts), expansion_payload, expanded_ctx

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
    expanded_ctx.schema_context = _schema_context_with_expansion(
        ctx.schema_context,
        expansion_payload,
    )
    return choose_best_attempt(attempts), expansion_payload, expanded_ctx


def _select_expansion_objects(
    ctx: TaskContext,
    attempt: AttemptRecord,
    trigger: str,
    *,
    schema_context_cache: Any,
    expansion_query: str,
    run_prompt: RunPrompt,
    build_planning_prompt: BuildPlanningPrompt,
    prompt_budget_diagnostics: PromptBudgetDiagnostics,
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
            "question": (
                f"{ctx.task.question}\n\n"
                "Schema expansion evidence:\n"
                f"{_schema_expansion_query(ctx, attempt, trigger)}"
            )
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
    diagnostics = {
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


def _resolve_expanded_schema(
    ctx: TaskContext,
    selected_additions: list[SelectedSchemaObject],
    *,
    schema_context_cache: Any,
    db_index: dict[str, Any],
    schema_context_evidence: list[Any],
    expansion_query: str,
) -> tuple[SchemaSelection, Any, list[str]]:
    """Resolve current and newly selected objects into one compact schema context."""

    object_ids = {schema_object.object_id for schema_object in schema_context_cache.objects}
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
        canonical_schema_objects=schema_context_cache.objects,
        db_index=db_index,
        question=ctx.task.question,
        schema_context_evidence=schema_context_evidence,
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
    ctx: TaskContext,
    attempt: AttemptRecord,
    trigger: str,
) -> str:
    """Build the augmented planning question used to find expansion candidates."""

    parts = [
        f"Original question: {ctx.task.question}",
        f"Schema expansion trigger: {trigger}",
        f"Failed SQL: {attempt.sql}",
        f"Validation errors: {'; '.join(attempt.validation.errors) or 'none'}",
        f"Validation warnings: {'; '.join(attempt.validation.warnings) or 'none'}",
        f"Execution error: {attempt.execution_result.error or 'none'}",
        "Current selected object ids: " + ", ".join(ctx.schema.selected_object_ids),
        "Current allowed tables: " + ", ".join(ctx.schema.expanded_tables),
    ]
    return "\n".join(parts)


def _deterministic_expansion_tables(
    attempt: AttemptRecord,
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


def _table_names_from_schema_errors(attempt: AttemptRecord) -> list[str]:
    """Extract table-like names explicitly reported by validation or execution."""

    names: list[str] = []
    for error in attempt.validation.errors:
        match = re.search(r"Unknown table referenced:\s*([A-Za-z0-9_.$\"]+)", error)
        if match:
            names.append(match.group(1).strip('".'))

    execution_error = attempt.execution_result.error or ""
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


def _schema_context_object_trace(schema_context_objects: list[Any]) -> list[dict[str, object]]:
    """Render expansion schema-context objects in the task trace."""

    return [
        {
            "object_id": item.schema_object.object_id,
            "object_type": item.schema_object.object_type,
            "name": item.schema_object.name,
            "table_name": item.schema_object.table_name,
            "rank": item.rank,
            "score": item.score,
        }
        for item in schema_context_objects
    ]


def _schema_context_with_expansion(
    schema_context: dict[str, Any],
    expansion_payload: dict[str, Any],
) -> dict[str, Any]:
    """Attach expansion diagnostics to the schema context trace block."""

    updated = dict(schema_context)
    expansions = list(updated.get("expansions", []))
    expansions.append(
        {
            "expansion_query": expansion_payload.get("expansion_query"),
            "schema_context_objects": expansion_payload.get("schema_context_objects", []),
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
