"""Recover from incomplete schema selection during candidate evaluation."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Protocol

from sol01.candidates.evaluator import evaluate_candidate
from sol01.candidates.scoring import _best_attempt
from sol01.infra.config import SchemaRetrievalConfig
from sol01.llm.prompt_builders import (
    _sql_generation_batch_prompt,
    sanitize_hybrid_planning_decision,
    schema_expansion_trigger,
)
from sol01.models import (
    HybridPlanningDecision,
    SchemaSelection,
    SelectedSchemaObject,
    SQLCandidateBatch,
)
from sol01.schema.hybrid_retrieval import retrieve_schema_objects
from sol01.schema.resolver import resolve_schema_context
from sol01.schema.retrieval import load_db_index
from sol01.schema.retrieval_index import build_retrieval_index


class ExpansionContext(Protocol):
    """Pipeline context fields used by schema expansion."""

    task: Any
    client: Any
    intent: Any
    schema: SchemaSelection
    table_schemas: dict[str, Any]
    sql_reference_context: str
    docs_context: str
    prompt_hashes: dict[str, str]
    schema_retrieval_config: SchemaRetrievalConfig
    schema_retrieval: dict[str, Any]


RunPrompt = Callable[..., Any]
BuildPlanningPrompt = Callable[..., str]
PromptBudgetDiagnostics = Callable[..., dict[str, object]]
LogCandidate = Callable[[str, dict[str, Any]], None]
RebuildContext = Callable[..., ExpansionContext]


def attempt_schema_expansion(
    ctx: ExpansionContext,
    attempts: list[dict[str, Any]],
    best_attempt: dict[str, Any] | None,
    *,
    run_prompt: RunPrompt,
    build_planning_prompt: BuildPlanningPrompt,
    prompt_budget_diagnostics: PromptBudgetDiagnostics,
    rebuild_context: RebuildContext,
    log_candidate: LogCandidate,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, ExpansionContext | None]:
    """Run one schema-expansion recovery attempt when evidence warrants it."""

    if best_attempt is None:
        return best_attempt, None, None

    trigger = schema_expansion_trigger(best_attempt)
    if trigger is None:
        return best_attempt, None, None

    try:
        db_index = load_db_index(ctx.task.db)
    except Exception:
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
            run_prompt=run_prompt,
            build_planning_prompt=build_planning_prompt,
            prompt_budget_diagnostics=prompt_budget_diagnostics,
        )
        expansion_payload["decision"] = {
            "source": "retrieval",
            "selected_object_ids": [selected.object_id for selected in selected_additions],
        }

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
            "prompt_budget": prompt_budget_diagnostics(
                sql_reference_context=resolved.prompt_context,
                schema_retrieval_config=ctx.schema_retrieval_config,
            ),
        }
    )
    if not added_tables:
        expansion_payload["outcome"] = "no_new_tables"
        return best_attempt, expansion_payload, None

    expanded_ctx = rebuild_context(
        ctx,
        expanded_schema,
        table_schemas=resolved.table_schemas,
        sql_reference_context=resolved.prompt_context,
    )

    expansion_stage = "schema_expansion"
    expansion_payload["expansion_attempt_stage"] = expansion_stage
    batch = run_prompt(
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
    log_candidate(ctx.task.instance_id, expansion_attempt)

    expansion_payload["outcome"] = (
        "expanded" if expansion_attempt["execution_result"]["ok"] else "expanded_failed"
    )
    expanded_ctx.schema_retrieval = _schema_retrieval_with_expansion(
        ctx.schema_retrieval,
        expansion_payload,
    )
    return _best_attempt(attempts), expansion_payload, expanded_ctx


def _retrieve_expansion_objects(
    ctx: ExpansionContext,
    attempt: dict[str, Any],
    trigger: str,
    *,
    retrieval_index: Any,
    expansion_query: str,
    run_prompt: RunPrompt,
    build_planning_prompt: BuildPlanningPrompt,
    prompt_budget_diagnostics: PromptBudgetDiagnostics,
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
    planning_prompt = build_planning_prompt(
        expansion_task,
        ctx.docs_context,
        retrieved_objects,
        schema_retrieval_config=ctx.schema_retrieval_config,
    )
    decision = run_prompt(
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
        "prompt_budget": prompt_budget_diagnostics(
            planning_prompt=planning_prompt,
            schema_retrieval_config=ctx.schema_retrieval_config,
        ),
        "rationale": sanitized_decision.rationale,
        "confidence": sanitized_decision.confidence,
    }
    return selected_additions, retrieved_objects, diagnostics


def _resolve_expanded_schema(
    ctx: ExpansionContext,
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
    ctx: ExpansionContext,
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
