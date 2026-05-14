"""Pure schema helpers for expansion trigger detection and object resolution."""

from __future__ import annotations

import re
from typing import Any

from sol01.models import (
    AttemptRecord,
    SchemaSelection,
    SelectedSchemaObject,
)
from sol01.schema.resolver import resolve_schema_context


def schema_expansion_query(
    question: str,
    attempt: AttemptRecord,
    trigger: str,
    selected_object_ids: list[str],
    expanded_tables: list[str],
) -> str:
    """Build the augmented planning question used to find expansion candidates."""

    parts = [
        f"Original question: {question}",
        f"Schema expansion trigger: {trigger}",
        f"Failed SQL: {attempt.sql}",
        f"Validation errors: {'; '.join(attempt.validation.errors) or 'none'}",
        f"Validation warnings: {'; '.join(attempt.validation.warnings) or 'none'}",
        f"Execution error: {attempt.execution_result.error or 'none'}",
        "Current selected object ids: " + ", ".join(selected_object_ids),
        "Current allowed tables: " + ", ".join(expanded_tables),
    ]
    return "\n".join(parts)


def deterministic_expansion_tables(
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


def resolve_expanded_schema(
    db: str,
    question: str,
    current_schema: SchemaSelection,
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
    for object_id in current_schema.selected_object_ids:
        if object_id in object_ids and object_id not in seen:
            selected_objects.append(SelectedSchemaObject(object_id=object_id, role="unknown"))
            seen.add(object_id)
    for selected in selected_additions:
        if selected.object_id in object_ids and selected.object_id not in seen:
            selected_objects.append(selected)
            seen.add(selected.object_id)

    resolved = resolve_schema_context(
        db=db,
        selected_objects=selected_objects,
        canonical_schema_objects=schema_context_cache.objects,
        db_index=db_index,
        question=question,
        schema_context_evidence=schema_context_evidence,
    )
    current_tables = set(current_schema.expanded_tables)
    added_tables = [table for table in resolved.resolved_tables if table not in current_tables]
    expanded_schema = current_schema.model_copy(
        update={
            "selected_object_ids": [selected.object_id for selected in selected_objects],
            "expanded_tables": list(resolved.resolved_tables),
            "diagnostics": {
                **current_schema.diagnostics,
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


def schema_context_object_trace(schema_context_objects: list[Any]) -> list[dict[str, object]]:
    """Render expansion schema-context objects in the task trace."""

    return [
        {
            "object_id": item.schema_object.object_id,
            "object_type": item.schema_object.object_type,
            "name": item.schema_object.name,
            "table_name": item.schema_object.table_name,
            "position": item.position,
        }
        for item in schema_context_objects
    ]


def schema_context_with_expansion(
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


def _table_names_from_schema_errors(attempt: AttemptRecord) -> list[str]:
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
    parts = [part for part in table_name.lower().split(".") if part]
    aliases = {parts[-1]} if parts else set()
    for start in range(len(parts)):
        aliases.add(".".join(parts[start:]))
    return sorted(aliases, key=len, reverse=True)
