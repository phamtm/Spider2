"""Resolve selected logical schema objects to concrete physical table context."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sol01.models import (
    ResolvedSchemaContext,
    SchemaContextObject,
    SchemaObject,
    SchemaPlanningConstraints,
    SelectedSchemaObject,
    TableSchema,
)
from sol01.schema.family_constraints import merge_question_constraints
from sol01.schema.family_resolution import (
    canonical_family_schemas,
    physical_tables_for_object,
    resolve_family,
    stable_sorted_tables,
    table_lookup,
)
from sol01.schema.resolved_prompt import render_prompt_context


def resolve_schema_context(
    *,
    db: str,
    selected_objects: Sequence[SelectedSchemaObject],
    canonical_schema_objects: Sequence[SchemaObject],
    db_index: Mapping[str, TableSchema],
    question: str,
    schema_context_evidence: Sequence[SchemaContextObject] = (),
    constraints: SchemaPlanningConstraints | None = None,
) -> ResolvedSchemaContext:
    """Resolve planner-selected schema objects into allowed tables and prompt context."""

    constraints = merge_question_constraints(constraints, question)
    object_by_id = {
        schema_object.object_id: schema_object for schema_object in canonical_schema_objects
    }
    tables_by_name = table_lookup(db_index)
    selected_table_names: list[str] = []
    warnings: list[str] = []
    resolution_entries: list[dict[str, object]] = []

    for selected in selected_objects:
        schema_object = object_by_id.get(selected.object_id)
        if schema_object is None:
            warnings.append(f"Selected schema object is not canonical: {selected.object_id}")
            resolution_entries.append(
                {
                    "object_id": selected.object_id,
                    "resolved_tables": [],
                    "warning": "missing_object",
                }
            )
            continue

        if schema_object.object_type == "family":
            resolved, family_warnings, reason, details = resolve_family(
                schema_object,
                constraints=constraints,
                question=question,
            )
            selected_table_names.extend(resolved)
            warnings.extend(family_warnings)
            resolution_entries.append(
                {
                    "object_id": selected.object_id,
                    "object_type": "family",
                    "resolved_tables": resolved,
                    "reason": reason,
                    **details,
                }
            )
            continue

        resolved = physical_tables_for_object(schema_object)
        selected_table_names.extend(resolved)
        resolution_entries.append(
            {
                "object_id": selected.object_id,
                "object_type": schema_object.object_type,
                "resolved_tables": resolved,
                "reason": "exact_physical_table",
            }
        )

    allowed_tables = stable_sorted_tables(selected_table_names)
    table_schemas = {
        table_name: tables_by_name[table_name.lower()]
        for table_name in allowed_tables
        if table_name.lower() in tables_by_name
    }
    table_schemas.update(
        canonical_family_schemas(
            selected_objects=selected_objects,
            object_by_id=object_by_id,
            table_lookup=tables_by_name,
            existing_tables=table_schemas,
        )
    )
    missing_schemas = [
        table_name for table_name in allowed_tables if table_name not in table_schemas
    ]
    if missing_schemas:
        warnings.append(f"Resolved table schemas are missing for: {', '.join(missing_schemas)}")

    diagnostics: dict[str, object] = {
        "selected_object_count": len(selected_objects),
        "schema_context_object_count": len(schema_context_evidence),
        "resolution_entries": resolution_entries,
        "warnings": warnings,
    }
    prompt_context = render_prompt_context(
        db=db,
        selected_objects=selected_objects,
        object_by_id=object_by_id,
        allowed_tables=allowed_tables,
        table_schemas=table_schemas,
        schema_context_evidence=schema_context_evidence,
        diagnostics=diagnostics,
    )
    return ResolvedSchemaContext(
        db=db,
        selected_objects=list(selected_objects),
        schema_context_objects=list(schema_context_evidence),
        resolved_tables=allowed_tables,
        table_schemas=table_schemas,
        prompt_context=prompt_context,
        diagnostics=diagnostics,
    )
