"""Render resolved schema selections into compact SQL prompt context."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from sol01.infra.policy import DEFAULT_SCHEMA_RENDER_POLICY
from sol01.models import SchemaContextObject, SchemaObject, SelectedSchemaObject, TableSchema
from sol01.schema.family_resolution import (
    canonical_family_member,
    family_member_count,
    family_member_refs,
    physical_tables_for_object,
    stable_sorted_tables,
)
from sol01.schema.reference_context import render_table_reference
from sol01.schema.utils import _string_list

MAX_FAMILY_MEMBERS_IN_PROMPT = DEFAULT_SCHEMA_RENDER_POLICY.family_members_in_prompt
MAX_FAMILY_MEMBERS_TO_EXPAND = DEFAULT_SCHEMA_RENDER_POLICY.family_members_to_expand
MAX_VARIANT_COLUMNS_IN_PROMPT = DEFAULT_SCHEMA_RENDER_POLICY.variant_columns_in_prompt
MAX_SCHEMA_CONTEXT_EVIDENCE_LINES = DEFAULT_SCHEMA_RENDER_POLICY.schema_context_evidence_lines


def render_prompt_context(
    *,
    db: str,
    selected_objects: Sequence[SelectedSchemaObject],
    object_by_id: Mapping[str, SchemaObject],
    allowed_tables: Sequence[str],
    table_schemas: Mapping[str, TableSchema],
    schema_context_evidence: Sequence[SchemaContextObject],
    diagnostics: Mapping[str, object],
) -> str:
    """Render compact schema context without expanding every family member DDL."""

    lines = [
        "Resolved schema context:",
        f"Database: {db}",
        "Allowed physical tables:",
        *(bullet_lines(allowed_tables) if allowed_tables else ["- none resolved"]),
        "",
    ]
    rendered_tables: set[str] = set()

    for selected in selected_objects:
        schema_object = object_by_id.get(selected.object_id)
        if schema_object is None:
            continue
        if schema_object.object_type == "family":
            lines.extend(
                render_family(
                    schema_object,
                    table_schemas,
                    resolution=family_resolution_entry(selected.object_id, diagnostics),
                )
            )
            lines.append("")
            continue
        for table_name in physical_tables_for_object(schema_object):
            if table_name in rendered_tables or table_name not in table_schemas:
                continue
            rendered_tables.add(table_name)
            lines.extend(render_table(table_schemas[table_name]))
            lines.append("")

    family_tables = {
        table_name
        for item in selected_objects
        if item.object_id in object_by_id and object_by_id[item.object_id].object_type == "family"
        for table_name in family_member_refs(object_by_id[item.object_id])
    }
    for table_name in allowed_tables:
        if table_name in rendered_tables or table_name not in table_schemas:
            continue
        if table_name in family_tables:
            continue
        rendered_tables.add(table_name)
        lines.extend(render_table(table_schemas[table_name]))
        lines.append("")

    evidence_lines = schema_context_evidence_lines(schema_context_evidence)
    if evidence_lines:
        lines.extend(["Schema context filter and join evidence:", *evidence_lines, ""])

    warnings = diagnostics.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.extend(
            ["Resolution warnings:", *bullet_lines(str(warning) for warning in warnings), ""]
        )

    return "\n".join(lines).rstrip()


def render_family(
    schema_object: SchemaObject,
    table_schemas: Mapping[str, TableSchema],
    *,
    resolution: Mapping[str, object] | None = None,
) -> list[str]:
    """Render one logical family with one canonical structure and compact members."""

    canonical = canonical_family_member(schema_object)
    canonical_schema = table_schemas.get(canonical)
    members = resolved_family_members(schema_object, resolution)
    common_columns = _string_list(schema_object.metadata.get("common_columns"))
    variant_columns = variant_column_lines(schema_object.metadata.get("variant_columns"))
    member_count = family_member_count(schema_object)
    symbolic = bool(resolution and resolution.get("symbolic"))
    lines = [
        f"Table family: {schema_object.name}",
        f"Canonical structure: {canonical}",
        f"Family members: {member_count} total",
    ]
    if symbolic:
        budget = resolution.get("member_expansion_budget", MAX_FAMILY_MEMBERS_TO_EXPAND)
        matched = resolution.get("matched_member_count", member_count)
        lines.append(
            f"Physical members: kept symbolic ({matched} matched; expansion budget {budget})"
        )
    elif members:
        lines.append(f"Physical members: {compact_member_list(members)}")
    if common_columns:
        lines.append(f"Common columns: {', '.join(common_columns)}")
    if variant_columns:
        lines.append("Variant columns:")
        lines.extend(bullet_lines(variant_columns[:MAX_VARIANT_COLUMNS_IN_PROMPT]))
    suffix_summary = suffix_dimension_lines(schema_object.metadata.get("suffix_dimensions"))
    if suffix_summary:
        lines.append("Detected suffix dimensions:")
        lines.extend(bullet_lines(suffix_summary))
    if canonical_schema is not None:
        lines.extend(render_table(canonical_schema, header="Canonical table DDL and columns"))
    return lines


def family_resolution_entry(
    object_id: str,
    diagnostics: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Look up one family resolution diagnostics entry."""

    entries = diagnostics.get("resolution_entries")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict) and entry.get("object_id") == object_id:
            return entry
    return None


def resolved_family_members(
    schema_object: SchemaObject,
    resolution: Mapping[str, object] | None,
) -> list[str]:
    """Return resolved family members or the canonical fallback preview."""

    if resolution is not None:
        resolved = resolution.get("resolved_tables")
        if isinstance(resolved, list):
            return stable_sorted_tables(str(item).strip() for item in resolved)
    return family_member_refs(schema_object)


def render_table(table: TableSchema, *, header: str | None = None) -> list[str]:
    """Render one table schema for SQL generation."""

    return render_table_reference(table, header=header)


def schema_context_evidence_lines(
    schema_context_evidence: Sequence[SchemaContextObject],
) -> list[str]:
    """Return capped prompt evidence from selected schema context objects."""

    lines: list[str] = []
    for item in schema_context_evidence:
        if item.schema_object.object_type not in {
            "join_candidate",
            "sample_value",
            "column",
            "column_group",
        }:
            continue
        if not item.planning_text:
            continue
        lines.append(f"- {item.schema_object.object_type}: {' '.join(item.planning_text.split())}")
        if len(lines) >= MAX_SCHEMA_CONTEXT_EVIDENCE_LINES:
            return lines
    return lines


def compact_member_list(members: Sequence[str]) -> str:
    """Return a compact family member preview."""

    if len(members) <= MAX_FAMILY_MEMBERS_IN_PROMPT:
        return ", ".join(members)
    return ", ".join(
        [
            *members[:MAX_FAMILY_MEMBERS_IN_PROMPT],
            f"... {len(members) - MAX_FAMILY_MEMBERS_IN_PROMPT} more",
        ]
    )


def variant_column_lines(raw_variant_columns: object) -> list[str]:
    """Render compact variant-column descriptions."""

    if not isinstance(raw_variant_columns, dict):
        return []
    raw_columns = raw_variant_columns.get("columns")
    if not isinstance(raw_columns, list):
        return []
    lines = []
    for raw_column in raw_columns:
        if not isinstance(raw_column, dict):
            continue
        column_name = str(raw_column.get("column_name") or "").strip()
        member_refs = _string_list(raw_column.get("member_table_refs"))
        if column_name:
            lines.append(f"{column_name}: {', '.join(member_refs)}")
    if raw_variant_columns.get("truncated"):
        lines.append(f"... {raw_variant_columns.get('total_count')} total variant columns")
    return lines


def suffix_dimension_lines(raw_dimensions: object) -> list[str]:
    """Render compact suffix-dimension hints."""

    if not isinstance(raw_dimensions, list):
        return []
    lines = []
    for raw_dimension in raw_dimensions:
        if not isinstance(raw_dimension, dict):
            continue
        kind = str(raw_dimension.get("kind") or "").strip()
        values = _string_list(raw_dimension.get("values"))
        raw_values = _string_list(raw_dimension.get("raw_values"))
        if kind:
            lines.append(
                f"{kind}: values={compact_value_list(values)} raw={compact_value_list(raw_values)}"
            )
    return lines


def compact_value_list(values: Sequence[str]) -> str:
    """Render a compact list of suffix values."""

    if len(values) <= MAX_FAMILY_MEMBERS_IN_PROMPT:
        return ", ".join(values)
    shown = ", ".join(values[:MAX_FAMILY_MEMBERS_IN_PROMPT])
    return f"{shown}, ... {len(values) - MAX_FAMILY_MEMBERS_IN_PROMPT} more"


def bullet_lines(values: Iterable[str]) -> list[str]:
    """Prefix strings with markdown list markers."""

    return [f"- {value}" for value in values]
