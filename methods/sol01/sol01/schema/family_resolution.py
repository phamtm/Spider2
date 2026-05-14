"""Resolve logical table families to concrete physical tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from typing import Any

from sol01.models import SchemaObject, SchemaPlanningConstraints, SelectedSchemaObject, TableSchema
from sol01.schema.constants import MAX_FAMILY_MEMBERS_TO_EXPAND
from sol01.schema.family_constraints import (
    dimension_date,
    has_explicit_constraints,
    member_matches_years,
    normalize_constraint,
    normalize_version,
    parse_date,
    question_asks_broad_range,
)
from sol01.schema.utils import _metadata_text, _string_list


def resolve_family(
    schema_object: SchemaObject,
    *,
    constraints: SchemaPlanningConstraints,
    question: str,
) -> tuple[list[str], list[str], str, dict[str, object]]:
    """Resolve one logical family object to physical members."""

    members = family_members(schema_object)
    canonical = canonical_family_member(schema_object)
    all_member_tables = [member["table_full_name"] for member in members]
    warnings: list[str] = []

    if constraints.include_all:
        return budget_family_expansion(
            schema_object,
            all_member_tables,
            "include_all",
            "include_all requested every family member",
        )

    matched_members = [
        member for member in members if member_matches_constraints(member, constraints)
    ]
    if matched_members:
        return budget_family_expansion(
            schema_object,
            [member["table_full_name"] for member in matched_members],
            "explicit_constraints",
            "explicit constraints matched family members",
        )

    if has_explicit_constraints(constraints):
        warnings.append(
            f"No family members matched constraints for {schema_object.object_id}; "
            f"using canonical member {canonical}."
        )
        return (
            [canonical],
            warnings,
            "constraints_no_match",
            family_resolution_details(schema_object, [canonical], symbolic=False),
        )

    if question_asks_broad_range(question):
        return budget_family_expansion(
            schema_object,
            all_member_tables,
            "broad_question",
            "broad historical question matched every family member",
        )

    warnings.append(
        f"No family member constraint was provided for {schema_object.object_id}; "
        f"using canonical member {canonical}."
    )
    return (
        [canonical],
        warnings,
        "ambiguous_family_default",
        family_resolution_details(schema_object, [canonical], symbolic=False),
    )


def budget_family_expansion(
    schema_object: SchemaObject,
    resolved_tables: list[str],
    reason: str,
    explanation: str,
) -> tuple[list[str], list[str], str, dict[str, object]]:
    """Expand small family selections and keep oversized selections symbolic."""

    details = family_resolution_details(schema_object, resolved_tables, symbolic=False)
    if len(resolved_tables) <= MAX_FAMILY_MEMBERS_TO_EXPAND:
        return resolved_tables, [], reason, details

    warning = (
        f"Table family {schema_object.object_id} kept symbolic: {explanation}, "
        f"but {len(resolved_tables)} matched members exceed expansion budget "
        f"{MAX_FAMILY_MEMBERS_TO_EXPAND}. Add an explicit date, date range, year, "
        "suffix, or version constraint before using physical family members."
    )
    return (
        [],
        [warning],
        f"symbolic_{reason}",
        family_resolution_details(
            schema_object,
            [],
            symbolic=True,
            matched_member_count=len(resolved_tables),
        ),
    )


def family_resolution_details(
    schema_object: SchemaObject,
    resolved_tables: Sequence[str],
    *,
    symbolic: bool,
    matched_member_count: int | None = None,
) -> dict[str, object]:
    """Return compact diagnostics for one table-family resolution."""

    return {
        "family_member_count": family_member_count(schema_object),
        "matched_member_count": matched_member_count
        if matched_member_count is not None
        else len(resolved_tables),
        "member_expansion_budget": MAX_FAMILY_MEMBERS_TO_EXPAND,
        "symbolic": symbolic,
    }


def family_member_count(schema_object: SchemaObject) -> int:
    """Return the tracked or inferred number of family members."""

    raw_count = schema_object.metadata.get("member_count")
    if isinstance(raw_count, int) and raw_count >= 0:
        return raw_count
    return len(family_member_refs(schema_object))


def member_matches_constraints(
    member: dict[str, Any],
    constraints: SchemaPlanningConstraints,
) -> bool:
    """Return True when one family member matches the selected constraints."""

    dimension = member.get("suffix_dimension")
    if not isinstance(dimension, dict):
        return False

    raw_value = str(dimension.get("raw_value") or "")
    value = str(dimension.get("value") or "")
    kind = str(dimension.get("kind") or "")
    comparable = {
        normalize_constraint(raw_value),
        normalize_constraint(value),
        normalize_constraint(member.get("short_name", "")),
    }

    if constraints.suffixes and any(
        normalize_constraint(suffix) in comparable for suffix in constraints.suffixes
    ):
        return True

    if constraints.version is not None and kind == "version":
        wanted = normalize_version(constraints.version)
        return wanted in {normalize_version(raw_value), normalize_version(value)}

    date_start = parse_date(constraints.date_start) if constraints.date_start else None
    date_end = parse_date(constraints.date_end) if constraints.date_end else None
    member_date = dimension_date(value)
    if member_date is not None and (date_start is not None or date_end is not None):
        lower = date_start or date.min
        upper = date_end or date.max
        return lower <= member_date <= upper

    if constraints.years and member_matches_years(value, raw_value, member_date, constraints.years):
        return True

    return False


def physical_tables_for_object(schema_object: SchemaObject) -> list[str]:
    """Return physical table names directly referenced by a non-family object."""

    tables: list[str] = []
    if schema_object.table_name:
        tables.append(schema_object.table_name)
    if table_full_name := _metadata_text(schema_object, "table_full_name"):
        tables.append(table_full_name)
    for side_key in ("left", "right"):
        side = schema_object.metadata.get(side_key)
        if isinstance(side, dict) and side.get("table_full_name"):
            tables.append(str(side["table_full_name"]))
    return stable_sorted_tables(tables)


def canonical_family_schemas(
    *,
    selected_objects: Sequence[SelectedSchemaObject],
    object_by_id: Mapping[str, SchemaObject],
    table_lookup: Mapping[str, TableSchema],
    existing_tables: Mapping[str, TableSchema],
) -> dict[str, TableSchema]:
    """Keep one canonical family schema available for prompt rendering."""

    schemas: dict[str, TableSchema] = {}
    for selected in selected_objects:
        schema_object = object_by_id.get(selected.object_id)
        if schema_object is None or schema_object.object_type != "family":
            continue
        canonical = canonical_family_member(schema_object)
        if canonical in existing_tables:
            continue
        table = table_lookup.get(canonical.lower())
        if table is not None:
            schemas[canonical] = table
    return schemas


def family_members(schema_object: SchemaObject) -> list[dict[str, Any]]:
    """Return family members with any available suffix metadata."""

    members = []
    for raw_member in schema_object.metadata.get("members", []):
        if not isinstance(raw_member, dict):
            continue
        table_name = str(raw_member.get("table_full_name") or "").strip()
        if not table_name:
            continue
        members.append(raw_member)
    if members:
        return sorted(members, key=lambda member: str(member["table_full_name"]))
    return [{"table_full_name": table_name} for table_name in family_member_refs(schema_object)]


def family_member_refs(schema_object: SchemaObject) -> list[str]:
    """Return stable family member table references."""

    return stable_sorted_tables(_string_list(schema_object.metadata.get("member_table_refs")))


def canonical_family_member(schema_object: SchemaObject) -> str:
    """Return the canonical representative for one family."""

    canonical = _metadata_text(schema_object, "canonical_member")
    if canonical:
        return canonical
    members = family_member_refs(schema_object)
    return members[0] if members else (schema_object.table_name or schema_object.name)


def table_lookup(db_index: Mapping[str, TableSchema]) -> dict[str, TableSchema]:
    """Build a case-insensitive lookup for known table identities."""

    lookup: dict[str, TableSchema] = {}
    for key, table in db_index.items():
        names = {key, table.name, table.full_name or ""}
        for name in names:
            if name:
                lookup[name.lower()] = table
    return lookup


def stable_sorted_tables(values: Iterable[str]) -> list[str]:
    """Return unique, non-empty table names in deterministic order."""

    return sorted({value for value in values if value})
