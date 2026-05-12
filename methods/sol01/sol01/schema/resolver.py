"""Resolve selected logical schema objects to concrete physical table context."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from typing import Any

from sol01.models import (
    ResolvedSchemaContext,
    SchemaContextObject,
    SchemaObject,
    SchemaPlanningConstraints,
    SelectedSchemaObject,
    TableSchema,
)
from sol01.schema.reference_context import render_table_reference

MAX_FAMILY_MEMBERS_IN_PROMPT = 16
MAX_FAMILY_MEMBERS_TO_EXPAND = 64
MAX_VARIANT_COLUMNS_IN_PROMPT = 12
MAX_SCHEMA_CONTEXT_EVIDENCE_LINES = 8

_BROAD_RANGE_RE = re.compile(
    r"\b(all|any|each|every|history|historical|daily|monthly|weekly|yearly|annual|"
    r"range|between|from|through|across|over time|time series)\b",
    re.I,
)
_DATE_RE = re.compile(r"\b(?P<year>(?:19|20)\d{2})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})\b")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_VERSION_RE = re.compile(r"\bv(?:ersion)?[_ -]?(?P<value>\d+)\b", re.I)


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

    constraints = _merge_question_constraints(constraints, question)
    object_by_id = {
        schema_object.object_id: schema_object for schema_object in canonical_schema_objects
    }
    table_lookup = _table_lookup(db_index)
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
            resolved, family_warnings, reason, details = _resolve_family(
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

        resolved = _physical_tables_for_object(schema_object)
        selected_table_names.extend(resolved)
        resolution_entries.append(
            {
                "object_id": selected.object_id,
                "object_type": schema_object.object_type,
                "resolved_tables": resolved,
                "reason": "exact_physical_table",
            }
        )

    allowed_tables = _stable_sorted_tables(selected_table_names)
    table_schemas = {
        table_name: table_lookup[table_name.lower()]
        for table_name in allowed_tables
        if table_name.lower() in table_lookup
    }
    table_schemas.update(
        _canonical_family_schemas(
            selected_objects=selected_objects,
            object_by_id=object_by_id,
            table_lookup=table_lookup,
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
    prompt_context = _render_prompt_context(
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
        allowed_tables=allowed_tables,
        table_schemas=table_schemas,
        prompt_context=prompt_context,
        diagnostics=diagnostics,
    )


def _merge_question_constraints(
    constraints: SchemaPlanningConstraints | None,
    question: str,
) -> SchemaPlanningConstraints:
    """Add clear dates, years, and version mentions from the user question."""

    base = constraints or SchemaPlanningConstraints()
    question_years = [int(match.group(0)) for match in _YEAR_RE.finditer(question)]
    years = _stable_unique_int([*base.years, *question_years])
    dates = [_parse_date(match.group(0)) for match in _DATE_RE.finditer(question)]
    dates = [item for item in dates if item is not None]
    date_start = base.date_start
    date_end = base.date_end
    if dates:
        date_start = date_start or min(dates).isoformat()
        date_end = date_end or max(dates).isoformat()
    elif len(question_years) >= 2:
        first_year = min(question_years)
        last_year = max(question_years)
        date_start = date_start or date(first_year, 1, 1).isoformat()
        date_end = date_end or date(last_year, 12, 31).isoformat()
    version = base.version
    if version is None and (match := _VERSION_RE.search(question)):
        version = match.group("value")
    return base.model_copy(
        update={"years": years, "date_start": date_start, "date_end": date_end, "version": version}
    )


def _resolve_family(
    schema_object: SchemaObject,
    *,
    constraints: SchemaPlanningConstraints,
    question: str,
) -> tuple[list[str], list[str], str, dict[str, object]]:
    """Resolve one logical family object to physical members."""

    members = _family_members(schema_object)
    canonical = _canonical_member(schema_object)
    all_member_tables = [member["table_full_name"] for member in members]
    warnings: list[str] = []

    if constraints.include_all:
        return _budget_family_expansion(
            schema_object,
            all_member_tables,
            "include_all",
            "include_all requested every family member",
        )

    matched_members = [
        member for member in members if _member_matches_constraints(member, constraints)
    ]
    if matched_members:
        return _budget_family_expansion(
            schema_object,
            [member["table_full_name"] for member in matched_members],
            "explicit_constraints",
            "explicit constraints matched family members",
        )

    has_constraints = _has_explicit_constraints(constraints)
    if has_constraints:
        warnings.append(
            f"No family members matched constraints for {schema_object.object_id}; "
            f"using canonical member {canonical}."
        )
        return (
            [canonical],
            warnings,
            "constraints_no_match",
            _family_resolution_details(schema_object, [canonical], symbolic=False),
        )

    if _question_asks_broad_range(question):
        return _budget_family_expansion(
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
        _family_resolution_details(schema_object, [canonical], symbolic=False),
    )


def _budget_family_expansion(
    schema_object: SchemaObject,
    resolved_tables: list[str],
    reason: str,
    explanation: str,
) -> tuple[list[str], list[str], str, dict[str, object]]:
    """Expand small family selections and keep oversized selections symbolic."""

    details = _family_resolution_details(schema_object, resolved_tables, symbolic=False)
    if len(resolved_tables) <= MAX_FAMILY_MEMBERS_TO_EXPAND:
        return resolved_tables, [], reason, details

    member_count = _family_member_count(schema_object)
    warning = (
        f"Table family {schema_object.object_id} kept symbolic: {explanation}, "
        f"but {len(resolved_tables)} matched members exceed expansion budget "
        f"{MAX_FAMILY_MEMBERS_TO_EXPAND}. Add an explicit date, date range, year, "
        "suffix, or version constraint before using physical family members."
    )
    symbolic_details = _family_resolution_details(
        schema_object,
        [],
        symbolic=True,
        matched_member_count=len(resolved_tables),
    )
    symbolic_details["member_count"] = member_count
    return [], [warning], f"symbolic_{reason}", symbolic_details


def _family_resolution_details(
    schema_object: SchemaObject,
    resolved_tables: Sequence[str],
    *,
    symbolic: bool,
    matched_member_count: int | None = None,
) -> dict[str, object]:
    """Return compact diagnostics for one table-family resolution."""

    member_count = _family_member_count(schema_object)
    return {
        "family_member_count": member_count,
        "matched_member_count": matched_member_count
        if matched_member_count is not None
        else len(resolved_tables),
        "member_expansion_budget": MAX_FAMILY_MEMBERS_TO_EXPAND,
        "symbolic": symbolic,
    }


def _family_member_count(schema_object: SchemaObject) -> int:
    raw_count = schema_object.metadata.get("member_count")
    if isinstance(raw_count, int) and raw_count >= 0:
        return raw_count
    return len(_family_member_refs(schema_object))


def _member_matches_constraints(
    member: dict[str, str],
    constraints: SchemaPlanningConstraints,
) -> bool:
    dimension = member.get("suffix_dimension")
    if not isinstance(dimension, dict):
        return False

    raw_value = str(dimension.get("raw_value") or "")
    value = str(dimension.get("value") or "")
    kind = str(dimension.get("kind") or "")
    comparable = {
        _normalize_constraint(raw_value),
        _normalize_constraint(value),
        _normalize_constraint(member.get("short_name", "")),
    }

    if constraints.suffixes and any(
        _normalize_constraint(suffix) in comparable for suffix in constraints.suffixes
    ):
        return True

    if constraints.version is not None and kind == "version":
        wanted = _normalize_version(constraints.version)
        return wanted in {_normalize_version(raw_value), _normalize_version(value)}

    date_start = _parse_date(constraints.date_start) if constraints.date_start else None
    date_end = _parse_date(constraints.date_end) if constraints.date_end else None
    member_date = _dimension_date(value)
    if member_date is not None and (date_start is not None or date_end is not None):
        lower = date_start or date.min
        upper = date_end or date.max
        return lower <= member_date <= upper

    if constraints.years and _member_matches_years(
        value, raw_value, member_date, constraints.years
    ):
        return True

    return False


def _physical_tables_for_object(schema_object: SchemaObject) -> list[str]:
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
    return _stable_sorted_tables(tables)


def _canonical_family_schemas(
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
        canonical = _canonical_member(schema_object)
        if canonical in existing_tables:
            continue
        table = table_lookup.get(canonical.lower())
        if table is not None:
            schemas[canonical] = table
    return schemas


def _render_prompt_context(
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
        *(_bullet_lines(allowed_tables) if allowed_tables else ["- none resolved"]),
        "",
    ]
    rendered_tables: set[str] = set()

    for selected in selected_objects:
        schema_object = object_by_id.get(selected.object_id)
        if schema_object is None:
            continue
        if schema_object.object_type == "family":
            lines.extend(
                _render_family(
                    schema_object,
                    table_schemas,
                    resolution=_family_resolution(selected.object_id, diagnostics),
                )
            )
            lines.append("")
            continue
        for table_name in _physical_tables_for_object(schema_object):
            if table_name in rendered_tables or table_name not in table_schemas:
                continue
            rendered_tables.add(table_name)
            lines.extend(_render_table(table_schemas[table_name]))
            lines.append("")

    for table_name in allowed_tables:
        if table_name in rendered_tables or table_name not in table_schemas:
            continue
        if any(
            table_name in _family_member_refs(object_by_id[item.object_id])
            for item in selected_objects
            if item.object_id in object_by_id
            and object_by_id[item.object_id].object_type == "family"
        ):
            continue
        rendered_tables.add(table_name)
        lines.extend(_render_table(table_schemas[table_name]))
        lines.append("")

    evidence_lines = _schema_context_evidence_lines(schema_context_evidence)
    if evidence_lines:
        lines.extend(["Schema context filter and join evidence:", *evidence_lines, ""])

    warnings = diagnostics.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.extend(
            ["Resolution warnings:", *_bullet_lines(str(warning) for warning in warnings), ""]
        )

    return "\n".join(lines).rstrip()


def _render_family(
    schema_object: SchemaObject,
    table_schemas: Mapping[str, TableSchema],
    *,
    resolution: Mapping[str, object] | None = None,
) -> list[str]:
    """Render one logical family with one canonical structure and compact members."""

    canonical = _canonical_member(schema_object)
    canonical_schema = table_schemas.get(canonical)
    members = _resolved_family_members(schema_object, resolution)
    common_columns = _string_list(schema_object.metadata.get("common_columns"))
    variant_columns = _variant_column_lines(schema_object.metadata.get("variant_columns"))
    member_count = _family_member_count(schema_object)
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
        lines.append(f"Physical members: {_compact_member_list(members)}")
    if common_columns:
        lines.append(f"Common columns: {', '.join(common_columns)}")
    if variant_columns:
        lines.append("Variant columns:")
        lines.extend(_bullet_lines(variant_columns[:MAX_VARIANT_COLUMNS_IN_PROMPT]))
    suffix_summary = _suffix_dimension_lines(schema_object.metadata.get("suffix_dimensions"))
    if suffix_summary:
        lines.append("Detected suffix dimensions:")
        lines.extend(_bullet_lines(suffix_summary))
    if canonical_schema is not None:
        lines.extend(_render_table(canonical_schema, header="Canonical table DDL and columns"))
    return lines


def _family_resolution(
    object_id: str,
    diagnostics: Mapping[str, object],
) -> Mapping[str, object] | None:
    entries = diagnostics.get("resolution_entries")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict) and entry.get("object_id") == object_id:
            return entry
    return None


def _resolved_family_members(
    schema_object: SchemaObject,
    resolution: Mapping[str, object] | None,
) -> list[str]:
    if resolution is not None:
        resolved = resolution.get("resolved_tables")
        if isinstance(resolved, list):
            return _stable_sorted_tables(str(item).strip() for item in resolved)
    return _family_member_refs(schema_object)


def _render_table(table: TableSchema, *, header: str | None = None) -> list[str]:
    """Render one table schema for SQL generation."""

    return render_table_reference(table, header=header)


def _schema_context_evidence_lines(
    schema_context_evidence: Sequence[SchemaContextObject],
) -> list[str]:
    """Return capped prompt evidence from selected schema context objects."""

    lines: list[str] = []
    for item in schema_context_evidence:
        for context_chunk in item.chunks:
            chunk = context_chunk.chunk
            if chunk.chunk_type not in {"join_candidate", "sample_value", "column", "column_group"}:
                continue
            text = (
                chunk.prompt_text or chunk.source_definition or chunk.inferred_usage or chunk.text
            )
            if not text:
                continue
            lines.append(f"- {chunk.chunk_type}: {' '.join(text.split())}")
            if len(lines) >= MAX_SCHEMA_CONTEXT_EVIDENCE_LINES:
                return lines
    return lines


def _family_members(schema_object: SchemaObject) -> list[dict[str, Any]]:
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
    return [{"table_full_name": table_name} for table_name in _family_member_refs(schema_object)]


def _family_member_refs(schema_object: SchemaObject) -> list[str]:
    return _stable_sorted_tables(_string_list(schema_object.metadata.get("member_table_refs")))


def _canonical_member(schema_object: SchemaObject) -> str:
    canonical = _metadata_text(schema_object, "canonical_member")
    if canonical:
        return canonical
    members = _family_member_refs(schema_object)
    return members[0] if members else (schema_object.table_name or schema_object.name)


def _table_lookup(db_index: Mapping[str, TableSchema]) -> dict[str, TableSchema]:
    lookup: dict[str, TableSchema] = {}
    for key, table in db_index.items():
        names = {key, table.name, table.full_name or ""}
        for name in names:
            if name:
                lookup[name.lower()] = table
    return lookup


def _has_explicit_constraints(constraints: SchemaPlanningConstraints) -> bool:
    return bool(
        constraints.date_start
        or constraints.date_end
        or constraints.years
        or constraints.suffixes
        or constraints.version is not None
    )


def _question_asks_broad_range(question: str) -> bool:
    if _BROAD_RANGE_RE.search(question):
        return True
    return len(_DATE_RE.findall(question)) >= 2 or len(_YEAR_RE.findall(question)) >= 2


def _member_matches_years(
    value: str,
    raw_value: str,
    member_date: date | None,
    years: Sequence[int],
) -> bool:
    year_set = set(years)
    if member_date is not None and member_date.year in year_set:
        return True
    return any(str(year) in {value, raw_value} for year in year_set)


def _dimension_date(value: str) -> date | None:
    if re.fullmatch(r"(?:19|20)\d{2}", value):
        return date(int(value), 1, 1)
    return _parse_date(value)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    match = _DATE_RE.search(str(value))
    if not match:
        return None
    try:
        return date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
    except ValueError:
        return None


def _compact_member_list(members: Sequence[str]) -> str:
    if len(members) <= MAX_FAMILY_MEMBERS_IN_PROMPT:
        return ", ".join(members)
    return ", ".join(
        [
            *members[:MAX_FAMILY_MEMBERS_IN_PROMPT],
            f"... {len(members) - MAX_FAMILY_MEMBERS_IN_PROMPT} more",
        ]
    )


def _variant_column_lines(raw_variant_columns: object) -> list[str]:
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


def _suffix_dimension_lines(raw_dimensions: object) -> list[str]:
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
                f"{kind}: values={_compact_value_list(values)} "
                f"raw={_compact_value_list(raw_values)}"
            )
    return lines


def _compact_value_list(values: Sequence[str]) -> str:
    if len(values) <= MAX_FAMILY_MEMBERS_IN_PROMPT:
        return ", ".join(values)
    shown = ", ".join(values[:MAX_FAMILY_MEMBERS_IN_PROMPT])
    return f"{shown}, ... {len(values) - MAX_FAMILY_MEMBERS_IN_PROMPT} more"


def _bullet_lines(values: Iterable[str]) -> list[str]:
    return [f"- {value}" for value in values]


def _stable_sorted_tables(values: Iterable[str]) -> list[str]:
    return sorted({value for value in values if value})


def _stable_unique_int(values: Iterable[int]) -> list[int]:
    return sorted({value for value in values})


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _metadata_text(schema_object: SchemaObject, key: str) -> str:
    value = schema_object.metadata.get(key)
    return "" if value is None else str(value).strip()


def _normalize_constraint(value: object) -> str:
    return str(value).strip().strip("_-").lower()


def _normalize_version(value: object) -> str:
    return re.sub(r"[^0-9]", "", str(value))
