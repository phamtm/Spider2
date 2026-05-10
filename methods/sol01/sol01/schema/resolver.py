"""Resolve selected logical schema objects to concrete physical table context."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from typing import Any

from sol01.models import (
    HybridPlanningConstraints,
    ResolvedSchemaContext,
    RetrievedSchemaObject,
    SchemaObject,
    SelectedSchemaObject,
    TableSchema,
)

MAX_FAMILY_MEMBERS_IN_PROMPT = 16
MAX_VARIANT_COLUMNS_IN_PROMPT = 12
MAX_RETRIEVAL_EVIDENCE_LINES = 8

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
    retrieval_evidence: Sequence[RetrievedSchemaObject] = (),
    constraints: HybridPlanningConstraints | None = None,
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
            resolved, family_warnings, reason = _resolve_family(
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
    missing_schemas = [
        table_name for table_name in allowed_tables if table_name not in table_schemas
    ]
    if missing_schemas:
        warnings.append(f"Resolved table schemas are missing for: {', '.join(missing_schemas)}")

    diagnostics: dict[str, object] = {
        "selected_object_count": len(selected_objects),
        "retrieved_object_count": len(retrieval_evidence),
        "resolution_entries": resolution_entries,
        "warnings": warnings,
    }
    prompt_context = _render_prompt_context(
        db=db,
        selected_objects=selected_objects,
        object_by_id=object_by_id,
        allowed_tables=allowed_tables,
        table_schemas=table_schemas,
        retrieval_evidence=retrieval_evidence,
        diagnostics=diagnostics,
    )
    return ResolvedSchemaContext(
        db=db,
        selected_objects=list(selected_objects),
        retrieved_objects=list(retrieval_evidence),
        resolved_tables=allowed_tables,
        allowed_tables=allowed_tables,
        table_schemas=table_schemas,
        prompt_context=prompt_context,
        schema_prompt=prompt_context,
        resolution_diagnostics=diagnostics,
        diagnostics=diagnostics,
    )


def _merge_question_constraints(
    constraints: HybridPlanningConstraints | None,
    question: str,
) -> HybridPlanningConstraints:
    """Add clear dates, years, and version mentions from the user question."""

    base = constraints or HybridPlanningConstraints()
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
    constraints: HybridPlanningConstraints,
    question: str,
) -> tuple[list[str], list[str], str]:
    """Resolve one logical family object to physical members."""

    members = _family_members(schema_object)
    canonical = _canonical_member(schema_object)
    all_member_tables = [member["table_full_name"] for member in members]
    warnings: list[str] = []

    if constraints.include_all:
        return all_member_tables, warnings, "include_all"

    matched_members = [
        member for member in members if _member_matches_constraints(member, constraints)
    ]
    if matched_members:
        return (
            [member["table_full_name"] for member in matched_members],
            warnings,
            "explicit_constraints",
        )

    has_constraints = _has_explicit_constraints(constraints)
    if has_constraints:
        warnings.append(
            f"No family members matched constraints for {schema_object.object_id}; "
            f"using canonical member {canonical}."
        )
        return [canonical], warnings, "constraints_no_match"

    if _question_asks_broad_range(question):
        return all_member_tables, warnings, "broad_question"

    warnings.append(
        f"No family member constraint was provided for {schema_object.object_id}; "
        f"using canonical member {canonical}."
    )
    return [canonical], warnings, "ambiguous_family_default"


def _member_matches_constraints(
    member: dict[str, str],
    constraints: HybridPlanningConstraints,
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


def _render_prompt_context(
    *,
    db: str,
    selected_objects: Sequence[SelectedSchemaObject],
    object_by_id: Mapping[str, SchemaObject],
    allowed_tables: Sequence[str],
    table_schemas: Mapping[str, TableSchema],
    retrieval_evidence: Sequence[RetrievedSchemaObject],
    diagnostics: Mapping[str, object],
) -> str:
    """Render compact schema context without expanding every family member DDL."""

    lines = [
        "Resolved schema context:",
        f"Database: {db}",
        "Allowed physical tables:",
        *_bullet_lines(allowed_tables),
        "",
    ]
    rendered_tables: set[str] = set()

    for selected in selected_objects:
        schema_object = object_by_id.get(selected.object_id)
        if schema_object is None:
            continue
        if schema_object.object_type == "family":
            lines.extend(_render_family(schema_object, table_schemas))
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

    evidence_lines = _retrieval_evidence_lines(retrieval_evidence)
    if evidence_lines:
        lines.extend(["Retrieved filter and join evidence:", *evidence_lines, ""])

    warnings = diagnostics.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.extend(
            ["Resolution warnings:", *_bullet_lines(str(warning) for warning in warnings), ""]
        )

    return "\n".join(lines).rstrip()


def _render_family(
    schema_object: SchemaObject,
    table_schemas: Mapping[str, TableSchema],
) -> list[str]:
    """Render one logical family with one canonical structure and compact members."""

    canonical = _canonical_member(schema_object)
    canonical_schema = table_schemas.get(canonical)
    members = _family_member_refs(schema_object)
    common_columns = _string_list(schema_object.metadata.get("common_columns"))
    variant_columns = _variant_column_lines(schema_object.metadata.get("variant_columns"))
    lines = [
        f"Table family: {schema_object.name}",
        f"Canonical structure: {canonical}",
        f"Physical members: {_compact_member_list(members)}",
    ]
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


def _render_table(table: TableSchema, *, header: str | None = None) -> list[str]:
    """Render one table schema for SQL generation."""

    table_name = table.full_name or table.name
    lines = [f"{header or 'Table'}: {table_name}"]
    if table.ddl.strip():
        lines.extend(["DDL:", "```sql", table.ddl.strip(), "```"])
    elif table.columns:
        lines.append("Columns:")
        lines.extend(_bullet_lines(_column_line(column) for column in table.columns))
    if table.sample_rows:
        row_count = min(len(table.sample_rows), 3)
        lines.append(f"Sample rows available: {row_count} shown by upstream table context.")
    return lines


def _retrieval_evidence_lines(retrieval_evidence: Sequence[RetrievedSchemaObject]) -> list[str]:
    """Return capped prompt evidence from selected retrieval hits."""

    lines: list[str] = []
    for item in retrieval_evidence:
        for retrieved_chunk in item.chunks:
            chunk = retrieved_chunk.chunk
            if chunk.chunk_type not in {"join_candidate", "sample_value", "column", "column_group"}:
                continue
            text = (
                chunk.prompt_text or chunk.source_definition or chunk.inferred_usage or chunk.text
            )
            if not text:
                continue
            lines.append(f"- {chunk.chunk_type}: {' '.join(text.split())}")
            if len(lines) >= MAX_RETRIEVAL_EVIDENCE_LINES:
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


def _has_explicit_constraints(constraints: HybridPlanningConstraints) -> bool:
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
            lines.append(f"{kind}: values={', '.join(values)} raw={', '.join(raw_values)}")
    return lines


def _column_line(column: Any) -> str:
    line = str(column.name)
    if column.type:
        line += f" [{column.type}]"
    if column.description:
        line += f" - {column.description}"
    if column.sample_values:
        line += f" - sample values: {', '.join(column.sample_values[:3])}"
    return line


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
