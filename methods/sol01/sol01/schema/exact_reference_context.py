"""Render exact selected-table metadata for SQL-stage prompts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from sol01.models import (
    ColumnSchema,
    FamilyProfile,
    SchemaProfileCatalog,
    TableProfile,
    TableSchema,
)
from sol01.schema.schema_profile_rendering import render_schema_profile_lines
from sol01.schema.schema_profiles import load_schema_profile_catalog

_GROUP_ORDER = ("TEXT", "NUMERIC", "BOOLEAN", "DATE/TIME", "OTHER")
_COLUMNS_PER_LINE = 8
_KEY_COLUMN_NOTE_LIMIT = 10
_SAMPLE_VALUES_PER_NOTE = 3
_FULL_DETAIL_TABLE_LIMIT = 4
_COMPACT_COLUMN_PREVIEW_LIMIT = 12


def render_exact_sql_reference_context(
    *,
    db: str,
    expanded_tables: Iterable[str],
    table_schemas: dict[str, TableSchema],
    schema_profile_catalog: SchemaProfileCatalog | None = None,
) -> str:
    """Render exact selected-table context with profile-backed compacting."""

    catalog = schema_profile_catalog or load_schema_profile_catalog(db)
    lines = [
        "SQL reference context:",
        f"Database: {db}",
        "Selected tables:",
    ]
    sorted_expanded_tables = sorted(expanded_tables)
    for table_name in sorted_expanded_tables:
        lines.append(f"- {table_name}")

    if not table_schemas:
        lines.extend(
            [
                "",
                "Selected table details:",
                "- none available",
                "",
                "Use only the exact selected-table metadata shown above.",
            ]
        )
        return "\n".join(lines)

    lines.extend(["", "Selected table details:"])
    full_detail_tables = set(sorted_expanded_tables[:_FULL_DETAIL_TABLE_LIMIT])
    for table_name in sorted(table_schemas):
        detail_level = "full" if table_name in full_detail_tables else "compact"
        lines.extend(
            render_exact_table_reference(
                table_schemas[table_name],
                detail_level=detail_level,
                schema_profile_catalog=catalog,
            )
        )
        lines.append("")
    lines.append("Use only the exact table and column names shown above.")
    return "\n".join(lines).rstrip()


def render_exact_table_reference(
    table: TableSchema,
    *,
    header: str | None = None,
    detail_level: str = "full",
    schema_profile_catalog: SchemaProfileCatalog | None = None,
) -> list[str]:
    """Render one exact table card from indexed column metadata."""

    table_name = table.full_name or table.name
    lines = [f"{header or 'Table'}: {table_name}"]
    profiles = _profiles_for_table(
        table,
        catalog=schema_profile_catalog,
    )
    for profile in profiles[:2]:
        lines.extend(render_schema_profile_lines(profile))
    if not table.columns:
        if table.ddl.strip():
            lines.extend(["DDL:", "```sql", table.ddl.strip(), "```"])
        else:
            lines.append("Columns: none available")
        return lines

    lines.append(f"Column count: {len(table.columns)}")
    if detail_level == "compact":
        lines.extend(_compact_exact_column_lines(table, profiles=profiles))
        return lines

    lines.append("Exact columns by type:")
    lines.extend(_grouped_column_lines(table.columns))
    note_lines = _key_column_note_lines(table.columns)
    if note_lines:
        lines.append("Key column notes:")
        lines.extend(note_lines)
    return lines


def _compact_exact_column_lines(
    table: TableSchema,
    *,
    profiles: Sequence[TableProfile | FamilyProfile],
) -> list[str]:
    labels_by_name = {column.name: _column_label(column) for column in table.columns}
    lines: list[str] = []
    if profiles:
        profile = profiles[0]
        for title, names in (
            ("Exact key columns", profile.key_columns),
            ("Exact time columns", profile.time_columns),
            ("Exact measure columns", profile.measure_columns),
            ("Exact dimension columns", profile.dimension_columns),
        ):
            labels = [labels_by_name[name] for name in names if name in labels_by_name]
            if labels:
                lines.append(f"{title}: {', '.join(labels[:_COMPACT_COLUMN_PREVIEW_LIMIT])}")
    preview = [_column_label(column) for column in table.columns[:_COMPACT_COLUMN_PREVIEW_LIMIT]]
    if preview:
        suffix = (
            f", ... {len(table.columns) - _COMPACT_COLUMN_PREVIEW_LIMIT} more"
            if len(table.columns) > _COMPACT_COLUMN_PREVIEW_LIMIT
            else ""
        )
        lines.append(f"Exact column preview: {', '.join(preview)}{suffix}")
    return lines


def _grouped_column_lines(columns: Sequence[ColumnSchema]) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for column in columns:
        grouped[_column_group(column.type)].append(_column_label(column))

    lines: list[str] = []
    for group_name in _GROUP_ORDER:
        labels = grouped.get(group_name)
        if not labels:
            continue
        for index, chunk in enumerate(_chunked(labels, _COLUMNS_PER_LINE)):
            prefix = group_name if index == 0 else f"{group_name} (cont.)"
            lines.append(f"- {prefix}: {', '.join(chunk)}")
    return lines


def _column_group(column_type: str | None) -> str:
    normalized = (column_type or "").upper()
    if any(token in normalized for token in ("CHAR", "TEXT", "STRING", "VARCHAR")):
        return "TEXT"
    if any(
        token in normalized
        for token in ("NUMBER", "NUMERIC", "DECIMAL", "INT", "FLOAT", "DOUBLE", "REAL")
    ):
        return "NUMERIC"
    if "BOOL" in normalized:
        return "BOOLEAN"
    if any(token in normalized for token in ("DATE", "TIME", "TIMESTAMP")):
        return "DATE/TIME"
    return "OTHER"


def _column_label(column: ColumnSchema) -> str:
    if column.type:
        return f"{column.name} [{column.type}]"
    return column.name


def _chunked(values: Sequence[str], size: int) -> list[list[str]]:
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def _key_column_note_lines(columns: Sequence[ColumnSchema]) -> list[str]:
    ranked: list[tuple[int, int, ColumnSchema]] = []
    for index, column in enumerate(columns):
        score = _column_note_score(column)
        if score <= 0:
            continue
        ranked.append((score, -index, column))
    ranked.sort(reverse=True)

    notes: list[str] = []
    for _, _, column in ranked[:_KEY_COLUMN_NOTE_LIMIT]:
        fragments = [f"- {column.name}"]
        if column.type:
            fragments[0] += f" [{column.type}]"
        detail_bits: list[str] = []
        if column.description:
            detail_bits.append(column.description.strip())
        if column.sample_values:
            preview = ", ".join(column.sample_values[:_SAMPLE_VALUES_PER_NOTE])
            detail_bits.append(f"sample values: {preview}")
        if detail_bits:
            fragments.append(" - ".join(detail_bits))
        notes.append(" - ".join(fragments))
    return notes


def _column_note_score(column: ColumnSchema) -> int:
    name = column.name.lower()
    score = 0
    if column.description:
        score += 100
    if _looks_semantically_important(name):
        score += 50
    if column.sample_values and _looks_semantically_important(name):
        score += 20
    return score


def _looks_semantically_important(name: str) -> bool:
    return any(
        token in name
        for token in (
            "id",
            "code",
            "name",
            "state",
            "county",
            "region",
            "geo",
            "place",
            "pop",
            "total",
            "date",
            "time",
            "lat",
            "lon",
        )
    )


def _profiles_for_table(
    table: TableSchema,
    *,
    catalog: SchemaProfileCatalog | None = None,
) -> list[TableProfile | FamilyProfile]:
    if catalog is None:
        return []
    table_name = table.full_name or table.name
    matches: list[TableProfile | FamilyProfile] = []
    matches.extend(
        profile for profile in catalog.table_profiles if table_name in profile.covered_tables
    )
    matches.extend(
        profile for profile in catalog.family_profiles if table_name in profile.covered_tables
    )
    return sorted(
        {profile.profile_id: profile for profile in matches}.values(),
        key=lambda profile: (
            0 if profile.abstraction_kind in {"exact_family", "wide_table"} else 1,
            profile.profile_id,
        ),
    )
