"""Render exact selected-table metadata for SQL-stage prompts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from sol01.models import ColumnSchema, TableSchema
from sol01.schema.large_schema_summaries import (
    LargeSchemaSummary,
    LargeSchemaSummaryRegistry,
    load_large_schema_summary_registry,
)
from sol01.schema.summary_rendering import render_summary_lines

_GROUP_ORDER = ("TEXT", "NUMERIC", "BOOLEAN", "DATE/TIME", "OTHER")
_COLUMNS_PER_LINE = 8
_KEY_COLUMN_NOTE_LIMIT = 10
_SAMPLE_VALUES_PER_NOTE = 3


def render_exact_sql_reference_context(
    *,
    db: str,
    expanded_tables: Iterable[str],
    table_schemas: dict[str, TableSchema],
    large_schema_summary_registry: LargeSchemaSummaryRegistry | None = None,
) -> str:
    """Render exact selected-table context without curated summary fallback."""

    lines = [
        "SQL reference context:",
        f"Database: {db}",
        "Selected tables:",
    ]
    for table_name in sorted(expanded_tables):
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
    for table_name in sorted(table_schemas):
        lines.extend(
            render_exact_table_reference(
                table_schemas[table_name],
                large_schema_summary_registry=large_schema_summary_registry,
            )
        )
        lines.append("")
    lines.append("Use only the exact table and column names shown above.")
    return "\n".join(lines).rstrip()


def render_exact_table_reference(
    table: TableSchema,
    *,
    header: str | None = None,
    large_schema_summary_registry: LargeSchemaSummaryRegistry | None = None,
) -> list[str]:
    """Render one exact table card from indexed column metadata."""

    table_name = table.full_name or table.name
    lines = [f"{header or 'Table'}: {table_name}"]
    summary = _best_large_schema_summary_for_table(
        table,
        registry=large_schema_summary_registry,
    )
    if summary is not None:
        lines.extend(render_summary_lines(summary))
    if not table.columns:
        if table.ddl.strip():
            lines.extend(["DDL:", "```sql", table.ddl.strip(), "```"])
        else:
            lines.append("Columns: none available")
        return lines

    lines.append(f"Column count: {len(table.columns)}")
    lines.append("Exact columns by type:")
    lines.extend(_grouped_column_lines(table.columns))
    note_lines = _key_column_note_lines(table.columns)
    if note_lines:
        lines.append("Key column notes:")
        lines.extend(note_lines)
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


def _best_large_schema_summary_for_table(
    table: TableSchema,
    *,
    registry: LargeSchemaSummaryRegistry | None = None,
) -> LargeSchemaSummary | None:
    registry = registry or load_large_schema_summary_registry()
    database, schema_name, table_name = _table_identity_parts(table)
    matches: list[LargeSchemaSummary] = []
    if schema_name and table_name:
        matches = registry.match_table(
            database=database,
            schema_name=schema_name,
            table_name=table_name,
        )
    if not matches:
        table_ref = table.full_name or table.name
        if table_ref.count(".") in {1, 2}:
            matches = registry.match_table_ref(table_ref)
    if not matches:
        return None
    return sorted(matches, key=_summary_specificity_key)[0]


def _summary_specificity_key(summary: LargeSchemaSummary) -> tuple[int, int, int, str]:
    exact_table_names = len(summary.match.table_names)
    pattern_length = len(summary.match.table_pattern or "")
    schema_copy_count = len(summary.schema_copies)
    return (
        0 if exact_table_names else 1,
        schema_copy_count,
        -pattern_length,
        summary.summary_id,
    )


def _table_identity_parts(table: TableSchema) -> tuple[str, str, str]:
    database = table.database_name or ""
    schema_name = table.schema_name or ""
    table_name = table.name
    full_name = table.full_name or ""
    parts = [part for part in full_name.split(".") if part]
    if len(parts) == 3:
        database = database or parts[0]
        schema_name = schema_name or parts[1]
        table_name = parts[2]
    elif len(parts) == 2:
        schema_name = schema_name or parts[0]
        table_name = parts[1]
    return database, schema_name, table_name
