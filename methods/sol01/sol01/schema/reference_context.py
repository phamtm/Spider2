"""Render selected table references for SQL prompts."""

from __future__ import annotations

from collections.abc import Iterable

from sol01.models import ColumnSchema, TableSchema
from sol01.schema.large_schema_summaries import (
    LargeSchemaSummary,
    load_large_schema_summary_registry,
)


def render_table_reference(table: TableSchema, *, header: str | None = None) -> list[str]:
    """Render one table with curated summaries when a large-schema rule covers it."""

    summary = _large_schema_summary_for_table(table)
    if summary is not None:
        return _render_large_schema_summary(table, summary, header=header)
    return _render_full_table_reference(table, header=header)


def render_sql_reference_context(
    *,
    db: str,
    expanded_tables: Iterable[str],
    table_schemas: dict[str, TableSchema],
) -> str:
    """Render deterministic selected-table context for cache-friendly SQL prompts."""

    lines = [
        "SQL reference context:",
        f"Database: {db}",
        "Selected tables:",
    ]
    for table_name in sorted(expanded_tables):
        lines.append(f"- {table_name}")

    if not table_schemas:
        return "\n".join(lines)

    lines.append("")
    lines.append("Selected table details:")
    for table_name in sorted(table_schemas):
        lines.extend(render_table_reference(table_schemas[table_name]))
        lines.append("")
    return "\n".join(lines).rstrip()


def _render_large_schema_summary(
    table: TableSchema,
    summary: LargeSchemaSummary,
    *,
    header: str | None = None,
) -> list[str]:
    table_name = table.full_name or table.name
    lines = [
        f"{header or 'Table'}: {table_name}",
        f"Large-schema summary: {summary.summary_id}",
        f"Purpose: {summary.purpose}",
        f"Grain: {summary.grain}",
        "Use only exact names from these references or names confirmed by validation.",
    ]
    lines.extend(_section("Stable exact columns", summary.stable_columns))
    lines.extend(_section("Repeated or partition column rules", summary.repeated_column_rules))
    lines.extend(_section("Inclusive ranges", summary.inclusive_ranges))
    lines.extend(_section("Quote and spelling rules", summary.quote_spelling_rules))
    lines.extend(_section("Exact safe examples", summary.examples))
    return lines


def _render_full_table_reference(table: TableSchema, *, header: str | None = None) -> list[str]:
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


def _large_schema_summary_for_table(table: TableSchema) -> LargeSchemaSummary | None:
    registry = load_large_schema_summary_registry()
    database, schema_name, table_name = _table_identity_parts(table)
    if schema_name and table_name:
        matches = registry.match_table(
            database=database,
            schema_name=schema_name,
            table_name=table_name,
        )
        if matches:
            return matches[0]

    table_ref = table.full_name or table.name
    if table_ref.count(".") in {1, 2}:
        matches = registry.match_table_ref(table_ref)
        if matches:
            return matches[0]
    return None


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


def _section(title: str, values: list[str]) -> list[str]:
    if not values:
        return []
    return [f"{title}:", *_bullet_lines(values)]


def _bullet_lines(values: Iterable[str]) -> list[str]:
    return [f"- {value}" for value in values]


def _column_line(column: ColumnSchema) -> str:
    line = column.name
    if column.type:
        line += f" [{column.type}]"
    if column.description:
        line += f" - {column.description}"
    if column.sample_values:
        preview = ", ".join(column.sample_values[:3])
        line += f" - sample values: {preview}"
    return line
