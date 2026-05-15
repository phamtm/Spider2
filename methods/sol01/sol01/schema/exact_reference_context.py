"""Render exact selected-table metadata for SQL-stage prompts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from sol01.models import ColumnSchema, TableSchema

_GROUP_ORDER = ("TEXT", "NUMERIC", "BOOLEAN", "DATE/TIME", "OTHER")
_COLUMNS_PER_LINE = 8


def render_exact_sql_reference_context(
    *,
    db: str,
    expanded_tables: Iterable[str],
    table_schemas: dict[str, TableSchema],
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
        lines.extend(render_exact_table_reference(table_schemas[table_name]))
        lines.append("")
    lines.append("Use only the exact table and column names shown above.")
    return "\n".join(lines).rstrip()


def render_exact_table_reference(
    table: TableSchema,
    *,
    header: str | None = None,
) -> list[str]:
    """Render one exact table card from indexed column metadata."""

    table_name = table.full_name or table.name
    lines = [f"{header or 'Table'}: {table_name}"]
    if not table.columns:
        if table.ddl.strip():
            lines.extend(["DDL:", "```sql", table.ddl.strip(), "```"])
        else:
            lines.append("Columns: none available")
        return lines

    lines.append(f"Column count: {len(table.columns)}")
    lines.append("Exact columns by type:")
    lines.extend(_grouped_column_lines(table.columns))
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
