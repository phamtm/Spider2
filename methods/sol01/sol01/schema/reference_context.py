"""Compatibility wrapper for exact SQL reference rendering."""

from __future__ import annotations

from collections.abc import Iterable

from sol01.models import SchemaProfileCatalog, TableSchema
from sol01.schema.exact_reference_context import (
    render_exact_sql_reference_context,
    render_exact_table_reference,
)


def render_table_reference(
    table: TableSchema,
    *,
    header: str | None = None,
    schema_profile_catalog: SchemaProfileCatalog | None = None,
) -> list[str]:
    """Render one exact table card with generated profile facts when available."""

    return render_exact_table_reference(
        table,
        header=header,
        schema_profile_catalog=schema_profile_catalog,
    )


def render_sql_reference_context(
    *,
    db: str,
    expanded_tables: Iterable[str],
    table_schemas: dict[str, TableSchema],
    schema_profile_catalog: SchemaProfileCatalog | None = None,
) -> str:
    """Render deterministic selected-table context for SQL prompts."""

    return render_exact_sql_reference_context(
        db=db,
        expanded_tables=expanded_tables,
        table_schemas=table_schemas,
        schema_profile_catalog=schema_profile_catalog,
    )
