"""Deterministic SQL normalization: quote Snowflake case-sensitive identifiers."""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.scope import Scope, build_scope

from sol01.models import TableSchema


def auto_quote_identifiers(sql: str, table_schemas: dict[str, TableSchema]) -> str:
    """Add double quotes to unquoted column refs that need them for Snowflake.

    A column needs quoting when its exact name contains lowercase characters
    (``name != name.upper()``).  In Snowflake, unquoted identifiers are
    uppercased automatically, so ``collection_id`` becomes ``COLLECTION_ID``
    which won't match a column actually named ``collection_id``.
    """

    per_table = _build_per_table_quoting(table_schemas)
    global_lookup = _build_global_quoting_lookup(table_schemas)
    if not per_table and not global_lookup:
        return sql

    try:
        statements = sqlglot.parse(sql, read="snowflake")
    except Exception:
        return sql

    if not statements or statements[0] is None:
        return sql

    statement = statements[0]
    changed = _quote_with_scope(statement, per_table, global_lookup)
    return statement.sql(dialect="snowflake") if changed else sql


def _quote_with_scope(
    statement: exp.Expression,
    per_table: dict[str, dict[str, str]],
    global_lookup: dict[str, str],
) -> bool:
    """Quote columns scope-by-scope using alias-resolved table lookups."""
    try:
        root_scope = build_scope(statement)
    except (OptimizeError, Exception):
        root_scope = None

    if root_scope is None:
        return _quote_global(statement, global_lookup)

    changed = False
    for scope in root_scope.traverse():
        alias_to_table = {
            alias.lower(): _table_id(source)
            for alias, (_, source) in scope.selected_sources.items()
            if isinstance(source, exp.Table)
        }
        source_tables = list(set(alias_to_table.values()))

        for column in scope.columns:
            identifier = column.args.get("this")
            if not isinstance(identifier, exp.Identifier) or identifier.args.get("quoted"):
                continue
            col_lower = column.name.lower()

            if column.table:
                canonical = alias_to_table.get(column.table.lower())
                if canonical:
                    exact = per_table.get(canonical, {}).get(col_lower)
                else:
                    # CTE or subquery alias: fall back to global lookup so that
                    # outer references to quoted CTE output columns get fixed too.
                    exact = global_lookup.get(col_lower)
            elif len(source_tables) == 1:
                exact = per_table.get(source_tables[0], {}).get(col_lower)
            else:
                matches = {per_table.get(t, {}).get(col_lower) for t in source_tables} - {None}
                exact = next(iter(matches)) if len(matches) == 1 else global_lookup.get(col_lower)

            if exact is not None:
                identifier.set("this", exact)
                identifier.set("quoted", True)
                changed = True

    return changed


def _quote_global(statement: exp.Expression, global_lookup: dict[str, str]) -> bool:
    """Fallback: quote columns using the global unambiguous lookup."""
    changed = False
    for column in statement.find_all(exp.Column):
        identifier = column.args.get("this")
        if not isinstance(identifier, exp.Identifier) or identifier.args.get("quoted"):
            continue
        exact = global_lookup.get(column.name.lower())
        if exact is not None:
            identifier.set("this", exact)
            identifier.set("quoted", True)
            changed = True
    return changed


def _table_id(table: exp.Table) -> str:
    parts = [p for p in (table.catalog, table.db, table.name) if p]
    return ".".join(parts).lower()


def _build_per_table_quoting(table_schemas: dict[str, TableSchema]) -> dict[str, dict[str, str]]:
    """Build {canonical_lower: {col_lower: exact_name}} for columns needing quotes."""
    result = {}
    for name, schema in table_schemas.items():
        cols = {
            col.name.lower(): col.name
            for col in schema.columns
            if col.name != col.name.upper()
        }
        if cols:
            result[name.lower()] = cols
    return result


def _build_global_quoting_lookup(table_schemas: dict[str, TableSchema]) -> dict[str, str]:
    """Build unambiguous {col_lower: exact_name} across all tables (fallback for unqualified cols)."""
    candidates: dict[str, str | None] = {}
    for schema in table_schemas.values():
        for col in schema.columns:
            if col.name == col.name.upper():
                continue
            key = col.name.lower()
            existing = candidates.get(key)
            if existing is None and key in candidates:
                continue
            if existing is not None and existing != col.name:
                candidates[key] = None
            else:
                candidates[key] = col.name
    return {key: name for key, name in candidates.items() if name is not None}
