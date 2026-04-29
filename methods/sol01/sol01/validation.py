"""Validate generated SQL before it touches Snowflake."""

from __future__ import annotations

from collections.abc import Iterable

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.optimizer.scope import Scope, build_scope

from sol01.models import ValidationReport

DISALLOWED_STATEMENTS = (
    exp.Attach,
    exp.Command,
    exp.Create,
    exp.Delete,
    exp.Describe,
    exp.Detach,
    exp.Drop,
    exp.Insert,
    exp.Merge,
    exp.Pragma,
    exp.TruncateTable,
    exp.Update,
    exp.Use,
)


def validate_sql(sql: str, *, allowed_tables: Iterable[str]) -> ValidationReport:
    """Check that one SQL statement is read-only and scoped to known tables."""

    statements, parse_error = _parse_statements(sql)
    if parse_error is not None:
        return ValidationReport(ok=False, errors=[parse_error])

    if len(statements) != 1:
        return ValidationReport(
            ok=False,
            errors=["SQL must contain exactly one statement."],
        )

    statement = statements[0]
    statement_error = _statement_error(statement)
    if statement_error is not None:
        return ValidationReport(ok=False, errors=[statement_error])

    extension_error = _extension_loading_error(statement)
    if extension_error is not None:
        return ValidationReport(ok=False, errors=[extension_error])

    referenced_tables, table_errors, warnings = _resolve_referenced_tables(
        statement,
        allowed_tables=allowed_tables,
    )
    return ValidationReport(
        ok=not table_errors,
        errors=table_errors,
        warnings=warnings,
        referenced_tables=referenced_tables,
    )


def _parse_statements(sql: str) -> tuple[list[exp.Expression], str | None]:
    """Parse SQL with the Snowflake dialect and return a clean parse error if needed."""

    try:
        return sqlglot.parse(sql, read="snowflake"), None
    except ParseError as exc:
        return [], f"SQL could not be parsed: {exc}"


def _statement_error(statement: exp.Expression) -> str | None:
    """Reject non-query roots and explicitly blocked Snowflake commands."""

    if isinstance(statement, DISALLOWED_STATEMENTS):
        return f"Disallowed statement type: {statement.key.upper()}."

    if not isinstance(statement, exp.Query):
        return "Only read-only query statements are allowed."

    for node in statement.walk():
        if node is statement:
            continue
        if isinstance(node, DISALLOWED_STATEMENTS):
            return f"Disallowed statement type: {node.key.upper()}."

    return None


def _extension_loading_error(statement: exp.Expression) -> str | None:
    """Reject extension loading even when it appears inside a SELECT."""

    for function in statement.find_all(exp.Func):
        if getattr(function, "name", "").lower() == "load_extension":
            return "Extension loading is not allowed."
    return None


def _resolve_referenced_tables(
    statement: exp.Expression,
    *,
    allowed_tables: Iterable[str],
) -> tuple[list[str], list[str], list[str]]:
    """Collect base table names from resolved scopes instead of raw name matching."""

    allowed_lookup = {table.lower(): table for table in allowed_tables}
    suffix_lookup = _allowed_suffix_lookup(allowed_tables)
    referenced_tables: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    seen_tables: set[str] = set()
    seen_unknown_tables: set[str] = set()

    for table in _resolved_base_tables(statement):
        table_name = _table_identifier(table)
        normalized_name = table_name.lower()
        canonical_name = allowed_lookup.get(normalized_name) or suffix_lookup.get(normalized_name)
        if canonical_name is None:
            if normalized_name not in seen_unknown_tables:
                seen_unknown_tables.add(normalized_name)
                errors.append(f"Unknown table referenced: {table_name}.")
            continue

        if canonical_name not in seen_tables:
            seen_tables.add(canonical_name)
            referenced_tables.append(canonical_name)

    return referenced_tables, errors, warnings


def _table_identifier(table: exp.Table) -> str:
    """Return a Snowflake table reference from catalog, schema, and table parts."""

    parts = [part for part in (table.catalog, table.db, table.name) if part]
    return ".".join(parts)


def _allowed_suffix_lookup(allowed_tables: Iterable[str]) -> dict[str, str]:
    """Allow unique short or schema-qualified references to selected tables."""

    candidates: dict[str, str | None] = {}
    for table in allowed_tables:
        parts = [part for part in table.split(".") if part]
        suffixes = [parts[-1:]]
        if len(parts) >= 2:
            suffixes.append(parts[-2:])

        for suffix_parts in suffixes:
            suffix = ".".join(suffix_parts).lower()
            existing = candidates.get(suffix)
            if existing is None and suffix in candidates:
                continue
            if existing is not None and existing != table:
                candidates[suffix] = None
            else:
                candidates[suffix] = table

    return {suffix: table for suffix, table in candidates.items() if table is not None}


def _resolved_base_tables(statement: exp.Expression) -> list[exp.Table]:
    """Return only real base-table references after sqlglot resolves CTE scopes."""

    root_scope = build_scope(statement)
    if root_scope is None:
        return list(statement.find_all(exp.Table))

    tables: list[exp.Table] = []
    for scope in root_scope.traverse():
        for _, source in scope.selected_sources.values():
            if isinstance(source, exp.Table):
                tables.append(source)
            elif isinstance(source, Scope):
                continue
    return tables
