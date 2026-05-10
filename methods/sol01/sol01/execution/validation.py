"""Validate generated SQL before it touches Snowflake."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError
from sqlglot.optimizer.scope import Scope, build_scope

from sol01.models import TableSchema, ValidationReport

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


def validate_sql(
    sql: str,
    *,
    allowed_tables: Iterable[str],
    table_schemas: Mapping[str, TableSchema] | Iterable[TableSchema] | None = None,
) -> ValidationReport:
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

    try:
        referenced_tables, table_errors, warnings = _resolve_referenced_tables(
            statement,
            allowed_tables=allowed_tables,
        )
    except OptimizeError as exc:
        return ValidationReport(ok=False, errors=[_scope_resolution_error(exc)])
    column_errors: list[str] = []
    schema_lookup = _schema_lookup(table_schemas)
    if schema_lookup and not table_errors:
        try:
            column_errors, column_warnings = _validate_columns(
                statement,
                allowed_tables=allowed_tables,
                schema_lookup=schema_lookup,
            )
            warnings.extend(column_warnings)
        except OptimizeError as exc:
            column_errors.append(_scope_resolution_error(exc))

    return ValidationReport(
        ok=not table_errors and not column_errors,
        errors=[*table_errors, *column_errors],
        warnings=warnings,
        referenced_tables=referenced_tables,
    )


@dataclass(frozen=True)
class _ColumnResolution:
    """Result of validating one identifier against one table schema."""

    matched: bool
    ok: bool
    error: str | None = None


def _parse_statements(sql: str) -> tuple[list[exp.Expression], str | None]:
    """Parse SQL with the Snowflake dialect and return a clean parse error if needed."""

    try:
        return sqlglot.parse(sql, read="snowflake"), None
    except ParseError as exc:
        return [], f"SQL could not be parsed: {exc}"


def _scope_resolution_error(error: OptimizeError) -> str:
    """Return a validation error when sqlglot cannot resolve query scopes."""

    detail = str(error).strip().rstrip(".").rstrip(":").strip() or error.__class__.__name__
    return f"SQL scope could not be resolved: {detail}."


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
        canonical_name = _canonical_table_name(
            table_name,
            allowed_lookup=allowed_lookup,
            suffix_lookup=suffix_lookup,
        )
        if canonical_name is None:
            if normalized_name not in seen_unknown_tables:
                seen_unknown_tables.add(normalized_name)
                errors.append(f"Unknown table referenced: {table_name}.")
            continue

        if canonical_name not in seen_tables:
            seen_tables.add(canonical_name)
            referenced_tables.append(canonical_name)

    return referenced_tables, errors, warnings


def _validate_columns(
    statement: exp.Expression,
    *,
    allowed_tables: Iterable[str],
    schema_lookup: dict[str, TableSchema],
) -> tuple[list[str], list[str]]:
    """Validate resolvable column references against selected table schemas."""

    allowed_lookup = {table.lower(): table for table in allowed_tables}
    suffix_lookup = _allowed_suffix_lookup(allowed_tables)
    errors: list[str] = []
    warnings: list[str] = []
    seen_errors: set[str] = set()
    seen_warnings: set[str] = set()
    root_scope = build_scope(statement)
    scopes = root_scope.traverse() if root_scope is not None else []

    for scope in scopes:
        source_tables = _scope_source_tables(
            scope,
            allowed_lookup=allowed_lookup,
            suffix_lookup=suffix_lookup,
        )
        for column in scope.columns:
            if column.name == "*":
                continue

            error, warning = _validate_scope_column(column, scope, source_tables, schema_lookup)
            if error and error not in seen_errors:
                seen_errors.add(error)
                errors.append(error)
            if warning and warning not in seen_warnings:
                seen_warnings.add(warning)
                warnings.append(warning)

    return errors, warnings


def _validate_scope_column(
    column: exp.Column,
    scope: Scope,
    source_tables: dict[str, str],
    schema_lookup: dict[str, TableSchema],
) -> tuple[str | None, str | None]:
    """Validate one column when its table source is clear enough."""

    if column.table:
        source = _scope_source(scope, column.table)
        if isinstance(source, Scope):
            return None, None
        table_name = source_tables.get(_normalize_identifier(column.table))
        if table_name is None:
            return None, f"Could not validate column {column.sql()}: unknown table alias."
        table_schema = schema_lookup.get(table_name.lower())
        if table_schema is None:
            return None, f"Could not validate column {column.sql()}: schema unavailable."
        resolution = _resolve_column(column, table_schema)
        return resolution.error, None

    table_schemas = [
        schema_lookup[table_name.lower()]
        for table_name in sorted(set(source_tables.values()))
        if table_name.lower() in schema_lookup
    ]
    if len(table_schemas) == 1:
        resolution = _resolve_column(column, table_schemas[0])
        return resolution.error, None
    if not table_schemas:
        return None, f"Could not validate column {column.sql()}: no table schema available."

    matching = [
        resolution
        for table_schema in table_schemas
        if (resolution := _resolve_column(column, table_schema)).matched
    ]
    if len(matching) == 1:
        return matching[0].error, None
    if len(matching) > 1:
        return (
            None,
            f"Could not validate unqualified column {column.sql()}: matches multiple tables.",
        )
    return None, f"Could not validate unqualified column {column.sql()}: no selected table has it."


def _resolve_column(column: exp.Column, table_schema: TableSchema) -> _ColumnResolution:
    """Return whether a Snowflake column identifier is valid for one table."""

    column_name = column.name
    exact_names = {schema_column.name for schema_column in table_schema.columns}
    casefold_names = _casefold_names(exact_names)
    identifier = column.args.get("this")
    quoted = isinstance(identifier, exp.Identifier) and bool(identifier.args.get("quoted"))

    if quoted:
        if column_name in exact_names:
            return _ColumnResolution(matched=True, ok=True)
        exact_match = casefold_names.get(column_name.lower())
        if exact_match is not None:
            return _ColumnResolution(
                matched=True,
                ok=False,
                error=(
                    f"Unknown quoted column {column.sql()} on {table_schema.full_name}; "
                    f'use "{exact_match}" instead.'
                ),
            )
        return _ColumnResolution(
            matched=False,
            ok=False,
            error=f"Unknown column {column.sql()} on {table_schema.full_name}.",
        )

    upper_name = column_name.upper()
    if upper_name in exact_names:
        return _ColumnResolution(matched=True, ok=True)

    exact_match = casefold_names.get(column_name.lower())
    if exact_match is not None:
        return _ColumnResolution(
            matched=True,
            ok=False,
            error=(
                f'Use "{exact_match}" instead of {column.sql()}; '
                f"Snowflake uppercases unquoted identifiers to {upper_name}."
            ),
        )

    return _ColumnResolution(
        matched=False,
        ok=False,
        error=f"Unknown column {column.sql()} on {table_schema.full_name}.",
    )


def _casefold_names(exact_names: set[str]) -> dict[str, str]:
    """Return unique case-insensitive column matches."""

    matches: dict[str, str | None] = {}
    for name in exact_names:
        normalized = name.lower()
        existing = matches.get(normalized)
        if existing is None and normalized in matches:
            continue
        if existing is not None and existing != name:
            matches[normalized] = None
        else:
            matches[normalized] = name
    return {normalized: name for normalized, name in matches.items() if name is not None}


def _scope_source_tables(
    scope: Scope,
    *,
    allowed_lookup: dict[str, str],
    suffix_lookup: dict[str, str],
) -> dict[str, str]:
    """Map table aliases/names in one scope to canonical selected table names."""

    sources: dict[str, str] = {}
    for source_name, (_, source) in scope.selected_sources.items():
        if not isinstance(source, exp.Table):
            continue
        table_name = _table_identifier(source)
        canonical = _canonical_table_name(
            table_name,
            allowed_lookup=allowed_lookup,
            suffix_lookup=suffix_lookup,
        )
        if canonical is None:
            continue
        sources[_normalize_identifier(source_name)] = canonical
        sources[_normalize_identifier(source.name)] = canonical
        alias = source.alias
        if alias:
            sources[_normalize_identifier(alias)] = canonical
    return sources


def _scope_source(scope: Scope, source_name: str) -> exp.Table | Scope | None:
    """Return the resolved table or derived scope for a column qualifier."""

    normalized = _normalize_identifier(source_name)
    for candidate_name, (_, source) in scope.selected_sources.items():
        if _normalize_identifier(candidate_name) == normalized:
            return source
    return None


def _canonical_table_name(
    table_name: str,
    *,
    allowed_lookup: dict[str, str],
    suffix_lookup: dict[str, str],
) -> str | None:
    """Return the canonical allowed table name for a raw table reference."""

    normalized_name = table_name.lower()
    return allowed_lookup.get(normalized_name) or suffix_lookup.get(normalized_name)


def _schema_lookup(
    table_schemas: Mapping[str, TableSchema] | Iterable[TableSchema] | None,
) -> dict[str, TableSchema]:
    """Normalize selected table schemas to a lower-case full-name lookup."""

    if table_schemas is None:
        return {}
    if isinstance(table_schemas, Mapping):
        schemas = table_schemas.values()
    else:
        schemas = table_schemas

    lookup: dict[str, TableSchema] = {}
    for table_schema in schemas:
        table_name = table_schema.full_name or table_schema.name
        lookup[table_name.lower()] = table_schema
    return lookup


def _normalize_identifier(identifier: str) -> str:
    """Normalize SQL aliases and identifiers for lookup."""

    return identifier.strip('"').lower()


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
