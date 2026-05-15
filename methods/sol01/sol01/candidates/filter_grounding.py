"""Probe empty-result string filters for grounded label rewrites."""

from __future__ import annotations

from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from sol01.candidates.query_heuristics import coerce_number, looks_aggregate_query
from sol01.execution.snowflake_runner import fetch_query_dataframe as _fetch_query_dataframe
from sol01.infra.policy import DEFAULT_FILTER_GROUNDING_POLICY
from sol01.infra.strings import column_looks_string_like
from sol01.models import (
    ExecutionResult,
    FilterGroundingReport,
    SchemaSelection,
    SQLCandidate,
    TableSchema,
    Task,
    ValidationReport,
)


def infer_filter_grounding_report(
    *,
    task: Task,
    candidate: SQLCandidate,
    schema: SchemaSelection,
    table_schemas: dict[str, TableSchema],
    validation: ValidationReport,
    execution: ExecutionResult,
) -> FilterGroundingReport | None:
    """Probe exact string filters when the candidate returns no rows."""

    if not _execution_is_zero_like(candidate, execution, result_profile=None):
        return None

    try:
        statement = sqlglot.parse_one(candidate.sql, read="snowflake")
    except ParseError:
        return FilterGroundingReport(
            exact_filters=[],
            probes=[],
            value_rewrites=[],
            reason="SQL could not be parsed for filter grounding.",
        )

    exact_filters = _extract_exact_string_filters(statement)
    if not exact_filters:
        return None

    selected_tables = validation.referenced_tables or schema.expanded_tables
    probes: list[dict[str, object]] = []
    value_rewrites: list[dict[str, object]] = []
    seen_probe_keys: set[tuple[str, str, str]] = set()

    for filter_match in exact_filters:
        targets = _filter_probe_targets(
            filter_match,
            selected_tables=selected_tables,
            table_schemas=table_schemas,
        )
        for target in targets:
            probe_key = (target["table"], target["column"], target["literal"])
            if probe_key in seen_probe_keys:
                continue
            seen_probe_keys.add(probe_key)

            probe_sql = _string_filter_probe_sql(
                table_name=target["table"],
                column_name=target["column"],
                literal=target["literal"],
            )
            try:
                probe_frame = _fetch_query_dataframe(probe_sql, db=task.db)
            except Exception as exc:
                probes.append(
                    {
                        **target,
                        "probe_sql": probe_sql,
                        "error": str(exc),
                    }
                )
                continue

            matched_values = [
                str(value)
                for value in (
                    row.get("MATCHED_VALUE")
                    or row.get("matched_value")
                    or row.get(target["column"])
                    for row in probe_frame.to_dict(orient="records")
                )
                if value not in {None, "", target["literal"]}
            ]
            probe_record: dict[str, object] = {
                **target,
                "probe_sql": probe_sql,
                "matched_values": matched_values,
            }
            probes.append(probe_record)
            if matched_values:
                value_rewrites.append(
                    {
                        "filter": target["raw_filter"],
                        "rewrite": matched_values[0],
                        "table": target["table"],
                        "column": target["column"],
                        "probe_sql": probe_sql,
                    }
                )
                break
        if value_rewrites:
            break

    if not probes:
        return FilterGroundingReport(
            exact_filters=[filter_match["raw_filter"] for filter_match in exact_filters],
            probes=[],
            value_rewrites=[],
            zero_like_result=True,
            reason="No probe targets were available for the empty result.",
        )

    reason = (
        "Empty result but probe values suggest a stored label variant."
        if value_rewrites
        else "Empty result and probes did not find a grounded label rewrite."
    )
    return FilterGroundingReport(
        exact_filters=[filter_match["raw_filter"] for filter_match in exact_filters],
        probes=probes,
        value_rewrites=value_rewrites,
        zero_like_result=True,
        reason=reason,
    )


def _execution_is_zero_like(
    candidate: SQLCandidate,
    execution: ExecutionResult,
    *,
    result_profile: dict[str, Any] | None,
) -> bool:
    """Return True when one result row still represents an empty aggregate result."""

    if not execution.ok:
        return False
    if execution.row_count == 0:
        return True
    if not looks_aggregate_query(candidate.sql):
        return False

    profile = result_profile or {
        "sample_rows": execution.sample_rows,
    }
    sample_rows = profile.get("sample_rows") or execution.sample_rows
    numeric_values = [
        value
        for value in (coerce_number(item) for row in sample_rows for item in row.values())
        if value is not None
    ]
    return bool(numeric_values) and max(numeric_values) == 0


def _extract_exact_string_filters(statement: exp.Expression) -> list[dict[str, str]]:
    """Return exact string equality filters from one parsed query."""

    filters: list[dict[str, str]] = []
    for node in statement.walk():
        if isinstance(node, exp.EQ):
            comparison = _extract_string_equality(node)
            if comparison is not None:
                filters.append(comparison)
        elif isinstance(node, exp.In):
            comparison = _extract_string_membership(node)
            if comparison is not None:
                filters.extend(comparison)
    return filters


def _extract_string_equality(node: exp.EQ) -> dict[str, str] | None:
    """Return one column-to-literal equality comparison when present."""

    left = node.left
    right = node.right
    column, literal = _column_and_string_literal(left, right)
    if column is None or literal is None:
        return None
    return {
        "raw_filter": node.sql(dialect="snowflake"),
        "column": column.sql(dialect="snowflake"),
        "literal": literal,
    }


def _extract_string_membership(node: exp.In) -> list[dict[str, str]] | None:
    """Return one column-to-literal IN comparison when present."""

    column = node.this if isinstance(node.this, exp.Column) else None
    if column is None:
        return None

    filters: list[dict[str, str]] = []
    for expression in node.expressions:
        if not isinstance(expression, exp.Literal) or not expression.is_string:
            return None
        filters.append(
            {
                "raw_filter": node.sql(dialect="snowflake"),
                "column": column.sql(dialect="snowflake"),
                "literal": expression.this,
            }
        )
    return filters


def _column_and_string_literal(
    left: exp.Expression,
    right: exp.Expression,
) -> tuple[exp.Column | None, str | None]:
    """Return a column and a string literal from one equality comparison."""

    if isinstance(left, exp.Column) and isinstance(right, exp.Literal) and right.is_string:
        return left, right.this
    if isinstance(right, exp.Column) and isinstance(left, exp.Literal) and left.is_string:
        return right, left.this
    return None, None


def _filter_probe_targets(
    filter_match: dict[str, str],
    *,
    selected_tables: list[str],
    table_schemas: dict[str, TableSchema],
) -> list[dict[str, str]]:
    """Return low-cost probe targets for one empty exact filter."""

    column_name = filter_match["column"].split(".")[-1].strip('"')
    literal = filter_match["literal"]
    schema_lookup = {
        table_name.lower(): table_schema for table_name, table_schema in table_schemas.items()
    }
    tables = [table_name for table_name in selected_tables if table_name.lower() in schema_lookup]
    tables.sort(key=lambda name: _table_probe_rank(name, column_name))

    targets: list[dict[str, str]] = []
    for table_name in tables:
        schema = schema_lookup[table_name.lower()]
        column_candidates = _probe_columns_for_table(schema, column_name, literal)
        for probe_column in column_candidates:
            targets.append(
                {
                    "raw_filter": filter_match["raw_filter"],
                    "table": table_name,
                    "column": probe_column,
                    "literal": literal,
                }
            )
            if len(targets) >= DEFAULT_FILTER_GROUNDING_POLICY.max_probe_targets:
                return targets
    return targets


def _probe_columns_for_table(
    table_schema: TableSchema,
    filter_column: str,
    literal: str,
) -> list[str]:
    """Return likely string columns to probe within one table."""

    string_columns = [
        column.name for column in table_schema.columns if column_looks_string_like(column.type)
    ]
    if not string_columns:
        return []

    exact_matches = [
        column.name
        for column in table_schema.columns
        if column.name.lower() == filter_column.lower()
    ]
    if exact_matches:
        return exact_matches

    preferred = [column for column in string_columns if filter_column.lower() in column.lower()]
    if preferred:
        return preferred
    return string_columns[: DEFAULT_FILTER_GROUNDING_POLICY.fallback_string_columns]


def _table_probe_rank(table_name: str, filter_column: str) -> tuple[int, str]:
    """Rank tables so the most obviously related table is probed first."""

    lowered = table_name.lower()
    if filter_column.lower() in lowered:
        return 0, lowered
    return 1, lowered


def _string_filter_probe_sql(*, table_name: str, column_name: str, literal: str) -> str:
    """Build one low-cost LIKE probe for a string filter."""

    escaped = literal.replace("'", "''")
    return (
        f'SELECT DISTINCT "{column_name}" AS MATCHED_VALUE '
        f"FROM {table_name} "
        f"WHERE LOWER(CAST(\"{column_name}\" AS VARCHAR)) LIKE LOWER('%{escaped}%') "
        f"LIMIT {DEFAULT_FILTER_GROUNDING_POLICY.probe_match_limit}"
    )
