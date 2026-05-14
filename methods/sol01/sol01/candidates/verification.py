"""SQL and result verification heuristics for candidate evaluation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from sol01.candidates.filter_grounding import (
    infer_filter_grounding_report as infer_filter_grounding_report_impl,
)
from sol01.candidates.query_heuristics import looks_aggregate_query
from sol01.infra.logging import get_logger
from sol01.llm.prompt_builders import infer_native_value_terms
from sol01.models import (
    AggregateGrainReport,
    ExecutionResult,
    Intent,
    OutputShapeReport,
    SchemaSelection,
    SQLCandidate,
    TableSchema,
    Task,
)
from sol01.schema.db_index import load_db_index
from sol01.schema.index import CACHE_PATH

logger = get_logger(__name__)


def infer_aggregate_grain(
    *,
    task: Task,
    candidate: SQLCandidate,
    schema: SchemaSelection,
    table_schemas: dict[str, TableSchema],
    validation: Any,
    execution: Any,
) -> AggregateGrainReport | None:
    """Infer the intended aggregate grain for one candidate."""

    sql = str(candidate.sql)
    normalized_sql = " ".join(sql.lower().split())
    if not looks_aggregate_query(normalized_sql):
        return None

    uses_distinct = "count(distinct" in normalized_sql
    has_joins = " join " in normalized_sql
    selected_tables = [
        table_schemas.get(table_name) or TableSchema(name=table_name, ddl="", searchable_text="")
        for table_name in schema.expanded_tables
    ]
    entity_like_tables = [table for table in selected_tables if _table_is_entity_like(table)]
    question_text = task.question.lower()
    wants_unique = _question_requests_unique_entities(question_text)
    target = _count_distinct_target(sql)

    if uses_distinct:
        if entity_like_tables and not has_joins and not wants_unique:
            table_name = entity_like_tables[0].full_name or entity_like_tables[0].name
            reason = (
                f"Single entity table {table_name} has no joins, so DISTINCT is likely redundant."
            )
            distinct_reason = (
                f"DISTINCT on {target or 'the counted column'} is probably unnecessary."
            )
            return AggregateGrainReport(
                inferred_grain="row_count",
                reason=reason,
                distinct_reason=distinct_reason,
                uses_distinct=True,
                has_joins=has_joins,
                selected_tables=[table.full_name or table.name for table in selected_tables],
            )

        if has_joins or wants_unique:
            reason = "Joins or unique-count wording suggest deduping entity rows."
            distinct_reason = f"DISTINCT on {target or 'the counted column'} is justified here."
            return AggregateGrainReport(
                inferred_grain="distinct_entity_count",
                reason=reason,
                distinct_reason=distinct_reason,
                uses_distinct=True,
                has_joins=has_joins,
                selected_tables=[table.full_name or table.name for table in selected_tables],
            )

        reason = "DISTINCT appears to be a cautious choice, but the target grain is unclear."
        distinct_reason = f"DISTINCT on {target or 'the counted column'} may be unnecessary."
        return AggregateGrainReport(
            inferred_grain="unknown",
            reason=reason,
            distinct_reason=distinct_reason,
            uses_distinct=True,
            has_joins=has_joins,
            selected_tables=[table.full_name or table.name for table in selected_tables],
        )

    if entity_like_tables and not has_joins:
        table_name = entity_like_tables[0].full_name or entity_like_tables[0].name
        return AggregateGrainReport(
            inferred_grain="row_count",
            reason=f"Single entity table {table_name} with no joins usually counts rows.",
            uses_distinct=False,
            has_joins=has_joins,
            selected_tables=[table.full_name or table.name for table in selected_tables],
        )

    if has_joins:
        return AggregateGrainReport(
            inferred_grain="value_count",
            reason=(
                "Join multiplicity suggests the result may count joined rows rather than entities."
            ),
            uses_distinct=False,
            has_joins=has_joins,
            selected_tables=[table.full_name or table.name for table in selected_tables],
        )

    return AggregateGrainReport(
        inferred_grain="unknown",
        reason="Aggregate query does not clearly indicate a unique entity grain.",
        uses_distinct=False,
        has_joins=has_joins,
        selected_tables=[table.full_name or table.name for table in selected_tables],
    )


def _projection_columns(statement: exp.Expression) -> list[str]:
    """Return the projected output column names from one query."""

    columns: list[str] = []
    for expression in getattr(statement, "expressions", []):
        name = _expression_output_name(expression)
        if name is not None:
            columns.append(name)
    return columns


def _grouped_columns(statement: exp.Expression) -> list[str]:
    """Return simple grouped column names from one query."""

    group = statement.args.get("group")
    if group is None:
        return []

    columns: list[str] = []
    for expression in group.expressions:
        if isinstance(expression, exp.Column):
            columns.append(_normalized_column_name(expression))
    return columns


def _expression_output_name(expression: exp.Expression) -> str | None:
    """Return the visible output name for one select expression."""

    if _is_aggregate_projection(expression):
        alias = getattr(expression, "alias_or_name", None)
        if alias:
            return str(alias)
        return None

    if isinstance(expression, exp.Alias):
        alias = expression.alias_or_name
        return str(alias) if alias else None
    if isinstance(expression, exp.Column):
        return _normalized_column_name(expression)

    alias = getattr(expression, "alias_or_name", None)
    if alias:
        return str(alias)

    if expression.is_star:
        return "*"
    return None


def _is_aggregate_projection(expression: exp.Expression) -> bool:
    """Return True when one select expression contains an aggregate."""

    return expression.find(exp.AggFunc) is not None


def _normalized_column_name(expression: exp.Column) -> str:
    """Return the visible column name for one AST column node."""

    return str(expression.alias_or_name or expression.name)


def _is_identifier_like_column_name(column_name: str) -> bool:
    """Return True for likely identifier columns that should stay visible."""

    normalized = column_name.replace('"', "").replace("`", "")
    return bool(
        re.search(r"(?:^|[_\W])id$", normalized, flags=re.IGNORECASE)
        or re.search(r"[A-Z]ID$", column_name)
        or normalized.lower().endswith("_id")
        or normalized.lower().endswith("id")
    )


def infer_output_shape_report(
    *,
    intent: Intent,
    candidate: SQLCandidate,
    execution: ExecutionResult,
    result_profile: dict[str, Any] | None,
) -> OutputShapeReport | None:
    """Infer the answer contract from the SQL, intent, and executed result."""

    if not execution.ok:
        return None

    profile = result_profile or {
        "columns": execution.columns,
        "sample_rows": execution.sample_rows,
    }
    observed_columns = [str(column) for column in profile.get("columns") or execution.columns]

    try:
        statement = sqlglot.parse_one(candidate.sql, read="snowflake")
    except ParseError:
        return OutputShapeReport(
            expected_columns=observed_columns,
            observed_columns=observed_columns,
            violations=["SQL could not be parsed for shape analysis."],
        )

    projected_columns = _projection_columns(statement)
    grouped_columns = _grouped_columns(statement)
    expected_columns = list(dict.fromkeys(projected_columns))
    violations: list[str] = []
    has_non_aggregate_projection = any(
        not _is_aggregate_projection(expression) for expression in statement.expressions
    )

    for grouped_column in grouped_columns:
        if (
            _is_identifier_like_column_name(grouped_column)
            and grouped_column not in expected_columns
            and has_non_aggregate_projection
        ):
            expected_columns.append(grouped_column)
            violations.append(f"missing grouped key {grouped_column}")

    missing_columns = [column for column in expected_columns if column not in observed_columns]
    if missing_columns:
        violations.extend(f"missing expected column {column}" for column in missing_columns)

    return OutputShapeReport(
        expected_columns=expected_columns,
        observed_columns=observed_columns,
        projected_columns=projected_columns,
        grouped_columns=grouped_columns,
        missing_columns=missing_columns,
        violations=list(dict.fromkeys(violations)),
    )


def infer_filter_grounding_report(
    *,
    task: Task,
    candidate: SQLCandidate,
    schema: SchemaSelection,
    table_schemas: dict[str, TableSchema],
    validation: Any,
    execution: Any,
):
    """Keep the public verification entrypoint while delegating the heavy probing logic."""

    return infer_filter_grounding_report_impl(
        task=task,
        candidate=candidate,
        schema=schema,
        table_schemas=table_schemas,
        validation=validation,
        execution=execution,
    )


def _table_is_entity_like(table: TableSchema) -> bool:
    """Return True when a table looks like a master/entity table."""

    table_name = (table.full_name or table.name).split(".")[-1].lower()
    return (
        table_name.startswith(("mst_", "dim_", "ref_", "lkp_"))
        or table_name.endswith(("_users", "_user"))
        or "master" in table_name
        or "entity" in table_name
    )


def _question_requests_unique_entities(question_text: str) -> bool:
    """Return True when the wording asks for unique or deduplicated entities."""

    normalized = question_text.lower()
    return any(
        keyword in normalized
        for keyword in ("unique", "distinct", "dedupe", "dedup", "without duplicates")
    )


def _count_distinct_target(sql: str) -> str | None:
    """Extract the DISTINCT target column from a COUNT aggregate."""

    match = re.search(r"count\s*\(\s*distinct\s+([^)]+)\)", sql, flags=re.IGNORECASE)
    if match is None:
        return None
    target = match.group(1).strip()
    target = re.sub(r"\s+as\s+\w+$", "", target, flags=re.IGNORECASE)
    return target.strip(" \"'`")


def augment_intent_with_value_groundings(
    intent: Intent,
    *,
    task: Task,
    schema: SchemaSelection,
    table_schemas: dict[str, TableSchema],
) -> Intent:
    """Attach native value matches from selected schemas to the answer contract."""

    native_value_terms = infer_native_value_terms(task, schema, table_schemas)
    if not native_value_terms:
        return intent

    merged_terms = list(dict.fromkeys([*intent.native_value_terms, *native_value_terms]))
    return intent.model_copy(update={"native_value_terms": merged_terms})


def table_schemas_for_selection(
    schema: SchemaSelection,
    *,
    cache_path: Path = CACHE_PATH,
) -> dict[str, TableSchema]:
    """Return indexed schemas for the selected tables, falling back safely in tests."""

    try:
        db_index = load_db_index(schema.db, cache_path=cache_path)
    except (FileNotFoundError, NotADirectoryError):
        logger.warning("schema index unavailable", db=schema.db, cache_path=str(cache_path))
        return {}

    selected: dict[str, TableSchema] = {}
    for table_name in schema.expanded_tables:
        table = db_index.get(table_name)
        if table is not None:
            selected[table_name] = table
    return selected


_infer_aggregate_grain = infer_aggregate_grain
