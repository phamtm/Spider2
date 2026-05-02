"""SQL and result verification heuristics for candidate evaluation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from sol01.candidate_scoring import (
    _coerce_number,
    _expects_scalar_output,
    _normalized_output_expectation,
)
from sol01.index import CACHE_PATH
from sol01.retrieval import load_db_index
from sol01.logging import get_logger
from sol01.models import (
    AggregateGrainReport,
    ExecutionResult,
    FilterGroundingReport,
    Intent,
    OutputShapeReport,
    SchemaSelection,
    SQLCandidate,
    TableSchema,
    Task,
    ValidationReport,
)
from sol01.prompt_builders import _infer_native_value_terms

logger = get_logger(__name__)


def _fetch_query_dataframe(sql: str, *, db: str):
    """Run one query without importing the Snowflake stack at startup."""

    from sol01.snowflake_runner import fetch_query_dataframe as _impl

    return _impl(sql, db=db)


def _infer_aggregate_grain(
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
    if not _looks_aggregate_query(normalized_sql):
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


def _infer_output_shape_report(
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


def _infer_filter_grounding_report(
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

    if value_rewrites:
        reason = "Empty result but probe values suggest a stored label variant."
    else:
        reason = "Empty result and probes did not find a grounded label rewrite."

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
    if not _looks_aggregate_query(candidate.sql):
        return False

    profile = result_profile or {
        "sample_rows": execution.sample_rows,
    }
    sample_rows = profile.get("sample_rows") or execution.sample_rows
    numeric_values = [
        value
        for value in (_coerce_number(item) for row in sample_rows for item in row.values())
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
            if len(targets) >= 4:
                return targets
    return targets


def _probe_columns_for_table(
    table_schema: TableSchema,
    filter_column: str,
    literal: str,
) -> list[str]:
    """Return likely string columns to probe within one table."""

    string_columns = [
        column.name for column in table_schema.columns if _column_looks_string_like(column.type)
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

    if _table_looks_like_lookup(table_schema) or _literal_looks_label_like(literal):
        preferred = [
            column
            for column in string_columns
            if _column_looks_label_like(column) or _column_looks_key_like(column)
        ]
        if preferred:
            return preferred

    preferred = [column for column in string_columns if filter_column.lower() in column.lower()]
    if preferred:
        return preferred
    return string_columns[:2]


def _table_probe_rank(table_name: str, filter_column: str) -> tuple[int, str]:
    """Rank tables so likely lookup tables are probed first."""

    lowered = table_name.lower()
    if filter_column.lower() in lowered:
        return 0, lowered
    if _table_looks_like_lookup_name(lowered):
        return 1, lowered
    return 2, lowered


def _string_filter_probe_sql(*, table_name: str, column_name: str, literal: str) -> str:
    """Build one low-cost LIKE probe for a string filter."""

    escaped = literal.replace("'", "''")
    return (
        f'SELECT DISTINCT "{column_name}" AS MATCHED_VALUE '
        f"FROM {table_name} "
        f"WHERE LOWER(CAST(\"{column_name}\" AS VARCHAR)) LIKE LOWER('%{escaped}%') "
        "LIMIT 5"
    )


def _column_looks_string_like(column_type: str | None) -> bool:
    """Return True when a schema column looks like a text field."""

    if column_type is None:
        return True
    lowered = column_type.lower()
    return any(token in lowered for token in ("char", "text", "string", "varchar", "variant"))


def _table_looks_like_lookup(table_schema: TableSchema) -> bool:
    """Return True when a table name suggests a lookup or code table."""

    return _table_looks_like_lookup_name((table_schema.full_name or table_schema.name).lower())


def _table_looks_like_lookup_name(table_name: str) -> bool:
    """Return True when a table name suggests a lookup or code table."""

    return any(
        token in table_name
        for token in (
            "summary",
            "lookup",
            "ref_",
            "_ref",
            "dim_",
            "_dim",
            "map",
            "code",
            "label",
            "country",
        )
    )


def _column_looks_label_like(column_name: str) -> bool:
    """Return True when a column name looks like a human-readable label."""

    lowered = column_name.lower()
    return any(token in lowered for token in ("name", "label", "display", "title", "desc"))


def _column_looks_key_like(column_name: str) -> bool:
    """Return True when a column name looks like a stored key."""

    lowered = column_name.lower()
    return any(token in lowered for token in ("key", "code", "id"))


def _literal_looks_label_like(literal: str) -> bool:
    """Return True when a filter literal looks like a display label."""

    return (any(part.isalpha() for part in literal) and " " in literal) or literal[:1].isupper()


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


def _aggregate_grain_guidance(
    task: Task,
    intent: Intent,
    schema: SchemaSelection,
    table_schemas: dict[str, TableSchema],
) -> str | None:
    """Return prompt guidance when a task looks like a plain count over one entity table."""

    normalized_question = " ".join(
        [
            task.question.lower(),
            intent.summary.lower(),
            intent.output_expectation.lower(),
            " ".join(intent.metrics).lower(),
            " ".join(intent.filters).lower(),
            " ".join(intent.assumptions).lower(),
        ]
    )
    if not (
        _expects_scalar_output(_normalized_output_expectation(intent))
        or any(token in normalized_question for token in ("how many", "count", "number of"))
    ):
        return None

    question_text = " ".join(
        [
            task.question,
            intent.summary,
            intent.output_expectation,
            " ".join(intent.metrics),
            " ".join(intent.filters),
            " ".join(intent.assumptions),
        ]
    )
    selected_tables = [
        table_schemas.get(table_name) or TableSchema(name=table_name, ddl="", searchable_text="")
        for table_name in schema.expanded_tables
    ]
    entity_like_tables = [table for table in selected_tables if _table_is_entity_like(table)]
    if len(selected_tables) != 1 or len(entity_like_tables) != 1:
        return None

    table_name = entity_like_tables[0].full_name or entity_like_tables[0].name
    if _question_requests_unique_entities(question_text):
        return (
            f"Grain guidance: {table_name} looks like a single entity table, and the question "
            "asks for unique or deduplicated entities. Use COUNT(DISTINCT ...) only if the "
            "intent explicitly requires uniqueness."
        )

    return (
        f"Grain guidance: {table_name} looks like a single entity table. Treat this as a "
        "row-count style aggregation and default to COUNT(*) per group. Do not switch to "
        "COUNT(DISTINCT ...) unless the question explicitly asks for unique or deduplicated "
        "entities."
    )


def _metric_source_guidance(
    task: Task,
    intent: Intent,
    table_schemas: dict[str, TableSchema],
) -> str | None:
    """Return task-aware guidance for choosing metric source columns."""

    if not table_schemas:
        return None

    task_text = " ".join(
        [
            task.question,
            intent.summary,
            intent.output_expectation,
            " ".join(intent.metrics),
            " ".join(intent.filters),
            " ".join(intent.assumptions),
        ]
    )
    if not _looks_metric_source_sensitive(task_text):
        return None

    table_lines: list[str] = []
    for table_name in sorted(table_schemas):
        table = table_schemas[table_name]
        metric_columns = [column for column in table.columns if _column_looks_native_metric(column)]
        if not metric_columns:
            continue
        metric_columns = sorted(
            metric_columns,
            key=lambda column: _metric_column_sort_key(column, task_text),
        )
        grain_columns = [
            column.name for column in table.columns if _column_looks_grain_or_filter_column(column)
        ]
        table_identity = table.full_name or table_name
        metrics_preview = ", ".join(column.name for column in metric_columns[:6])
        if grain_columns:
            grain_preview = ", ".join(grain_columns[:8])
            table_lines.append(
                f"- {table_identity}: native metrics [{metrics_preview}]; "
                f"grain/filter/time columns [{grain_preview}]"
            )
        else:
            table_lines.append(f"- {table_identity}: native metrics [{metrics_preview}]")
        if len(table_lines) >= 8:
            break

    if not table_lines:
        return None

    return (
        "Choose the metric source at the requested answer grain. Prefer a native metric "
        "column on a table that already has the needed grouping keys, time key, and "
        "filters when its semantics match the question. Join lower-grain detail tables "
        "only when the task requires detail-level filters, grouping, output columns, an "
        "explicit formula, or no suitable native metric exists.\n"
        "When several native metric columns exist, choose by column-name semantics from "
        "the question; do not treat subtotal, total due, tax, freight, or line-item "
        "formulas as interchangeable. Candidate metric columns below are ordered by "
        "semantic fit to the task text.\n"
        "Native metric candidates in selected tables:\n" + "\n".join(table_lines)
    )


def _looks_metric_source_sensitive(text: str) -> bool:
    """Return True when the task likely depends on choosing the right metric source."""

    normalized = text.lower()
    return any(
        token in normalized
        for token in (
            "amount",
            "annual",
            "average",
            "balance",
            "compare",
            "cost",
            "difference",
            "fee",
            "freight",
            "metric",
            "payment",
            "price",
            "quota",
            "rate",
            "revenue",
            "sales",
            "sum",
            "tax",
            "total",
            "value",
        )
    )


def _column_looks_native_metric(column: Any) -> bool:
    """Return True for columns that look like stored business metrics."""

    if not _column_looks_numeric_like(column.type):
        return False
    if _column_looks_key_like(column.name):
        return False
    normalized = column.name.lower().replace("_", "")
    return any(
        token in normalized
        for token in (
            "amount",
            "amt",
            "balance",
            "cost",
            "due",
            "fare",
            "fee",
            "freight",
            "price",
            "quota",
            "rate",
            "revenue",
            "sales",
            "subtotal",
            "tax",
            "total",
            "value",
        )
    )


def _metric_column_sort_key(column: Any, task_text: str) -> tuple[int, str]:
    """Rank metric columns so prompt candidates lead with better semantic matches."""

    normalized_text = task_text.lower()
    normalized_name = column.name.lower().replace("_", "")
    score = 0

    if "quota" in normalized_text and "quota" in normalized_name:
        score += 8
    if "sales" in normalized_text and "sales" in normalized_name:
        score += 6
    if "revenue" in normalized_text and "revenue" in normalized_name:
        score += 6
    if "total" in normalized_text and "total" in normalized_name:
        score += 5
    if "amount" in normalized_text and ("amount" in normalized_name or "amt" in normalized_name):
        score += 5
    if "due" in normalized_text and "due" in normalized_name:
        score += 5
    if "tax" in normalized_text and "tax" in normalized_name:
        score += 5
    if (
        any(token in normalized_text for token in ("freight", "shipping"))
        and "freight" in normalized_name
    ):
        score += 5

    if "subtotal" in normalized_name and not any(
        token in normalized_text for token in ("subtotal", "sub total", "net", "pre-tax", "pretax")
    ):
        score -= 3
    if any(token in normalized_name for token in ("tax", "freight")) and not any(
        token in normalized_text for token in ("tax", "freight", "shipping")
    ):
        score -= 4

    return (-score, normalized_name)


def _column_looks_numeric_like(column_type: str | None) -> bool:
    """Return True when a schema type appears numeric."""

    if column_type is None:
        return False
    lowered = column_type.lower()
    return any(
        token in lowered
        for token in ("number", "numeric", "decimal", "int", "float", "double", "real")
    )


def _column_looks_grain_or_filter_column(column: Any) -> bool:
    """Return True for columns likely to define grouping, time, joins, or filters."""

    normalized = column.name.lower()
    return any(
        token in normalized
        for token in (
            "date",
            "time",
            "year",
            "month",
            "day",
            "week",
            "quarter",
            "period",
            "id",
            "key",
            "code",
            "name",
            "type",
            "status",
            "category",
            "group",
        )
    )


def _count_distinct_target(sql: str) -> str | None:
    """Extract the DISTINCT target column from a COUNT aggregate."""

    match = re.search(r"count\s*\(\s*distinct\s+([^)]+)\)", sql, flags=re.IGNORECASE)
    if match is None:
        return None
    target = match.group(1).strip()
    target = re.sub(r"\s+as\s+\w+$", "", target, flags=re.IGNORECASE)
    return target.strip(" \"'`")


def _aggregate_verification_reason(attempt: dict[str, Any]) -> str | None:
    """Return a reason when an aggregate result looks suspiciously small."""

    execution = attempt["execution_result"]
    if not execution["ok"]:
        return None
    sql = str(attempt["sql"]).lower()
    if not _looks_aggregate_query(sql):
        return None

    sample_rows = execution.get("sample_rows") or []
    if not sample_rows:
        return None

    first_row = sample_rows[0]
    numeric_values = [
        value
        for value in (_coerce_number(item) for item in first_row.values())
        if value is not None
    ]
    if not numeric_values:
        return None

    row_count = int(execution.get("row_count") or 0)
    max_value = max(numeric_values)
    if row_count == 1 and max_value <= 1:
        return "Aggregate query returned a single very small numeric result."
    if row_count <= 2 and max_value <= 2:
        return "Aggregate query returned only tiny numeric results."
    return None


def _looks_aggregate_query(sql: str) -> bool:
    """Heuristically detect aggregate queries that deserve extra scrutiny."""

    normalized = " ".join(sql.lower().split())
    return any(
        keyword in normalized
        for keyword in (" count(", " sum(", " avg(", " min(", " max(", " group by ", " having ")
    )


def _augment_intent_with_value_groundings(
    intent: Intent,
    *,
    task: Task,
    schema: SchemaSelection,
    table_schemas: dict[str, TableSchema],
) -> Intent:
    """Attach native value matches from selected schemas to the answer contract."""

    native_value_terms = _infer_native_value_terms(task, schema, table_schemas)
    if not native_value_terms:
        return intent

    merged_terms = list(dict.fromkeys([*intent.native_value_terms, *native_value_terms]))
    return intent.model_copy(update={"native_value_terms": merged_terms})


def _table_schemas_for_selection(
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
