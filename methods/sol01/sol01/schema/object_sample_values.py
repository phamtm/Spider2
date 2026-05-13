"""Build sample-value schema objects."""

from __future__ import annotations

from dataclasses import dataclass

from sol01.models import ColumnSchema, SchemaObject, TableSchema
from sol01.schema import objects as common

DEFAULT_MAX_SAMPLE_VALUES_PER_COLUMN = common.DEFAULT_MAX_SAMPLE_VALUES_PER_COLUMN
_has_categorical_name = common._has_categorical_name
_has_text_like_name = common._has_text_like_name
_is_indexable_sample_value = common._is_indexable_sample_value
_is_key_like = common._is_key_like
_is_time_like = common._is_time_like
_join_text = common._join_text
_primitive_type = common._primitive_type
_stable_distinct = common._stable_distinct
_stable_hash = common._stable_hash


def sample_value_objects(
    table_key: str,
    table: TableSchema,
    table_full_name: str,
    *,
    max_sample_values_per_column: int,
) -> list[SchemaObject]:
    """Create sample-value objects only for bounded categorical evidence."""

    objects: list[SchemaObject] = []
    for column in table.columns:
        decision = _sample_value_decision(column, table.sample_rows)
        if not decision.include:
            continue

        for value in decision.values[:max_sample_values_per_column]:
            value_text = str(value)
            value_hash = _stable_hash(table_full_name, column.name, value_text)
            objects.append(
                SchemaObject(
                    object_id=f"sample_value:{table_full_name}#{column.name}:{value_hash}",
                    object_type="sample_value",
                    name=value_text,
                    db=table.database_name,
                    table_name=table_full_name,
                    column_name=column.name,
                    searchable_text=_join_text([value_text, column.name, table_full_name]),
                    metadata={
                        "source": "load_db_index",
                        "source_table_key": table_key,
                        "database_name": table.database_name,
                        "schema_name": table.schema_name,
                        "table_full_name": table_full_name,
                        "column_name": column.name,
                        "column_type": column.type,
                        "value": value,
                        "sample_size": decision.sample_size,
                        "distinct_count": decision.distinct_count,
                        "inclusion_reason": decision.reason,
                        "provenance": {
                            "source": "load_db_index",
                            "table_key": table_key,
                            "sample_size": decision.sample_size,
                            "distinct_count": decision.distinct_count,
                        },
                    },
                )
            )
    return objects


@dataclass(frozen=True)
class _SampleValueDecision:
    """Decision details for one column's sample-value indexing policy."""

    include: bool
    values: list[object]
    sample_size: int
    distinct_count: int
    reason: str


def _sample_value_decision(
    column: ColumnSchema,
    sample_rows: list[dict[str, object]],
) -> _SampleValueDecision:
    """Return the conservative sample-value indexing decision for one column."""

    raw_values = [
        row[column.name]
        for row in sample_rows
        if column.name in row and row[column.name] not in (None, "")
    ]
    distinct_values = _stable_distinct(raw_values)
    sample_size = len(raw_values)
    distinct_count = len(distinct_values)
    primitive_type = _primitive_type(column.type)

    if sample_size < 2 or not distinct_values:
        return _exclude_sample_values(sample_size, distinct_count, "insufficient_sample")
    if _is_key_like(column):
        return _exclude_sample_values(sample_size, distinct_count, "key_like_column")
    if _is_time_like(column):
        return _exclude_sample_values(sample_size, distinct_count, "time_like_column")
    if primitive_type in {"numeric", "temporal", "semi_structured"}:
        return _exclude_sample_values(sample_size, distinct_count, f"{primitive_type}_column")
    if _has_text_like_name(column.name):
        return _exclude_sample_values(sample_size, distinct_count, "raw_text_like_column")
    if any(not _is_indexable_sample_value(value) for value in distinct_values):
        return _exclude_sample_values(sample_size, distinct_count, "unsafe_value_shape")
    if distinct_count > DEFAULT_MAX_SAMPLE_VALUES_PER_COLUMN:
        return _exclude_sample_values(sample_size, distinct_count, "too_many_distinct_values")
    if distinct_count == sample_size and not _has_categorical_name(column.name):
        return _exclude_sample_values(sample_size, distinct_count, "unknown_cardinality")

    reason = (
        "repeated_bounded_sample" if distinct_count < sample_size else "categorical_column_name"
    )
    return _SampleValueDecision(
        include=True,
        values=distinct_values,
        sample_size=sample_size,
        distinct_count=distinct_count,
        reason=reason,
    )


def _exclude_sample_values(
    sample_size: int,
    distinct_count: int,
    reason: str,
) -> _SampleValueDecision:
    """Return a negative sample-value decision with diagnostic counts."""

    return _SampleValueDecision(
        include=False,
        values=[],
        sample_size=sample_size,
        distinct_count=distinct_count,
        reason=reason,
    )
