"""Table, column, group, and sample-value schema objects."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from sol01.models import ColumnSchema, SchemaObject, TableSchema

from ._object_shared import (
    DEFAULT_MAX_SAMPLE_VALUES_PER_COLUMN,
    _ColumnRef,
    _first_non_empty,
    _has_categorical_name,
    _has_text_like_name,
    _is_indexable_sample_value,
    _is_key_like,
    _is_numeric_measure_candidate,
    _is_time_like,
    _join_text,
    _normalize_identifier,
    _primitive_type,
    _SampleValueDecision,
    _slug,
    _stable_distinct,
    _stable_hash,
    _table_full_name,
    _tokens,
)


def _column_refs(db_index: Mapping[str, TableSchema]) -> list[_ColumnRef]:
    """Flatten table columns into sorted references for deterministic processing."""

    refs: list[_ColumnRef] = []
    for table_key in sorted(db_index):
        table = db_index[table_key]
        table_full_name = _table_full_name(table_key, table)
        for column in sorted(table.columns, key=lambda item: item.name):
            refs.append(
                _ColumnRef(
                    table_key=table_key,
                    table=table,
                    column=column,
                    table_full_name=table_full_name,
                    database_name=table.database_name,
                    schema_name=table.schema_name,
                )
            )
    return refs


def _table_object(table_key: str, table: TableSchema, table_full_name: str) -> SchemaObject:
    columns = [column.name for column in table.columns]
    description_parts = [column.description for column in table.columns if column.description]
    searchable_parts = [
        table_full_name,
        table.name,
        table.searchable_text,
        " ".join(columns),
        " ".join(description_parts),
    ]
    return SchemaObject(
        object_id=f"table:{table_full_name}",
        object_type="table",
        name=table_full_name,
        db=table.database_name,
        table_name=table_full_name,
        description=_first_non_empty(description_parts),
        searchable_text=_join_text(searchable_parts),
        metadata={
            "source": "load_db_index",
            "source_table_key": table_key,
            "database_name": table.database_name,
            "schema_name": table.schema_name,
            "full_name": table_full_name,
            "short_name": table.name,
            "ddl": table.ddl,
            "columns": [column.model_dump(mode="json") for column in table.columns],
            "sample_row_count": len(table.sample_rows),
            "normalized_table_name": _normalize_identifier(table.name),
            "name_tokens": _tokens(table.name),
            "family_hooks": {
                "database_name": table.database_name,
                "schema_name": table.schema_name,
                "normalized_table_name": _normalize_identifier(table.name),
                "name_tokens": _tokens(table.name),
            },
            "provenance": {"source": "load_db_index", "table_key": table_key},
        },
    )


def _column_object(ref: _ColumnRef) -> SchemaObject:
    primitive_type = _primitive_type(ref.column.type)
    searchable_parts = [
        ref.table_full_name,
        ref.column.name,
        ref.column.type,
        ref.column.description,
    ]
    return SchemaObject(
        object_id=f"column:{ref.table_full_name}#{ref.column.name}",
        object_type="column",
        name=f"{ref.table_full_name}.{ref.column.name}",
        db=ref.database_name,
        table_name=ref.table_full_name,
        column_name=ref.column.name,
        description=ref.column.description,
        searchable_text=_join_text(searchable_parts),
        metadata={
            "source": "load_db_index",
            "source_table_key": ref.table_key,
            "database_name": ref.database_name,
            "schema_name": ref.schema_name,
            "table_full_name": ref.table_full_name,
            "column_name": ref.column.name,
            "column_type": ref.column.type,
            "primitive_type": primitive_type,
            "description": ref.column.description,
            "sample_values": list(ref.column.sample_values),
            "is_key_like": _is_key_like(ref.column),
            "is_time_like": _is_time_like(ref.column),
            "normalized_column_name": _normalize_identifier(ref.column.name),
            "provenance": {"source": "load_db_index", "table_key": ref.table_key},
        },
    )


def _column_group_objects(
    table_key: str,
    table: TableSchema,
    table_full_name: str,
) -> list[SchemaObject]:
    """Create deterministic structural column groups for one table."""

    groups: list[tuple[str, str, list[ColumnSchema]]] = []
    key_columns = [column for column in table.columns if _is_key_like(column)]
    time_columns = [column for column in table.columns if _is_time_like(column)]
    measure_columns = [
        column
        for column in table.columns
        if _is_numeric_measure_candidate(column)
        and not _is_key_like(column)
        and not _is_time_like(column)
    ]
    if key_columns:
        groups.append(("key_like", "Key-like columns", key_columns))
    if time_columns:
        groups.append(("time_like", "Time-like columns", time_columns))
    if measure_columns:
        groups.append(("numeric_measures", "Numeric measure candidates", measure_columns))
    groups.extend(_repeated_prefix_groups(table.columns))

    objects: list[SchemaObject] = []
    for group_slug, group_name, columns in groups:
        column_names = sorted({column.name for column in columns})
        if not column_names:
            continue
        group_hash = _stable_hash(table_full_name, group_slug, *column_names)
        objects.append(
            SchemaObject(
                object_id=f"column_group:{table_full_name}#{group_slug}:{group_hash}",
                object_type="column_group",
                name=f"{table_full_name}.{group_name}",
                db=table.database_name,
                table_name=table_full_name,
                searchable_text=_join_text([table_full_name, group_name, " ".join(column_names)]),
                metadata={
                    "source": "load_db_index",
                    "source_table_key": table_key,
                    "database_name": table.database_name,
                    "schema_name": table.schema_name,
                    "table_full_name": table_full_name,
                    "group_type": group_slug,
                    "columns": column_names,
                    "provenance": {"source": "load_db_index", "table_key": table_key},
                },
            )
        )
    return objects


def _repeated_prefix_groups(
    columns: Iterable[ColumnSchema],
) -> list[tuple[str, str, list[ColumnSchema]]]:
    """Find repeated-prefix groups using column-name structure only."""

    candidates: dict[str, list[ColumnSchema]] = defaultdict(list)
    for column in columns:
        tokens = _tokens(column.name)
        for prefix_length in range(1, min(len(tokens), 3)):
            prefix = "_".join(tokens[:prefix_length])
            if prefix:
                candidates[prefix].append(column)

    groups: list[tuple[str, str, list[ColumnSchema]]] = []
    used_signatures: set[tuple[str, ...]] = set()
    for prefix in sorted(candidates):
        members = sorted(
            {column.name: column for column in candidates[prefix]}.values(),
            key=lambda column: column.name,
        )
        if len(members) < 3:
            continue
        signature = tuple(column.name for column in members)
        if signature in used_signatures:
            continue
        used_signatures.add(signature)
        groups.append((_slug(f"prefix_{prefix}"), f"Repeated prefix {prefix}", members))
    return groups


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
    return _SampleValueDecision(
        include=False,
        values=[],
        sample_size=sample_size,
        distinct_count=distinct_count,
        reason=reason,
    )
