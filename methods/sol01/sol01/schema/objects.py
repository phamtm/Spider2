"""Build canonical schema objects from table metadata."""

from __future__ import annotations

from collections.abc import Collection, Mapping

from sol01.infra.policy import DEFAULT_SCHEMA_CONTEXT_POLICY
from sol01.models import SchemaObject, TableSchema

from ._object_basics import (
    _column_group_objects,
    _column_object,
    _column_refs,
    _table_object,
    sample_value_objects,
)
from ._object_families import table_family_objects
from ._object_joins import join_candidate_objects
from ._object_shared import (
    DEFAULT_MAX_JOIN_CANDIDATES,
    DEFAULT_MAX_JOIN_CANDIDATES_PER_COLUMN,
    DEFAULT_MAX_SAMPLE_VALUES_PER_COLUMN,
    _table_full_name,
)


def build_schema_objects(
    db_index: Mapping[str, TableSchema],
    *,
    max_sample_values_per_column: int = DEFAULT_MAX_SAMPLE_VALUES_PER_COLUMN,
    max_join_candidates: int = DEFAULT_MAX_JOIN_CANDIDATES,
    max_join_candidates_per_column: int = DEFAULT_MAX_JOIN_CANDIDATES_PER_COLUMN,
    family_similarity_threshold: float = DEFAULT_SCHEMA_CONTEXT_POLICY.family_similarity_threshold,
    compact_only: bool = False,
    compact_table_keys: Collection[str] = (),
) -> list[SchemaObject]:
    """Build deterministic schema objects from a database table index.

    When *compact_only* is True, only table and family objects are emitted.
    This keeps planner-time discovery compact while later stages can still
    expand selected tables into exact metadata.
    """
    objects: list[SchemaObject] = []

    compact_table_key_set = {
        table_key.strip() for table_key in compact_table_keys if table_key.strip()
    }

    if compact_only:
        for table_key in sorted(db_index):
            table = db_index[table_key]
            objects.append(_table_object(table_key, table, _table_full_name(table_key, table)))
        objects.extend(
            table_family_objects(
                db_index,
                family_similarity_threshold=family_similarity_threshold,
            )
        )
        return objects

    column_refs = _column_refs(db_index)
    for table_key in sorted(db_index):
        table = db_index[table_key]
        table_full_name = _table_full_name(table_key, table)
        objects.append(_table_object(table_key, table, table_full_name))
        if table_key in compact_table_key_set:
            continue
        objects.extend(_column_object(ref) for ref in column_refs if ref.table_key == table_key)
        objects.extend(_column_group_objects(table_key, table, table_full_name))
        objects.extend(
            sample_value_objects(
                table_key,
                table,
                table_full_name,
                max_sample_values_per_column=max_sample_values_per_column,
            )
        )

    objects.extend(
        table_family_objects(
            db_index,
            family_similarity_threshold=family_similarity_threshold,
        )
    )
    objects.extend(
        join_candidate_objects(
            [ref for ref in column_refs if ref.table_key not in compact_table_key_set],
            max_join_candidates=max_join_candidates,
            max_join_candidates_per_column=max_join_candidates_per_column,
        )
    )
    return objects
