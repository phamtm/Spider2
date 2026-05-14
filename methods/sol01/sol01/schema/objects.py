"""Build canonical schema objects from table metadata."""

from __future__ import annotations

from collections.abc import Mapping

from sol01.infra.config import DEFAULT_FAMILY_SIMILARITY_THRESHOLD
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
    family_similarity_threshold: float = DEFAULT_FAMILY_SIMILARITY_THRESHOLD,
    covered_table_keys: set[str] | None = None,
) -> list[SchemaObject]:
    """Build deterministic schema objects from a database table index.

    When *covered_table_keys* is provided, tables whose key appears in that
    set will only emit a table object. Column, column-group, and sample-value
    objects are skipped because a curated large-schema summary replaces their
    raw metadata at chunk-render time.
    """
    _covered = covered_table_keys or set()
    column_refs = _column_refs(db_index)
    objects: list[SchemaObject] = []

    for table_key in sorted(db_index):
        table = db_index[table_key]
        table_full_name = _table_full_name(table_key, table)
        objects.append(_table_object(table_key, table, table_full_name))
        if table_key in _covered:
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
            column_refs,
            max_join_candidates=max_join_candidates,
            max_join_candidates_per_column=max_join_candidates_per_column,
        )
    )
    return objects
