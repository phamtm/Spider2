"""Join-candidate schema objects inferred from column evidence."""

from __future__ import annotations

import itertools
from collections import defaultdict

from sol01.models import SchemaObject

from ._object_shared import (
    PrimitiveType,
    _ColumnRef,
    _join_text,
    _normalize_identifier,
    _primitive_type,
    _stable_hash,
)


def join_candidate_objects(
    column_refs: list[_ColumnRef],
    *,
    max_join_candidates: int,
    max_join_candidates_per_column: int,
) -> list[SchemaObject]:
    """Infer same-name compatible-type join evidence across tables."""

    grouped_refs: dict[
        tuple[str | None, str | None, str, PrimitiveType],
        list[_ColumnRef],
    ] = defaultdict(list)
    for ref in column_refs:
        primitive_type = _primitive_type(ref.column.type)
        if primitive_type in {"semi_structured", "unknown"}:
            continue
        key = (
            ref.database_name,
            ref.schema_name,
            _normalize_identifier(ref.column.name),
            primitive_type,
        )
        grouped_refs[key].append(ref)

    objects: list[SchemaObject] = []
    for (database_name, schema_name, normalized_name, primitive_type), refs in sorted(
        grouped_refs.items()
    ):
        unique_refs = sorted(refs, key=lambda ref: (ref.table_full_name, ref.column.name))
        if len({ref.table_full_name for ref in unique_refs}) < 2:
            continue

        emitted_for_group = 0
        for left, right in itertools.combinations(unique_refs, 2):
            if left.table_full_name == right.table_full_name:
                continue
            objects.append(
                SchemaObject(
                    object_id=_join_candidate_id(left, right),
                    object_type="join_candidate",
                    name=(
                        f"{left.table_full_name}.{left.column.name} -> "
                        f"{right.table_full_name}.{right.column.name}"
                    ),
                    db=database_name,
                    searchable_text=_join_text(
                        [
                            "join candidate",
                            normalized_name,
                            left.table_full_name,
                            left.column.name,
                            right.table_full_name,
                            right.column.name,
                        ]
                    ),
                    metadata={
                        "source": "load_db_index",
                        "evidence_type": "same_normalized_column_name_and_compatible_type",
                        "is_inferred_evidence": True,
                        "is_foreign_key": False,
                        "database_name": database_name,
                        "schema_name": schema_name,
                        "normalized_column_name": normalized_name,
                        "primitive_type": primitive_type,
                        "left": _join_side_metadata(left),
                        "right": _join_side_metadata(right),
                        "provenance": {"source": "load_db_index"},
                    },
                )
            )
            emitted_for_group += 1
            if emitted_for_group >= max_join_candidates_per_column:
                break
            if len(objects) >= max_join_candidates:
                return objects
    return objects


def _join_side_metadata(ref: _ColumnRef) -> dict[str, object]:
    return {
        "source_table_key": ref.table_key,
        "table_full_name": ref.table_full_name,
        "column_name": ref.column.name,
        "column_type": ref.column.type,
        "primitive_type": _primitive_type(ref.column.type),
    }


def _join_candidate_id(left: _ColumnRef, right: _ColumnRef) -> str:
    join_hash = _stable_hash(
        left.table_full_name,
        left.column.name,
        right.table_full_name,
        right.column.name,
        "join_candidate",
    )
    return (
        f"join_candidate:{left.table_full_name}#{left.column.name}->"
        f"{right.table_full_name}#{right.column.name}:{join_hash}"
    )
