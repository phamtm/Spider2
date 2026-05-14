"""Build planner-visible schema context from deterministic metadata."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from sol01.models import SchemaContextObject, SchemaObject
from sol01.schema.object_text import build_object_planning_text
from sol01.schema.schema_context_cache import SchemaContextCache

_OBJECT_TYPE_ORDER = {
    "family": 0,
    "table": 1,
    "column_group": 2,
    "column": 3,
    "join_candidate": 4,
    "sample_value": 5,
}


def build_available_schema_context(
    cache: SchemaContextCache,
) -> tuple[list[SchemaContextObject], dict[str, object]]:
    """Return planner-visible schema objects with no question-time ranking.

    The cache already contains only planner-visible objects — mode was decided
    at build time and only the appropriate object families were stored.
    """
    context_mode = str(cache.manifest.get("context_mode") or "full_metadata")
    objects = _sort_schema_objects(cache.objects)
    schema_context_objects = _schema_context_objects(objects)
    diagnostics = {
        "context_mode": context_mode,
        "context_counts": {
            "objects_total": len(cache.objects),
            "available_objects": len(cache.objects),
        },
    }
    return schema_context_objects, diagnostics


def _schema_context_objects(
    objects: Sequence[SchemaObject],
) -> list[SchemaContextObject]:
    return [
        SchemaContextObject(
            schema_object=schema_object,
            planning_text=build_object_planning_text(schema_object),
            position=rank,
        )
        for rank, schema_object in enumerate(objects, start=1)
    ]


def _sort_schema_objects(objects: Iterable[SchemaObject]) -> list[SchemaObject]:
    return sorted(
        objects,
        key=lambda obj: (
            _OBJECT_TYPE_ORDER.get(obj.object_type, 99),
            obj.table_name or "",
            obj.column_name or "",
            obj.object_id,
        ),
    )
