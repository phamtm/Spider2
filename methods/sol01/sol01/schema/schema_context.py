"""Build planner-visible schema context from deterministic metadata."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from sol01.models import SchemaContextObject, SchemaObject
from sol01.schema.object_text import build_object_planning_text, object_has_large_schema_summary
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
    """Return planner-visible schema objects with no question-time ranking."""

    context_mode, context_objects = _available_schema_objects(cache.objects)
    schema_context_objects = _schema_context_objects(context_objects)
    diagnostics = _diagnostics(
        context_mode=context_mode,
        object_count=len(cache.objects),
        context_object_count=len(context_objects),
    )
    return schema_context_objects, diagnostics


def _available_schema_objects(
    objects: Sequence[SchemaObject],
) -> tuple[str, list[SchemaObject]]:
    """Return planner input objects for summary-only or full-metadata mode."""

    ordered = _sort_schema_objects(objects)
    summary_objects = [obj for obj in ordered if object_has_large_schema_summary(obj)]
    if summary_objects:
        return "summary_only", summary_objects
    return "full_metadata", ordered


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


def _diagnostics(
    *,
    context_mode: str,
    object_count: int,
    context_object_count: int,
) -> dict[str, object]:
    return {
        "context_mode": context_mode,
        "context_counts": {
            "objects_total": object_count,
            "available_objects": context_object_count,
        },
    }


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
