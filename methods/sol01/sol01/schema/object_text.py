"""Render planner text directly from schema objects."""

from __future__ import annotations

from collections.abc import Iterable

from sol01.infra.policy import DEFAULT_SCHEMA_RENDER_POLICY
from sol01.models import SchemaObject, SchemaProfileCatalog
from sol01.schema.schema_profile_rendering import (
    render_schema_profile_payload,
    render_schema_profile_search_terms,
    render_schema_profile_text,
)
from sol01.schema.schema_profiles import schema_profiles_for_object
from sol01.schema.utils import _metadata_text, _string_list

MAX_FAMILY_MEMBERS_IN_PROMPT = DEFAULT_SCHEMA_RENDER_POLICY.family_members_in_prompt
MAX_COLUMNS_IN_PROMPT = DEFAULT_SCHEMA_RENDER_POLICY.table_columns_in_prompt
MAX_SAMPLE_LITERAL_CHARS = DEFAULT_SCHEMA_RENDER_POLICY.sample_literal_chars


def annotate_schema_profile_metadata(
    schema_objects: Iterable[SchemaObject],
    *,
    schema_profile_catalog: SchemaProfileCatalog | None = None,
) -> list[SchemaObject]:
    """Attach generated schema-profile metadata to table and family objects when relevant."""

    catalog = schema_profile_catalog
    annotated: list[SchemaObject] = []
    for obj in schema_objects:
        profiles = schema_profiles_for_object(obj, catalog=catalog)
        if not profiles:
            annotated.append(obj)
            continue
        annotated.append(
            obj.model_copy(
                update={
                    "metadata": {
                        **obj.metadata,
                        **schema_profile_metadata(profiles),
                    }
                }
            )
        )
    return annotated


def build_object_planning_text(obj: SchemaObject) -> str:
    """Return planner-visible text for one schema object."""

    profiles = schema_profile_payloads(obj)
    if obj.object_type == "table":
        return _table_planning_text(obj, profiles=profiles)
    if obj.object_type == "column":
        column_type = _metadata_text(obj, "column_type")
        return _join_sentences(
            [
                f"Column {obj.table_name}.{obj.column_name}",
                f"type {column_type}" if column_type else "",
                obj.description or "",
            ]
        )
    if obj.object_type == "column_group":
        columns = _string_list(obj.metadata.get("columns"))
        group_type = _metadata_text(obj, "group_type")
        return _join_sentences(
            [f"Column group {group_type} on {obj.table_name}.", ", ".join(columns)]
        )
    if obj.object_type == "join_candidate":
        left = _mapping(obj.metadata.get("left"))
        right = _mapping(obj.metadata.get("right"))
        return f"Join candidate: {_side_ref(left)} = {_side_ref(right)}."
    if obj.object_type == "sample_value":
        literal = _safe_literal(obj.metadata.get("value", obj.name))
        return f"Sample value {obj.table_name}.{obj.column_name} = {literal}."
    return _family_planning_text(obj, profiles=profiles)


def object_has_schema_profile(obj: SchemaObject) -> bool:
    return bool(schema_profile_payloads(obj))


def schema_profile_payloads(obj: SchemaObject) -> list[dict[str, object]]:
    raw = obj.metadata.get("schema_profiles")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def schema_profile_metadata(profiles: list[object]) -> dict[str, object]:
    return {
        "schema_profiles": [
            {
                **render_schema_profile_payload(profile),
                "text": render_schema_profile_text([profile]),
                "search_terms": render_schema_profile_search_terms([profile]),
            }
            for profile in profiles
        ],
        "schema_profile_ids": [str(profile.profile_id) for profile in profiles],
        "schema_profile_aliases": _stable_unique(
            alias for profile in profiles for alias in getattr(profile, "aliases", [])
        ),
    }


def render_schema_profile_text_from_payloads(payloads: list[dict[str, object]]) -> str:
    if not payloads:
        return ""
    parts = [str(payload.get("text") or "").strip() for payload in payloads]
    return _join_sentences(part for part in parts if part)


def _table_planning_text(obj: SchemaObject, *, profiles: list[dict[str, object]]) -> str:
    table_ref = obj.table_name or obj.name
    if profiles:
        return _join_sentences(
            [
                f"Table {table_ref}.",
                render_schema_profile_text_from_payloads(profiles),
            ]
        )
    columns = _column_summaries(obj.metadata.get("columns"))
    displayed_columns = _bounded_list(
        [
            f"{column['name']} [{column['type']}]" if column["type"] else column["name"]
            for column in columns
        ],
        MAX_COLUMNS_IN_PROMPT,
    )
    return _join_sentences(
        [
            f"Table {table_ref}.",
            _field_list("Columns", displayed_columns),
        ]
    )


def _family_planning_text(obj: SchemaObject, *, profiles: list[dict[str, object]]) -> str:
    common_columns = _string_list(obj.metadata.get("common_columns"))
    canonical = _metadata_text(obj, "canonical_member")
    if profiles:
        return _join_sentences(
            [
                f"Table family {obj.name}.",
                f"Canonical member: {canonical}." if canonical else "",
                _field_list("Common columns", common_columns),
                render_schema_profile_text_from_payloads(profiles),
            ]
        )
    members = _string_list(obj.metadata.get("member_table_refs"))
    member_preview = _bounded_list(members, MAX_FAMILY_MEMBERS_IN_PROMPT)
    return _join_sentences(
        [
            f"Table family {obj.name}: canonical={canonical}, members={len(members)}.",
            _field_list("Member preview", member_preview),
            _field_list("Common columns", common_columns),
            _suffix_dimension_summary(obj.metadata.get("suffix_dimensions")),
        ]
    )


def _column_summaries(raw_columns: object) -> list[dict[str, str]]:
    if not isinstance(raw_columns, list):
        return []
    summaries: list[dict[str, str]] = []
    for raw_column in raw_columns:
        if not isinstance(raw_column, dict):
            continue
        name = str(raw_column.get("name") or "").strip()
        if not name:
            continue
        summaries.append(
            {
                "name": name,
                "type": str(raw_column.get("type") or "").strip(),
                "description": str(raw_column.get("description") or "").strip(),
            }
        )
    return summaries


def _side_ref(side: dict[str, object]) -> str:
    table_name = str(side.get("table_full_name") or "").strip()
    column_name = str(side.get("column_name") or "").strip()
    return f"{table_name}.{column_name}" if table_name and column_name else ""


def _suffix_dimension_summary(raw_dimensions: object) -> str:
    if not isinstance(raw_dimensions, list):
        return ""
    parts: list[str] = []
    for raw_dimension in raw_dimensions:
        if not isinstance(raw_dimension, dict):
            continue
        kind = str(raw_dimension.get("kind") or "").strip()
        values = _string_list(raw_dimension.get("values"))
        if kind and values:
            parts.append(f"{kind} values {', '.join(values[:8])}")
    return f"Partition dimensions: {'; '.join(parts)}." if parts else ""


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _safe_literal(value: object) -> str:
    text = " ".join(str(value).split())
    if len(text) <= MAX_SAMPLE_LITERAL_CHARS:
        return repr(text)
    return repr(text[: MAX_SAMPLE_LITERAL_CHARS - 3] + "...")


def _field_list(label: str, values: list[str]) -> str:
    if not values:
        return ""
    return f"{label}: {', '.join(values)}."


def _bounded_list(values: Iterable[str], limit: int) -> list[str]:
    bounded = [value for value in values if value]
    if len(bounded) <= limit:
        return bounded
    hidden = len(bounded) - limit
    return [*bounded[:limit], f"... {hidden} more"]


def _join_sentences(parts: Iterable[str]) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


def _stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip().casefold()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(value.strip())
    return ordered
