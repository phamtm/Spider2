"""Render planner and search text directly from schema objects."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from sol01.models import SchemaObject, is_schema_object_id
from sol01.schema.constants import MAX_FAMILY_MEMBERS_IN_PROMPT
from sol01.schema.large_schema_summaries import (
    LargeSchemaSummary,
    LargeSchemaSummaryRegistry,
    load_large_schema_summary_registry,
)
from sol01.schema.summary_rendering import (
    render_summary_payload,
    render_summary_search_terms,
    render_summary_text,
)
from sol01.schema.utils import _metadata_text, _string_list

MAX_COLUMNS_IN_PROMPT = 30
MAX_SAMPLE_LITERAL_CHARS = 80

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def annotate_summary_metadata(
    schema_objects: Iterable[SchemaObject],
    *,
    large_schema_summary_registry: LargeSchemaSummaryRegistry | None = None,
) -> list[SchemaObject]:
    """Attach summary metadata to table and family objects when relevant."""

    registry = large_schema_summary_registry or load_large_schema_summary_registry()
    annotated: list[SchemaObject] = []
    for obj in schema_objects:
        summaries = large_schema_summaries_for_object(obj, registry=registry)
        if not summaries:
            annotated.append(obj)
            continue
        annotated.append(
            obj.model_copy(
                update={
                    "metadata": {
                        **obj.metadata,
                        **summary_metadata(summaries),
                    }
                }
            )
        )
    return annotated


def build_object_search_text(obj: SchemaObject) -> str:
    """Return compact search text for one schema object."""

    summaries = summary_payloads(obj)
    if obj.object_type == "table":
        return _table_search_text(obj, summaries=summaries)
    if obj.object_type == "column":
        return _join_text(_identifier_terms(obj, extra=_sample_values(obj)))
    if obj.object_type == "column_group":
        columns = _string_list(obj.metadata.get("columns"))
        return _join_text(_identifier_terms(obj, extra=[*columns, *_normalized_tokens(columns)]))
    if obj.object_type == "join_candidate":
        left = _mapping(obj.metadata.get("left"))
        right = _mapping(obj.metadata.get("right"))
        return _join_text(
            _identifier_terms(
                obj,
                extra=[
                    _side_ref(left),
                    _side_ref(right),
                    _metadata_text(obj, "normalized_column_name"),
                    _metadata_text(obj, "primitive_type"),
                ],
            )
        )
    if obj.object_type == "sample_value":
        literal = _safe_literal(obj.metadata.get("value", obj.name))
        return _join_text(
            [
                obj.object_id,
                obj.db,
                obj.table_name,
                obj.column_name,
                _metadata_text(obj, "table_full_name"),
                literal,
                _metadata_text(obj, "inclusion_reason"),
            ]
        )
    return _family_search_text(obj, summaries=summaries)


def build_object_planning_text(obj: SchemaObject) -> str:
    """Return planner-visible text for one schema object."""

    summaries = summary_payloads(obj)
    if obj.object_type == "table":
        return _table_planning_text(obj, summaries=summaries)
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
    return _family_planning_text(obj, summaries=summaries)


def object_parent_ids(obj: SchemaObject) -> list[str]:
    """Return related parent object ids used for score propagation."""

    if obj.object_type == "column":
        return _table_parent_ids(obj)
    if obj.object_type == "column_group":
        columns = _string_list(obj.metadata.get("columns"))
        return [*_table_parent_ids(obj), *_column_parent_ids(obj.table_name, columns)]
    if obj.object_type == "join_candidate":
        return _join_parent_ids(
            _mapping(obj.metadata.get("left")),
            _mapping(obj.metadata.get("right")),
        )
    if obj.object_type == "sample_value":
        return [
            *_table_parent_ids(obj),
            *_column_parent_ids(obj.table_name, [obj.column_name]),
        ]
    if obj.object_type == "family":
        return _table_parent_ids_from_names(_string_list(obj.metadata.get("member_table_refs")))
    return []


def object_has_large_schema_summary(obj: SchemaObject) -> bool:
    return bool(summary_payloads(obj))


def covered_table_ids_for_summary_object(obj: SchemaObject) -> set[str]:
    """Return physical table ids represented by one summary-backed object."""

    if not object_has_large_schema_summary(obj):
        return set()
    if obj.object_id.startswith("table:"):
        return {obj.object_id}
    if obj.object_type == "family":
        return {
            f"table:{member_ref}"
            for member_ref in _string_list(obj.metadata.get("member_table_refs"))
            if member_ref
        }
    return set()


def summary_payloads(obj: SchemaObject) -> list[dict[str, object]]:
    raw = obj.metadata.get("large_schema_summaries")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def summary_metadata(summaries: list[LargeSchemaSummary]) -> dict[str, object]:
    return {
        "large_schema_summaries": [
            {
                **render_summary_payload(summary),
                "text": render_summary_text([summary]),
                "search_terms": render_summary_search_terms([summary]),
            }
            for summary in summaries
        ],
        "summary_ids": [summary.summary_id for summary in summaries],
        "summary_aliases": _stable_unique(
            alias for summary in summaries for alias in summary.aliases
        ),
    }


def large_schema_summaries_for_object(
    obj: SchemaObject,
    *,
    registry: LargeSchemaSummaryRegistry | None = None,
) -> list[LargeSchemaSummary]:
    if obj.object_type == "table":
        refs = _table_ref_candidates(obj)
    elif obj.object_type == "family":
        refs = _string_list(obj.metadata.get("member_table_refs"))
        canonical = _metadata_text(obj, "canonical_member")
        if canonical:
            refs = [canonical, *refs]
    else:
        return []
    return large_schema_summaries_for_refs(refs, registry=registry)


def large_schema_summaries_for_refs(
    refs: Iterable[str],
    *,
    registry: LargeSchemaSummaryRegistry | None = None,
) -> list[LargeSchemaSummary]:
    registry = registry or load_large_schema_summary_registry()
    by_id: dict[str, LargeSchemaSummary] = {}
    for ref in refs:
        if ref.count(".") not in {1, 2}:
            continue
        for summary in registry.match_table_ref(ref):
            by_id.setdefault(summary.summary_id, summary)
    return [by_id[summary_id] for summary_id in sorted(by_id)]


def render_summary_text_from_payloads(payloads: list[dict[str, object]]) -> str:
    if not payloads:
        return ""
    parts = [str(payload.get("text") or "").strip() for payload in payloads]
    return _join_sentences(part for part in parts if part)


def render_summary_search_terms_from_payloads(payloads: list[dict[str, object]]) -> list[str]:
    terms: list[str] = []
    for payload in payloads:
        raw_terms = payload.get("search_terms")
        if isinstance(raw_terms, list):
            terms.extend(str(term).strip() for term in raw_terms if str(term).strip())
    return _stable_unique(terms)


def _table_search_text(obj: SchemaObject, *, summaries: list[dict[str, object]]) -> str:
    if summaries:
        table_ref = obj.table_name or _metadata_text(obj, "full_name") or obj.name
        return _join_text(
            [
                *_identifier_terms(obj),
                table_ref,
                *render_summary_search_terms_from_payloads(summaries),
            ]
        )
    columns = _column_summaries(obj.metadata.get("columns"))
    column_names = [column["name"] for column in columns]
    return _join_text(
        _identifier_terms(obj, extra=[*_column_names(columns), *_normalized_tokens(column_names)])
    )


def _table_planning_text(obj: SchemaObject, *, summaries: list[dict[str, object]]) -> str:
    table_ref = obj.table_name or obj.name
    if summaries:
        return _join_sentences(
            [
                f"Table {table_ref}.",
                render_summary_text_from_payloads(summaries),
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


def _family_search_text(obj: SchemaObject, *, summaries: list[dict[str, object]]) -> str:
    common_columns = _string_list(obj.metadata.get("common_columns"))
    canonical = _metadata_text(obj, "canonical_member")
    if summaries:
        return _join_text(
            [
                *_identifier_terms(obj, extra=[canonical, *common_columns]),
                *render_summary_search_terms_from_payloads(summaries),
            ]
        )
    members = _string_list(obj.metadata.get("member_table_refs"))
    return _join_text(
        _identifier_terms(obj, extra=[*members, *common_columns, *_normalized_tokens(members)])
    )


def _family_planning_text(obj: SchemaObject, *, summaries: list[dict[str, object]]) -> str:
    common_columns = _string_list(obj.metadata.get("common_columns"))
    canonical = _metadata_text(obj, "canonical_member")
    if summaries:
        return _join_sentences(
            [
                f"Table family {obj.name}.",
                f"Canonical member: {canonical}." if canonical else "",
                _field_list("Common columns", common_columns),
                render_summary_text_from_payloads(summaries),
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


def _column_names(columns: list[dict[str, str]]) -> list[str]:
    return [column["name"] for column in columns]


def _sample_values(obj: SchemaObject) -> list[str]:
    return [_safe_literal(value) for value in _string_list(obj.metadata.get("sample_values"))]


def _identifier_terms(obj: SchemaObject, *, extra: Iterable[str] = ()) -> list[str]:
    terms = [
        obj.object_id,
        obj.name,
        obj.db,
        obj.table_name,
        obj.column_name,
        obj.searchable_text,
        _metadata_text(obj, "full_name"),
        _metadata_text(obj, "table_full_name"),
        _metadata_text(obj, "short_name"),
        _metadata_text(obj, "normalized_table_name"),
        _metadata_text(obj, "normalized_column_name"),
        *_string_list(obj.metadata.get("name_tokens")),
        *extra,
    ]
    return [term for term in terms if term]


def _table_ref_candidates(obj: SchemaObject) -> list[str]:
    candidates = [
        obj.table_name,
        _metadata_text(obj, "full_name"),
        _metadata_text(obj, "table_full_name"),
    ]
    database = _metadata_text(obj, "database_name")
    schema_name = _metadata_text(obj, "schema_name")
    short_name = _metadata_text(obj, "short_name")
    if schema_name and short_name:
        candidates.append(
            f"{database}.{schema_name}.{short_name}" if database else f"{schema_name}.{short_name}"
        )
    return _stable_unique([candidate for candidate in candidates if candidate])


def _table_parent_ids(obj: SchemaObject) -> list[str]:
    return _table_parent_ids_from_names([obj.table_name or _metadata_text(obj, "table_full_name")])


def _table_parent_ids_from_names(table_names: Iterable[str | None]) -> list[str]:
    return _valid_object_ids(f"table:{table_name}" for table_name in table_names if table_name)


def _column_parent_ids(table_name: str | None, column_names: Iterable[str | None]) -> list[str]:
    if not table_name:
        return []
    return _valid_object_ids(
        f"column:{table_name}#{column_name}" for column_name in column_names if column_name
    )


def _join_parent_ids(left: dict[str, object], right: dict[str, object]) -> list[str]:
    ids: list[str] = []
    for side in (left, right):
        table_name = str(side.get("table_full_name") or "").strip()
        column_name = str(side.get("column_name") or "").strip()
        ids.extend(_table_parent_ids_from_names([table_name]))
        ids.extend(_column_parent_ids(table_name, [column_name]))
    return _stable_unique(ids)


def _valid_object_ids(object_ids: Iterable[str]) -> list[str]:
    return [object_id for object_id in _stable_unique(object_ids) if is_schema_object_id(object_id)]


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


def _normalized_tokens(values: Iterable[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        tokens.extend(token.casefold() for token in _TOKEN_RE.findall(value))
    return _stable_unique(tokens)


def _safe_literal(value: object) -> str:
    text = str(value).strip().replace("\n", " ").replace("\r", " ")
    if len(text) <= MAX_SAMPLE_LITERAL_CHARS:
        return repr(text)
    return repr(f"{text[: MAX_SAMPLE_LITERAL_CHARS - 3]}...")


def _field_list(label: str, values: list[str]) -> str:
    if not values:
        return ""
    return f"{label}: {', '.join(values)}."


def _join_text(parts: Iterable[Any]) -> str:
    return " ".join(str(part).strip() for part in parts if str(part).strip())


def _join_sentences(parts: Iterable[str]) -> str:
    return " ".join(part.strip() for part in parts if part.strip())


def _bounded_list(values: list[str], limit: int) -> list[str]:
    if len(values) <= limit:
        return values
    return [*values[:limit], f"... {len(values) - limit} more"]


def _stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
