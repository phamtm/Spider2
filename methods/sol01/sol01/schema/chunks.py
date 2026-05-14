"""Render deterministic schema-context chunks from canonical schema objects."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from sol01.models import SchemaContextChunk, SchemaObject, is_schema_object_id
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

MAX_COLUMNS_IN_CHUNK = 30
MAX_SAMPLE_LITERAL_CHARS = 80

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def render_schema_chunks(
    schema_objects: Iterable[SchemaObject],
    *,
    large_schema_summary_registry: LargeSchemaSummaryRegistry | None = None,
) -> list[SchemaContextChunk]:
    """Render one compact schema-context chunk per canonical schema object."""

    return [
        _render_schema_chunk(obj, large_schema_summary_registry=large_schema_summary_registry)
        for obj in sorted(schema_objects, key=lambda item: item.object_id)
    ]


def _render_schema_chunk(
    obj: SchemaObject,
    *,
    large_schema_summary_registry: LargeSchemaSummaryRegistry | None,
) -> SchemaContextChunk:
    if obj.object_type == "table":
        return _table_chunk(obj, large_schema_summary_registry=large_schema_summary_registry)
    if obj.object_type == "column":
        return _column_chunk(obj)
    if obj.object_type == "column_group":
        return _column_group_chunk(obj)
    if obj.object_type == "join_candidate":
        return _join_candidate_chunk(obj)
    if obj.object_type == "sample_value":
        return _sample_value_chunk(obj)
    return _table_family_chunk(
        obj,
        large_schema_summary_registry=large_schema_summary_registry,
    )


def _table_chunk(
    obj: SchemaObject,
    *,
    large_schema_summary_registry: LargeSchemaSummaryRegistry | None,
) -> SchemaContextChunk:
    summaries = _large_schema_summaries_for_obj(
        obj,
        registry=large_schema_summary_registry,
    )
    if summaries:
        return _summary_table_chunk(obj, summaries)

    columns = _column_summaries(obj.metadata.get("columns"))
    column_names = [column["name"] for column in columns]
    displayed_columns = _bounded_list(
        [
            f"{column['name']} [{column['type']}]" if column["type"] else column["name"]
            for column in columns
        ],
        MAX_COLUMNS_IN_CHUNK,
    )
    source_definition = _join_sentences(
        [
            f"Source table {obj.table_name or obj.name}.",
            _field_list("Columns", displayed_columns),
            _description_sentence(obj.description),
        ]
    )
    prompt_text = _join_sentences(
        [
            f"Table {obj.table_name or obj.name}.",
            _field_list("Columns", displayed_columns),
        ]
    )
    search_terms = _identifier_terms(
        obj,
        extra=[*_column_names(columns), *_normalized_tokens(column_names)],
    )
    return SchemaContextChunk(
        chunk_id=f"{obj.object_id}::table",
        object_id=obj.object_id,
        chunk_type="table",
        evidence_text=_join_text(search_terms),
        prompt_text=prompt_text,
        source_definition=source_definition,
        source="schema",
        metadata=_base_metadata(obj),
    )


def _column_chunk(obj: SchemaObject) -> SchemaContextChunk:
    column_type = _metadata_text(obj, "column_type")
    source_definition = _join_sentences(
        [
            f"Source column {obj.table_name}.{obj.column_name}.",
            f"Type: {column_type}." if column_type else "",
            _description_sentence(obj.description),
        ]
    )
    inferred_usage = _join_sentences(
        [
            "Inferred usage: key-like identifier." if obj.metadata.get("is_key_like") else "",
            "Inferred usage: time-like filter or ordering column."
            if obj.metadata.get("is_time_like")
            else "",
        ]
    )
    prompt_text = _join_sentences(
        [
            f"Column {obj.table_name}.{obj.column_name}",
            f"type {column_type}" if column_type else "",
            obj.description or "",
        ]
    )
    return SchemaContextChunk(
        chunk_id=f"{obj.object_id}::column",
        object_id=obj.object_id,
        chunk_type="column",
        parent_object_ids=_table_parent_ids(obj),
        evidence_text=_join_text(_identifier_terms(obj, extra=_sample_values(obj))),
        prompt_text=prompt_text,
        source_definition=source_definition,
        inferred_usage=inferred_usage,
        source="schema",
        metadata=_base_metadata(obj),
    )


def _column_group_chunk(obj: SchemaObject) -> SchemaContextChunk:
    columns = _string_list(obj.metadata.get("columns"))
    group_type = _metadata_text(obj, "group_type")
    source_definition = _join_sentences(
        [
            f"Source table {obj.table_name} column group {group_type}.",
            _field_list("Columns", columns),
        ]
    )
    inferred_usage = f"Inferred structural column group: {group_type}." if group_type else ""
    return SchemaContextChunk(
        chunk_id=f"{obj.object_id}::column_group",
        object_id=obj.object_id,
        chunk_type="column_group",
        parent_object_ids=[*_table_parent_ids(obj), *_column_parent_ids(obj.table_name, columns)],
        evidence_text=_join_text(
            _identifier_terms(obj, extra=[*columns, *_normalized_tokens(columns)])
        ),
        prompt_text=_join_sentences(
            [f"Column group {group_type} on {obj.table_name}.", ", ".join(columns)]
        ),
        source_definition=source_definition,
        inferred_usage=inferred_usage,
        source="schema",
        metadata=_base_metadata(obj),
    )


def _join_candidate_chunk(obj: SchemaObject) -> SchemaContextChunk:
    left = _mapping(obj.metadata.get("left"))
    right = _mapping(obj.metadata.get("right"))
    left_ref = _side_ref(left)
    right_ref = _side_ref(right)
    evidence_type = _metadata_text(obj, "evidence_type")
    source_definition = _join_sentences(
        [
            f"Source columns {left_ref} and {right_ref}.",
            f"Compatible primitive type: {_metadata_text(obj, 'primitive_type')}."
            if obj.metadata.get("primitive_type")
            else "",
        ]
    )
    inferred_usage = _join_sentences(
        [
            f"Inferred join evidence: {evidence_type}." if evidence_type else "",
            "This is not a declared foreign key."
            if obj.metadata.get("is_foreign_key") is False
            else "",
        ]
    )
    return SchemaContextChunk(
        chunk_id=f"{obj.object_id}::join_candidate",
        object_id=obj.object_id,
        chunk_type="join_candidate",
        parent_object_ids=_join_parent_ids(left, right),
        evidence_text=_join_text(
            _identifier_terms(
                obj,
                extra=[
                    left_ref,
                    right_ref,
                    _metadata_text(obj, "normalized_column_name"),
                    _metadata_text(obj, "primitive_type"),
                ],
            )
        ),
        prompt_text=f"Join candidate: {left_ref} = {right_ref}.",
        source_definition=source_definition,
        inferred_usage=inferred_usage,
        source="join",
        metadata=_base_metadata(obj),
    )


def _sample_value_chunk(obj: SchemaObject) -> SchemaContextChunk:
    literal = _safe_literal(obj.metadata.get("value", obj.name))
    source_definition = _join_sentences(
        [
            f"Categorical sample value for {obj.table_name}.{obj.column_name}: {literal}.",
            f"Sample size: {_metadata_text(obj, 'sample_size')}."
            if obj.metadata.get("sample_size") is not None
            else "",
            f"Distinct values: {_metadata_text(obj, 'distinct_count')}."
            if obj.metadata.get("distinct_count") is not None
            else "",
        ]
    )
    inferred_usage = "Use only as exact filter evidence."
    return SchemaContextChunk(
        chunk_id=f"{obj.object_id}::sample_value",
        object_id=obj.object_id,
        chunk_type="sample_value",
        parent_object_ids=[
            *_table_parent_ids(obj),
            *_column_parent_ids(obj.table_name, [obj.column_name]),
        ],
        evidence_text=_join_text(
            [
                obj.object_id,
                obj.db,
                obj.table_name,
                obj.column_name,
                _metadata_text(obj, "table_full_name"),
                literal,
                _metadata_text(obj, "inclusion_reason"),
            ]
        ),
        prompt_text=f"Sample value {obj.table_name}.{obj.column_name} = {literal}.",
        source_definition=source_definition,
        inferred_usage=inferred_usage,
        source="sample",
        metadata=_base_metadata(obj),
    )


def _table_family_chunk(
    obj: SchemaObject,
    *,
    large_schema_summary_registry: LargeSchemaSummaryRegistry | None,
) -> SchemaContextChunk:
    summaries = _large_schema_summaries_for_obj(
        obj,
        registry=large_schema_summary_registry,
    )
    if summaries:
        return _summary_table_family_chunk(obj, summaries)

    members = _string_list(obj.metadata.get("member_table_refs"))
    common_columns = _string_list(obj.metadata.get("common_columns"))
    family_kind = _metadata_text(obj, "family_kind")
    canonical = _metadata_text(obj, "canonical_member")
    suffix_summary = _suffix_dimension_summary(obj.metadata.get("suffix_dimensions"))
    member_preview = _bounded_list(members, MAX_FAMILY_MEMBERS_IN_PROMPT)
    source_definition = _join_sentences(
        [
            f"Table-family record {obj.name}.",
            f"Canonical member: {canonical}." if canonical else "",
            f"Members: {len(members)}." if members else "",
            _field_list("Common columns", common_columns),
            suffix_summary,
        ]
    )
    inferred_usage = _join_sentences(
        [
            f"Inferred table-family structure from {family_kind} detection." if family_kind else "",
            "Near-family caveats apply." if _string_list(obj.metadata.get("caveats")) else "",
        ]
    )
    return SchemaContextChunk(
        chunk_id=f"{obj.object_id}::table_family",
        object_id=obj.object_id,
        chunk_type="table_family",
        parent_object_ids=_table_parent_ids_from_names(members),
        evidence_text=_join_text(
            _identifier_terms(obj, extra=[*members, *common_columns, *_normalized_tokens(members)])
        ),
        prompt_text=_join_sentences(
            [
                f"Table family {obj.name}: canonical={canonical}, members={len(members)}.",
                _field_list("Member preview", member_preview),
                _field_list("Common columns", common_columns),
                suffix_summary,
            ]
        ),
        source_definition=source_definition,
        inferred_usage=inferred_usage,
        source="family",
        metadata=_base_metadata(obj),
    )


def _summary_table_chunk(
    obj: SchemaObject,
    summaries: list[LargeSchemaSummary],
) -> SchemaContextChunk:
    table_ref = obj.table_name or _metadata_text(obj, "full_name") or obj.name
    summary_text = render_summary_text(summaries)
    metadata = {**_base_metadata(obj), **_summary_chunk_metadata(summaries)}
    return SchemaContextChunk(
        chunk_id=f"{obj.object_id}::table",
        object_id=obj.object_id,
        chunk_type="table",
        evidence_text=_join_text(
            [
                *_identifier_terms(obj),
                table_ref,
                *render_summary_search_terms(summaries),
            ]
        ),
        prompt_text=_join_sentences(
            [
                f"Table {table_ref}.",
                summary_text,
            ]
        ),
        source_definition=_join_sentences(
            [
                f"Source table {table_ref}.",
                summary_text,
            ]
        ),
        source="schema",
        metadata=metadata,
    )


def _summary_table_family_chunk(
    obj: SchemaObject,
    summaries: list[LargeSchemaSummary],
) -> SchemaContextChunk:
    members = _string_list(obj.metadata.get("member_table_refs"))
    common_columns = _string_list(obj.metadata.get("common_columns"))
    canonical = _metadata_text(obj, "canonical_member")
    summary_text = render_summary_text(summaries)
    metadata = {**_base_metadata(obj), **_summary_chunk_metadata(summaries)}
    return SchemaContextChunk(
        chunk_id=f"{obj.object_id}::table_family",
        object_id=obj.object_id,
        chunk_type="table_family",
        parent_object_ids=_table_parent_ids_from_names(members),
        evidence_text=_join_text(
            [
                *_identifier_terms(obj, extra=[canonical, *common_columns]),
                *render_summary_search_terms(summaries),
            ]
        ),
        prompt_text=_join_sentences(
            [
                f"Table family {obj.name}.",
                f"Canonical member: {canonical}." if canonical else "",
                _field_list("Common columns", common_columns),
                summary_text,
            ]
        ),
        source_definition=_join_sentences(
            [
                f"Table-family record {obj.name}.",
                f"Canonical member: {canonical}." if canonical else "",
                summary_text,
            ]
        ),
        inferred_usage=_join_sentences(
            [
                "Use curated summary evidence for this large table family.",
                "Expand physical members only when question constraints require it.",
            ]
        ),
        source="family",
        metadata=metadata,
    )


def _base_metadata(obj: SchemaObject) -> dict[str, object]:
    return {
        "object_type": obj.object_type,
        "db": obj.db,
        "table_name": obj.table_name,
        "column_name": obj.column_name,
    }


def _large_schema_summaries_for_obj(
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
    return _large_schema_summaries_for_refs(refs, registry=registry)


def _large_schema_summaries_for_refs(
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


def _summary_chunk_metadata(summaries: list[LargeSchemaSummary]) -> dict[str, object]:
    return {
        "large_schema_summaries": [render_summary_payload(summary) for summary in summaries],
        "summary_ids": [summary.summary_id for summary in summaries],
        "summary_aliases": _stable_unique(
            alias for summary in summaries for alias in summary.aliases
        ),
    }


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
        _metadata_text(obj, "full_name"),
        _metadata_text(obj, "table_full_name"),
        _metadata_text(obj, "short_name"),
        _metadata_text(obj, "normalized_table_name"),
        _metadata_text(obj, "normalized_column_name"),
        *_string_list(obj.metadata.get("name_tokens")),
        *extra,
    ]
    return [term for term in terms if term]


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


def _description_sentence(description: str | None) -> str:
    return f"Description: {description}." if description else ""


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
