"""Build planner-visible schema context from available metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from sol01.infra.config import SchemaContextConfig
from sol01.models import (
    SchemaContextChunk,
    SchemaContextObject,
    SchemaObject,
)
from sol01.schema.schema_context_cache import SchemaContextCache

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_DOTTED_OR_UNDERSCORE_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[._][A-Za-z0-9]+)+\b")
_UPPER_CODE_RE = re.compile(r"\b[A-Z][A-Z0-9_]{1,}\b")
_DATE_RE = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_QUOTED_LITERAL_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"")

_OBJECT_TYPE_ORDER = {
    "family": 0,
    "table": 1,
    "column_group": 2,
    "column": 3,
    "join_candidate": 4,
    "sample_value": 5,
}


@dataclass(frozen=True)
class SchemaContextInputs:
    """Question-derived context kept for diagnostics and linked-document clipping."""

    text: str
    question: str
    linked_doc_context: str
    exact_literals: tuple[str, ...]
    dates: tuple[str, ...]
    years: tuple[str, ...]
    identifiers: tuple[str, ...]
    uppercase_codes: tuple[str, ...]
    normalized_tokens: tuple[str, ...]


def build_schema_context_inputs(
    question: str,
    *,
    linked_docs: Sequence[str] = (),
    exact_literals: Sequence[str] = (),
    max_doc_chars: int = 6000,
) -> SchemaContextInputs:
    """Build normalized question context without ranking schema objects."""

    clean_question = _clean_text(question)
    literals = _stable_unique(
        [
            *_quoted_literals(clean_question),
            *(_clean_text(value) for value in exact_literals),
        ]
    )
    dates = _stable_unique(_DATE_RE.findall(clean_question))
    years = _stable_unique(_YEAR_RE.findall(clean_question))
    identifiers = _stable_unique(_DOTTED_OR_UNDERSCORE_RE.findall(clean_question))
    uppercase_codes = _stable_unique(_UPPER_CODE_RE.findall(clean_question))
    normalized_tokens = _normalized_tokens(
        [clean_question, *literals, *dates, *years, *identifiers, *uppercase_codes]
    )
    linked_doc_context = clip_linked_docs(
        linked_docs,
        query_terms=normalized_tokens,
        max_doc_chars=max_doc_chars,
    )
    parts = [
        clean_question,
        linked_doc_context,
        " ".join(literals),
        " ".join(dates),
        " ".join(identifiers),
        " ".join(uppercase_codes),
        " ".join(normalized_tokens),
    ]
    return SchemaContextInputs(
        text=_join_text(parts),
        question=clean_question,
        linked_doc_context=linked_doc_context,
        exact_literals=tuple(literals),
        dates=tuple(dates),
        years=tuple(years),
        identifiers=tuple(identifiers),
        uppercase_codes=tuple(uppercase_codes),
        normalized_tokens=tuple(normalized_tokens),
    )


def clip_linked_docs(
    linked_docs: Sequence[str],
    *,
    query_terms: Sequence[str],
    max_doc_chars: int,
) -> str:
    """Clip linked-document context, preferring passages with question overlap."""

    if max_doc_chars < 1 or not linked_docs:
        return ""
    passages = [
        _clean_text(passage)
        for doc in linked_docs
        for passage in re.split(r"\n\s*\n|(?<=[.!?])\s+", doc)
        if _clean_text(passage)
    ]
    if not passages:
        return ""

    remaining = max_doc_chars
    selected: list[str] = []
    terms = set(query_terms)
    if terms:
        ranked = sorted(
            enumerate(passages),
            key=lambda item: (
                -len(terms.intersection(_tokenize(item[1]))),
                item[0],
            ),
        )
    else:
        ranked = list(enumerate(passages))

    for _, passage in ranked:
        if remaining <= 0:
            break
        if not selected and len(passage) > remaining:
            selected.append(passage[:remaining].rstrip())
            break
        if len(passage) + (2 if selected else 0) > remaining:
            continue
        selected.append(passage)
        remaining -= len(passage) + (2 if selected else 0)
    return "\n\n".join(selected)[:max_doc_chars].rstrip()


def build_available_schema_context(
    cache: SchemaContextCache,
    question: str,
    *,
    linked_docs: Sequence[str] = (),
    exact_literals: Sequence[str] = (),
    config: SchemaContextConfig | None = None,
) -> tuple[list[SchemaContextObject], dict[str, object]]:
    """Return schema objects available to the planner.

    Curated large-schema summaries replace raw metadata for the tables they
    cover. Uncovered tables still flow through as normal database metadata.
    """

    config = config or SchemaContextConfig()
    inputs = build_schema_context_inputs(
        question,
        linked_docs=linked_docs,
        exact_literals=exact_literals,
        max_doc_chars=config.max_linked_doc_chars,
    )
    context_mode, context_objects = _available_schema_objects(cache.objects, cache.chunks)
    schema_context_objects = _schema_context_objects(
        context_objects,
        chunks=cache.chunks,
    )
    diagnostics = _diagnostics(
        inputs,
        context_mode=context_mode,
        schema_context_objects=schema_context_objects,
        object_count=len(cache.objects),
        chunk_count=len(cache.chunks),
        context_object_count=len(context_objects),
    )
    return schema_context_objects, diagnostics


def _available_schema_objects(
    objects: Sequence[SchemaObject],
    chunks: Sequence[SchemaContextChunk],
) -> tuple[str, list[SchemaObject]]:
    summary_object_ids = {
        chunk.object_id
        for chunk in chunks
        if chunk.metadata.get("large_schema_summaries") or chunk.metadata.get("summary_ids")
    }
    if summary_object_ids:
        covered_table_ids = _covered_table_ids(chunks, summary_object_ids)
        return (
            "large_schema_summary",
            _sort_schema_objects(
                obj
                for obj in objects
                if obj.object_id in summary_object_ids
                or not _is_covered_by_summary(obj, covered_table_ids)
            ),
        )
    return "full_database_metadata", _sort_schema_objects(objects)


def _covered_table_ids(
    chunks: Sequence[SchemaContextChunk],
    summary_object_ids: set[str],
) -> set[str]:
    table_ids: set[str] = set()
    for chunk in chunks:
        if chunk.object_id not in summary_object_ids:
            continue
        if chunk.object_id.startswith("table:"):
            table_ids.add(chunk.object_id)
        table_ids.update(
            parent_id for parent_id in chunk.parent_object_ids if parent_id.startswith("table:")
        )
    return table_ids


def _is_covered_by_summary(schema_object: SchemaObject, covered_table_ids: set[str]) -> bool:
    if schema_object.object_id.startswith("table:"):
        return schema_object.object_id in covered_table_ids
    if schema_object.table_name and f"table:{schema_object.table_name}" in covered_table_ids:
        return True
    member_refs = schema_object.metadata.get("member_table_refs")
    if isinstance(member_refs, list) and any(
        f"table:{member_ref}" in covered_table_ids for member_ref in member_refs
    ):
        return True
    if schema_object.object_type == "join_candidate":
        return any(
            table_id.removeprefix("table:") in schema_object.object_id
            for table_id in covered_table_ids
        )
    return False


def _schema_context_objects(
    objects: Sequence[SchemaObject],
    *,
    chunks: Sequence[SchemaContextChunk],
) -> list[SchemaContextObject]:
    chunks_by_object: dict[str, SchemaContextChunk] = {chunk.object_id: chunk for chunk in chunks}

    return [
        SchemaContextObject(
            schema_object=schema_object,
            planning_text=_chunk_planning_text(chunks_by_object.get(schema_object.object_id)),
            rank=rank,
        )
        for rank, schema_object in enumerate(objects, start=1)
    ]


def _chunk_planning_text(chunk: SchemaContextChunk | None) -> str:
    if chunk is None:
        return ""
    return chunk.prompt_text or chunk.source_definition or chunk.inferred_usage or chunk.text


def _diagnostics(
    inputs: SchemaContextInputs,
    *,
    context_mode: str,
    schema_context_objects: Sequence[SchemaContextObject],
    object_count: int,
    chunk_count: int,
    context_object_count: int,
) -> dict[str, object]:
    return {
        "context_mode": context_mode,
        "question_context": {
            "text": inputs.text,
            "exact_literals": list(inputs.exact_literals),
            "dates": list(inputs.dates),
            "years": list(inputs.years),
            "identifiers": list(inputs.identifiers),
            "uppercase_codes": list(inputs.uppercase_codes),
            "normalized_tokens": list(inputs.normalized_tokens[:50]),
            "linked_doc_chars": len(inputs.linked_doc_context),
        },
        "context_counts": {
            "objects_total": object_count,
            "chunks_total": chunk_count,
            "available_objects": context_object_count,
        },
        "schema_context_objects": [
            {
                "object_id": obj.schema_object.object_id,
                "object_type": obj.schema_object.object_type,
                "position": obj.rank,
            }
            for obj in schema_context_objects
        ],
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


def _quoted_literals(text: str) -> list[str]:
    literals: list[str] = []
    for match in _QUOTED_LITERAL_RE.finditer(text):
        literal = match.group(1) or match.group(2)
        if literal:
            literals.append(_clean_text(literal))
    return literals


def _normalized_tokens(values: Sequence[str]) -> list[str]:
    tokens: list[str] = []
    for value in values:
        for token in _TOKEN_RE.findall(value):
            tokens.append(token.casefold())
            if "_" in token:
                tokens.extend(part.casefold() for part in token.split("_") if part)
    return _stable_unique(tokens)


def _tokenize(text: str) -> list[str]:
    return _normalized_tokens([text])


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _join_text(parts: Sequence[object]) -> str:
    return " ".join(str(part).strip() for part in parts if str(part).strip())


def _stable_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        cleaned = _clean_text(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique
