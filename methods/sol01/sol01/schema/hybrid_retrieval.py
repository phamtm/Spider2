"""Deterministic schema context selection over versioned schema chunks."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from sol01.infra.config import SchemaRetrievalConfig
from sol01.models import RetrievalChunk, RetrievedChunk, RetrievedSchemaObject, SchemaObject
from sol01.schema.retrieval_index import SchemaRetrievalIndex

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
class RetrievalQuery:
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


def build_retrieval_query(
    question: str,
    *,
    linked_docs: Sequence[str] = (),
    exact_literals: Sequence[str] = (),
    max_doc_chars: int = 6000,
) -> RetrievalQuery:
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
    return RetrievalQuery(
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


def retrieve_schema_objects(
    index: SchemaRetrievalIndex,
    question: str,
    *,
    linked_docs: Sequence[str] = (),
    exact_literals: Sequence[str] = (),
    config: SchemaRetrievalConfig | None = None,
    type_quotas: object | None = None,
    top_k_objects: int | None = None,
) -> tuple[list[RetrievedSchemaObject], dict[str, object]]:
    """Return available schema objects without BM25, embeddings, or exact-match ranking.

    Curated large-schema summaries replace raw metadata for the tables they
    cover. Uncovered tables still flow through as normal database metadata.
    """

    del type_quotas
    config = config or SchemaRetrievalConfig()
    query = build_retrieval_query(
        question,
        linked_docs=linked_docs,
        exact_literals=exact_literals,
        max_doc_chars=config.max_linked_doc_chars,
    )
    context_mode, context_objects = _context_objects(index.objects, index.chunks)
    retrieved_objects = _retrieved_objects(
        context_objects,
        chunks=index.chunks,
        top_k=top_k_objects,
    )
    diagnostics = _diagnostics(
        query,
        context_mode=context_mode,
        retrieved_objects=retrieved_objects,
        object_count=len(index.objects),
        chunk_count=len(index.chunks),
        context_object_count=len(context_objects),
    )
    return retrieved_objects, diagnostics


def _context_objects(
    objects: Sequence[SchemaObject],
    chunks: Sequence[RetrievalChunk],
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
    chunks: Sequence[RetrievalChunk],
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


def _retrieved_objects(
    objects: Sequence[SchemaObject],
    *,
    chunks: Sequence[RetrievalChunk],
    top_k: int | None,
) -> list[RetrievedSchemaObject]:
    chunks_by_object: dict[str, list[RetrievalChunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_object[chunk.object_id].append(chunk)

    selected_objects = list(objects[:top_k] if top_k is not None else objects)
    return [
        RetrievedSchemaObject(
            schema_object=schema_object,
            chunks=[
                RetrievedChunk(chunk=chunk, rank=chunk_rank)
                for chunk_rank, chunk in enumerate(
                    sorted(
                        chunks_by_object.get(schema_object.object_id, []),
                        key=lambda item: item.chunk_id,
                    ),
                    start=1,
                )
            ],
            rank=rank,
        )
        for rank, schema_object in enumerate(selected_objects, start=1)
    ]


def _diagnostics(
    query: RetrievalQuery,
    *,
    context_mode: str,
    retrieved_objects: Sequence[RetrievedSchemaObject],
    object_count: int,
    chunk_count: int,
    context_object_count: int,
) -> dict[str, object]:
    return {
        "mode": context_mode,
        "query": {
            "text": query.text,
            "exact_literals": list(query.exact_literals),
            "dates": list(query.dates),
            "years": list(query.years),
            "identifiers": list(query.identifiers),
            "uppercase_codes": list(query.uppercase_codes),
            "normalized_tokens": list(query.normalized_tokens[:50]),
            "linked_doc_chars": len(query.linked_doc_context),
        },
        "candidate_counts": {
            "objects_total": object_count,
            "chunks_total": chunk_count,
            "available_objects": context_object_count,
        },
        "selected_objects": [
            {
                "object_id": obj.schema_object.object_id,
                "object_type": obj.schema_object.object_type,
                "rank": obj.rank,
                "score": obj.score,
                "evidence_chunk_ids": [chunk.chunk.chunk_id for chunk in obj.chunks],
            }
            for obj in retrieved_objects
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
