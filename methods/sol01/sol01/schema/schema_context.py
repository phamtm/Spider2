"""Build planner-visible schema context from available metadata."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from sol01.infra.config import SchemaContextConfig
from sol01.models import (
    SchemaContextChunk,
    SchemaContextObject,
    SchemaObject,
)
from sol01.schema.embedding import BM25Index
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
    """Return schema objects ranked by BM25 relevance to the planning question.

    Curated large-schema summaries replace raw metadata for the tables they
    cover, and their containing objects are pre-filtered before BM25 ranking.
    For databases with more objects than top_k_objects, only the top-ranked
    objects are returned.
    """

    config = config or SchemaContextConfig()
    inputs = build_schema_context_inputs(
        question,
        linked_docs=linked_docs,
        exact_literals=exact_literals,
        max_doc_chars=config.max_linked_doc_chars,
    )
    context_mode, context_objects = _available_schema_objects(
        cache.objects,
        cache.chunks,
        inputs.text,
        top_k=config.top_k_objects,
        top_k_sparse=config.top_k_sparse,
    )
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
    query_text: str,
    *,
    top_k: int,
    top_k_sparse: int,
) -> tuple[str, list[SchemaObject]]:
    """Return BM25-ranked schema objects, respecting large-schema-summary pre-filtering."""

    summary_object_ids = {
        chunk.object_id
        for chunk in chunks
        if chunk.metadata.get("large_schema_summaries") or chunk.metadata.get("summary_ids")
    }
    if summary_object_ids:
        covered_table_ids = _covered_table_ids(chunks, summary_object_ids)
        pre_filtered = list(
            obj
            for obj in objects
            if obj.object_id in summary_object_ids
            or not _is_covered_by_summary(obj, covered_table_ids)
        )
        if len(pre_filtered) <= top_k:
            return "large_schema_summary", _sort_schema_objects(pre_filtered)
        filtered_ids = {obj.object_id for obj in pre_filtered}
        filtered_chunks = [c for c in chunks if c.object_id in filtered_ids]
        return "large_schema_summary", _bm25_rank(
            pre_filtered, filtered_chunks, query_text, top_k=top_k, top_k_sparse=top_k_sparse
        )

    if len(objects) <= top_k:
        return "hybrid_retrieval", _bm25_rank(
            objects, chunks, query_text, top_k=len(objects), top_k_sparse=len(chunks)
        )

    return "hybrid_retrieval", _bm25_rank(
        objects, chunks, query_text, top_k=top_k, top_k_sparse=top_k_sparse
    )


_CHUNK_TYPE_TO_OBJECT_TYPE: dict[str, str] = {
    "table": "table",
    "table_family": "family",
    "column_group": "column_group",
    "column": "column",
    "join_candidate": "join_candidate",
    "sample_value": "sample_value",
}
_PER_TYPE_QUOTAS: dict[str, int] = {
    "family": 20,
    "table": 30,
    "column_group": 20,
    "column": 40,
    "join_candidate": 20,
    "sample_value": 20,
}
_DIRECT_WEIGHTS: dict[str, float] = {
    "family": 2.0,
    "table": 1.5,
    "column_group": 1.0,
    "column": 1.0,
    "join_candidate": 0.8,
    "sample_value": 0.5,
}
_PARENT_PROPAGATION: dict[str, float] = {
    "column": 0.4,
    "column_group": 0.4,
    "join_candidate": 0.3,
    "sample_value": 0.3,
}


def _bm25_rank(
    objects: Sequence[SchemaObject],
    chunks: Sequence[SchemaContextChunk],
    query_text: str,
    *,
    top_k: int,
    top_k_sparse: int,
) -> list[SchemaObject]:
    """Return up to top_k schema objects ranked by BM25 relevance."""

    if not chunks or not objects:
        return list(objects)[:top_k]

    evidence_texts = [c.evidence_text or c.text for c in chunks]
    bm25 = BM25Index(evidence_texts)

    sparse_ranked = bm25.scores(query_text, top_k=top_k_sparse)
    exact_boosts = bm25.exact_match_boosts(_extract_exact_tokens(query_text))

    chunk_scores: dict[int, float] = {}
    for doc_id, score in sparse_ranked:
        chunk_scores[doc_id] = score + exact_boosts.get(doc_id, 0.0)
    for doc_id, boost in exact_boosts.items():
        chunk_scores.setdefault(doc_id, boost)

    type_counts: dict[str, int] = defaultdict(int)
    selected_ids: set[int] = set()
    for doc_id, _ in sorted(chunk_scores.items(), key=lambda x: -x[1]):
        ct = _CHUNK_TYPE_TO_OBJECT_TYPE.get(chunks[doc_id].chunk_type, "table")
        if type_counts[ct] < _PER_TYPE_QUOTAS.get(ct, 40):
            selected_ids.add(doc_id)
            type_counts[ct] += 1

    object_by_id = {obj.object_id: obj for obj in objects}
    object_scores: dict[str, float] = defaultdict(float)
    for doc_id in selected_ids:
        chunk = chunks[doc_id]
        score = chunk_scores[doc_id]
        ot = object_by_id.get(chunk.object_id)
        if ot is not None:
            weight = _DIRECT_WEIGHTS.get(ot.object_type, 1.0)
            object_scores[chunk.object_id] += score * weight
        prop_weight = _PARENT_PROPAGATION.get(chunk.chunk_type, 0.2)
        for parent_id in chunk.parent_object_ids:
            if parent_id in object_by_id:
                object_scores[parent_id] += score * prop_weight

    ranked_ids = sorted(object_scores.items(), key=lambda x: -x[1])
    top_ids = {obj_id for obj_id, _ in ranked_ids[:top_k]}
    top_ids.update(obj_id for obj_id, _ in ranked_ids if obj_id in top_ids)

    result = [
        obj for obj_id, _ in ranked_ids[:top_k] if (obj := object_by_id.get(obj_id)) is not None
    ]
    seen = {obj.object_id for obj in result}
    if len(result) < top_k:
        for obj in objects:
            if len(result) >= top_k:
                break
            if obj.object_id not in seen:
                result.append(obj)
                seen.add(obj.object_id)
    return result


def _extract_exact_tokens(text: str) -> list[str]:
    """Extract quoted literals, dotted names, uppercase codes, years, and dates."""
    tokens: list[str] = []
    for match in re.finditer(r"'([^']+)'|\"([^\"]+)\"", text):
        literal = match.group(1) or match.group(2)
        if literal:
            tokens.append(literal.strip())
    tokens.extend(re.findall(r"\b[A-Za-z][A-Za-z0-9]*(?:[._][A-Za-z0-9]+)+\b", text))
    tokens.extend(re.findall(r"\b[A-Z][A-Z0-9_]{1,}\b", text))
    tokens.extend(re.findall(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", text))
    tokens.extend(re.findall(r"\b(?:19|20)\d{2}\b", text))
    return list(dict.fromkeys(filter(None, tokens)))


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
            position=rank,
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
                "position": obj.position,
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
