"""Build planner-visible schema context from available metadata."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from sol01.infra.config import SchemaContextConfig
from sol01.models import SchemaContextObject, SchemaObject
from sol01.schema.embedding import BM25Index
from sol01.schema.object_text import (
    build_object_planning_text,
    build_object_search_text,
    covered_table_ids_for_summary_object,
    object_has_large_schema_summary,
    object_parent_ids,
)
from sol01.schema.schema_context_cache import SchemaContextCache

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
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
class QuestionContext:
    """Compact question context used for schema-object ranking and traces."""

    text: str
    question: str
    linked_doc_context: str


def build_question_context(
    question: str,
    *,
    linked_docs: Sequence[str] = (),
    exact_literals: Sequence[str] = (),
    max_doc_chars: int = 6000,
) -> QuestionContext:
    """Build normalized question context without ranking schema objects."""

    clean_question = _clean_text(question)
    literals = _stable_unique(
        [
            *_quoted_literals(clean_question),
            *(_clean_text(value) for value in exact_literals),
        ]
    )
    query_terms = _normalized_tokens([clean_question, *literals])
    linked_doc_context = clip_linked_docs(
        linked_docs,
        query_terms=query_terms,
        max_doc_chars=max_doc_chars,
    )
    parts = [
        clean_question,
        linked_doc_context,
        " ".join(literals),
    ]
    return QuestionContext(
        text=_join_text(parts),
        question=clean_question,
        linked_doc_context=linked_doc_context,
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
    """Return planner-visible schema objects for the current question.

    Curated large-schema summaries replace raw metadata for the tables they
    cover, and their containing objects are pre-filtered before ranking.
    For databases with more objects than top_k_objects, only the strongest
    matches are returned.
    """

    config = config or SchemaContextConfig()
    context = build_question_context(
        question,
        linked_docs=linked_docs,
        exact_literals=exact_literals,
        max_doc_chars=config.max_linked_doc_chars,
    )
    context_mode, context_objects = _available_schema_objects(
        cache.objects,
        context.text,
        top_k=config.top_k_objects,
        top_k_sparse=config.top_k_sparse,
    )
    schema_context_objects = _schema_context_objects(context_objects)
    diagnostics = _diagnostics(
        context,
        context_mode=context_mode,
        schema_context_objects=schema_context_objects,
        object_count=len(cache.objects),
        context_object_count=len(context_objects),
    )
    return schema_context_objects, diagnostics


def _available_schema_objects(
    objects: Sequence[SchemaObject],
    query_text: str,
    *,
    top_k: int,
    top_k_sparse: int,
) -> tuple[str, list[SchemaObject]]:
    """Return ranked schema objects, respecting large-schema-summary pre-filtering."""

    summary_object_ids = {obj.object_id for obj in objects if object_has_large_schema_summary(obj)}
    if summary_object_ids:
        covered_table_ids = _covered_table_ids(objects, summary_object_ids)
        pre_filtered = list(
            obj
            for obj in objects
            if obj.object_id in summary_object_ids
            or not _is_covered_by_summary(obj, covered_table_ids)
        )
        if len(pre_filtered) <= top_k:
            return "large_schema_summary", _sort_schema_objects(pre_filtered)
        return "large_schema_summary", _bm25_rank(
            pre_filtered, query_text, top_k=top_k, top_k_sparse=top_k_sparse
        )

    if len(objects) <= top_k:
        return "schema_objects", _bm25_rank(
            objects, query_text, top_k=len(objects), top_k_sparse=len(objects)
        )

    return "schema_objects", _bm25_rank(objects, query_text, top_k=top_k, top_k_sparse=top_k_sparse)


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
    query_text: str,
    *,
    top_k: int,
    top_k_sparse: int,
) -> list[SchemaObject]:
    """Return up to top_k schema objects ranked by BM25 relevance."""

    if not objects:
        return list(objects)[:top_k]

    evidence_texts = [build_object_search_text(obj) for obj in objects]
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
        object_type = objects[doc_id].object_type
        if type_counts[object_type] < _PER_TYPE_QUOTAS.get(object_type, 40):
            selected_ids.add(doc_id)
            type_counts[object_type] += 1

    object_by_id = {obj.object_id: obj for obj in objects}
    object_scores: dict[str, float] = defaultdict(float)
    for doc_id in selected_ids:
        schema_object = objects[doc_id]
        score = chunk_scores[doc_id]
        weight = _DIRECT_WEIGHTS.get(schema_object.object_type, 1.0)
        object_scores[schema_object.object_id] += score * weight
        prop_weight = _PARENT_PROPAGATION.get(schema_object.object_type, 0.2)
        for parent_id in object_parent_ids(schema_object):
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
    objects: Sequence[SchemaObject],
    summary_object_ids: set[str],
) -> set[str]:
    table_ids: set[str] = set()
    for schema_object in objects:
        if schema_object.object_id not in summary_object_ids:
            continue
        table_ids.update(covered_table_ids_for_summary_object(schema_object))
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
    context: QuestionContext,
    *,
    context_mode: str,
    schema_context_objects: Sequence[SchemaContextObject],
    object_count: int,
    context_object_count: int,
) -> dict[str, object]:
    return {
        "context_mode": context_mode,
        "question_context": {
            "text": context.text,
            "linked_doc_chars": len(context.linked_doc_context),
        },
        "context_counts": {
            "objects_total": object_count,
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
