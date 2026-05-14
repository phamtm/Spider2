"""Build planner-visible schema context from deterministic metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from sol01.infra.config import SchemaContextConfig
from sol01.models import SchemaContextObject, SchemaObject
from sol01.schema.object_text import (
    build_object_planning_text,
    object_has_large_schema_summary,
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
    """Return planner-visible schema objects with no question-time ranking."""

    _ = question, linked_docs, exact_literals, config
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
