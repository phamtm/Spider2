"""Lexical query-time retrieval over versioned schema chunks."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from sol01.infra.config import SchemaRetrievalConfig
from sol01.models import RetrievalChunk, RetrievedChunk, RetrievedSchemaObject, SchemaObject
from sol01.schema.retrieval_index import SchemaRetrievalIndex

DEFAULT_TYPE_QUOTAS: dict[str, int] = {
    "table_family": 20,
    "table": 30,
    "column_group": 20,
    "column": 40,
    "join_candidate": 20,
    "sample_value": 20,
}
EXACT_BOOST = 4.0
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_DOTTED_OR_UNDERSCORE_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[._][A-Za-z0-9]+)+\b")
_UPPER_CODE_RE = re.compile(r"\b[A-Z][A-Z0-9_]{1,}\b")
_DATE_RE = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_QUOTED_LITERAL_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"")


@dataclass(frozen=True)
class RetrievalQuery:
    """Search text and extracted high-signal terms for schema retrieval."""

    text: str
    question: str
    linked_doc_context: str
    exact_literals: tuple[str, ...]
    dates: tuple[str, ...]
    years: tuple[str, ...]
    identifiers: tuple[str, ...]
    uppercase_codes: tuple[str, ...]
    normalized_tokens: tuple[str, ...]


@dataclass
class _Candidate:
    chunk: RetrievalChunk
    sparse_score: float = 0.0
    exact_score: float = 0.0
    evidence: set[str] = field(default_factory=set)
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def pre_score(self) -> float:
        return self.sparse_score + self.exact_score

    @property
    def final_score(self) -> float:
        return self.pre_score


@dataclass(frozen=True)
class _ExactLookup:
    text: str
    tokens: frozenset[str]
    table_names: frozenset[str]
    column_names: frozenset[str]
    aliases: tuple[str, ...]


def build_retrieval_query(
    question: str,
    *,
    linked_docs: Sequence[str] = (),
    exact_literals: Sequence[str] = (),
    max_doc_chars: int = 6000,
) -> RetrievalQuery:
    """Build the query text used by sparse and exact schema retrieval."""

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
    """Clip linked-document context, preferring passages with lexical overlap."""

    if max_doc_chars < 1 or not linked_docs:
        return ""
    passages = [
        _clean_text(passsage)
        for doc in linked_docs
        for passsage in re.split(r"\n\s*\n|(?<=[.!?])\s+", doc)
        if _clean_text(passsage)
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
    type_quotas: Mapping[str, int] | None = None,
    top_k_objects: int | None = None,
) -> tuple[list[RetrievedSchemaObject], dict[str, object]]:
    """Return ranked schema objects and compact retrieval diagnostics."""

    config = config or SchemaRetrievalConfig()
    query = build_retrieval_query(
        question,
        linked_docs=linked_docs,
        exact_literals=exact_literals,
        max_doc_chars=config.max_linked_doc_chars,
    )
    quotas = dict(DEFAULT_TYPE_QUOTAS if type_quotas is None else type_quotas)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in index.chunks}
    candidates: dict[str, _Candidate] = {}

    sparse_hits = _sparse_hits(index, query.text)
    exact_hits = _exact_hits(index.chunks, query)

    _merge_hits(candidates, chunk_by_id, _quota_hits(sparse_hits, chunk_by_id, quotas), "sparse")
    _merge_hits(candidates, chunk_by_id, _quota_hits(exact_hits, chunk_by_id, quotas), "exact")
    _prune_candidates_to_type_quotas(candidates, quotas)

    retrieved_objects = _aggregate_objects(
        candidates,
        objects=index.objects,
        top_k=top_k_objects or config.object_top_k,
    )
    diagnostics = _diagnostics(
        query,
        candidates,
        retrieved_objects,
        sparse_hits=sparse_hits,
        exact_hits=exact_hits,
    )
    return retrieved_objects, diagnostics


def _sparse_hits(index: SchemaRetrievalIndex, query_text: str) -> list[tuple[str, float]]:
    query_terms = Counter(_tokenize(query_text))
    if not query_terms:
        return []
    sparse = index.sparse
    document_count = int(sparse.get("document_count") or 0)
    average_length = float(sparse.get("average_document_length") or 0.0)
    if document_count < 1 or average_length <= 0.0:
        return []

    parameters = sparse.get("parameters") if isinstance(sparse.get("parameters"), dict) else {}
    k1 = float(parameters.get("k1") or 1.2)
    b = float(parameters.get("b") or 0.75)
    document_frequency = {
        str(term): int(count)
        for term, count in dict(sparse.get("document_frequency") or {}).items()
    }
    lengths = [int(length) for length in list(sparse.get("document_lengths") or [])]
    hits: list[tuple[str, float]] = []
    for index_position, document in enumerate(list(sparse.get("documents") or [])):
        if not isinstance(document, dict):
            continue
        terms = {str(term): int(count) for term, count in dict(document.get("terms") or {}).items()}
        length = lengths[index_position] if index_position < len(lengths) else sum(terms.values())
        score = 0.0
        for term, query_count in query_terms.items():
            frequency = terms.get(term, 0)
            if frequency == 0:
                continue
            df = document_frequency.get(term, 0)
            idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1.0 - b + b * (length / average_length))
            score += query_count * idf * ((frequency * (k1 + 1.0)) / denominator)
        chunk_id = str(document.get("chunk_id") or "")
        if chunk_id and score > 0.0:
            hits.append((chunk_id, score))
    return sorted(hits, key=lambda item: (-item[1], item[0]))


def _exact_hits(chunks: Sequence[RetrievalChunk], query: RetrievalQuery) -> list[tuple[str, float]]:
    hits: list[tuple[str, float]] = []
    for chunk in chunks:
        lookup = _exact_lookup(chunk)
        score = _exact_score(query, lookup)
        if score > 0.0:
            hits.append((chunk.chunk_id, score))
    return sorted(hits, key=lambda item: (-item[1], item[0]))


def _exact_score(query: RetrievalQuery, lookup: _ExactLookup) -> float:
    score = 0.0
    for literal in query.exact_literals:
        if _contains_phrase(lookup.text, literal):
            score += EXACT_BOOST
    for date in query.dates:
        score += EXACT_BOOST * _variant_match_count(_date_variants(date), lookup)
    for year in query.years:
        if _contains_token(lookup.tokens, year):
            score += EXACT_BOOST * 0.75
    for identifier in query.identifiers:
        score += EXACT_BOOST * _identifier_match_count(identifier, lookup)
    for code in query.uppercase_codes:
        if _contains_token(lookup.tokens, code):
            score += EXACT_BOOST * 0.75
    for token in query.normalized_tokens:
        if token in lookup.table_names:
            score += EXACT_BOOST
        if token in lookup.column_names:
            score += EXACT_BOOST
    for alias in lookup.aliases:
        if _contains_phrase(query.text.casefold(), alias):
            score += EXACT_BOOST * 1.5
    return score


def _exact_lookup(chunk: RetrievalChunk) -> _ExactLookup:
    metadata_strings = _flatten_strings(chunk.metadata)
    lookup_text = _join_text(
        [
            chunk.chunk_id,
            chunk.object_id,
            chunk.bm25_text,
            chunk.prompt_text,
            chunk.source_definition,
            chunk.inferred_usage,
            " ".join(metadata_strings),
        ]
    ).casefold()
    return _ExactLookup(
        text=lookup_text,
        tokens=frozenset(_tokenize(lookup_text)),
        table_names=frozenset(_table_name_terms(_table_name_values(chunk))),
        column_names=frozenset(_column_name_terms(_column_name_values(chunk))),
        aliases=tuple(alias.casefold() for alias in _summary_aliases(chunk) if alias.strip()),
    )


def _table_name_values(chunk: RetrievalChunk) -> list[str]:
    values = [
        str(chunk.metadata.get("table_name") or ""),
        str(chunk.metadata.get("table_full_name") or ""),
        str(chunk.metadata.get("full_name") or ""),
        str(chunk.metadata.get("short_name") or ""),
    ]
    if chunk.object_id.startswith("table:"):
        values.append(chunk.object_id.removeprefix("table:"))
    return [value for value in values if value.strip()]


def _column_name_values(chunk: RetrievalChunk) -> list[str]:
    values = [
        str(chunk.metadata.get("column_name") or ""),
        str(chunk.metadata.get("normalized_column_name") or ""),
    ]
    if chunk.object_id.startswith("column:") and "#" in chunk.object_id:
        values.append(chunk.object_id.rsplit("#", 1)[-1])
    return [value for value in values if value.strip()]


def _summary_aliases(chunk: RetrievalChunk) -> list[str]:
    aliases: list[str] = []
    for key in ("aliases", "summary_aliases", "curated_summary_aliases"):
        aliases.extend(_string_list(chunk.metadata.get(key)))
    summaries = chunk.metadata.get("large_schema_summaries")
    if isinstance(summaries, Sequence) and not isinstance(summaries, str):
        for summary in summaries:
            if isinstance(summary, Mapping):
                aliases.extend(_string_list(summary.get("aliases")))
    return _stable_unique(aliases)


def _table_name_terms(values: Sequence[str]) -> list[str]:
    terms: list[str] = []
    for value in values:
        cleaned = _clean_text(value)
        short_name = cleaned.rsplit(".", 1)[-1]
        terms.extend([cleaned.casefold(), short_name.casefold()])
        terms.extend(_normalized_tokens([short_name]))
    return _stable_unique(terms)


def _column_name_terms(values: Sequence[str]) -> list[str]:
    terms: list[str] = []
    for value in values:
        cleaned = _clean_text(value)
        terms.append(cleaned.casefold())
        terms.extend(_normalized_tokens([cleaned]))
    return _stable_unique(terms)


def _identifier_variants(identifier: str) -> list[str]:
    cleaned = _clean_text(identifier)
    variants = [cleaned.casefold()]
    if "." in cleaned:
        variants.append(cleaned.rsplit(".", 1)[-1].casefold())
    return _stable_unique(variants)


def _identifier_match_count(identifier: str, lookup: _ExactLookup) -> int:
    variants = _identifier_variants(identifier)
    return sum(1 for variant in variants if _contains_token_or_phrase(lookup, variant))


def _date_variants(date: str) -> list[str]:
    match = re.fullmatch(r"(?P<year>\d{4})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})", date)
    if match is None:
        return [date]
    year = match.group("year")
    month = match.group("month").zfill(2)
    day = match.group("day").zfill(2)
    compact = f"{year}{month}{day}"
    return [date, f"{year}-{month}-{day}", f"{year}/{month}/{day}", compact, f"_{compact}"]


def _variant_match_count(variants: Sequence[str], lookup: _ExactLookup) -> int:
    return sum(
        1 for variant in _stable_unique(variants) if _contains_token_or_phrase(lookup, variant)
    )


def _contains_token_or_phrase(lookup: _ExactLookup, value: str) -> bool:
    normalized = value.casefold()
    if _contains_token(lookup.tokens, normalized):
        return True
    return _contains_phrase(lookup.text, normalized)


def _contains_token(tokens: frozenset[str], value: str) -> bool:
    return value.casefold() in tokens


def _contains_phrase(haystack: str, phrase: str) -> bool:
    normalized = phrase.casefold().strip()
    if not normalized:
        return False
    if re.fullmatch(r"[A-Za-z0-9_]+", normalized):
        boundary_pattern = rf"(?<![A-Za-z0-9_]){re.escape(normalized)}(?![A-Za-z0-9_])"
        return re.search(boundary_pattern, haystack) is not None
    return normalized in haystack


def _quota_hits(
    hits: Sequence[tuple[str, float]],
    chunk_by_id: Mapping[str, RetrievalChunk],
    quotas: Mapping[str, int],
) -> list[tuple[str, float]]:
    counts: defaultdict[str, int] = defaultdict(int)
    selected: list[tuple[str, float]] = []
    for chunk_id, score in hits:
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None:
            continue
        limit = quotas.get(chunk.chunk_type, max(quotas.values(), default=20))
        if counts[chunk.chunk_type] >= limit:
            continue
        counts[chunk.chunk_type] += 1
        selected.append((chunk_id, score))
    return selected


def _merge_hits(
    candidates: dict[str, _Candidate],
    chunk_by_id: Mapping[str, RetrievalChunk],
    hits: Sequence[tuple[str, float]],
    source: str,
) -> None:
    for rank, (chunk_id, score) in enumerate(hits, start=1):
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None:
            continue
        candidate = candidates.setdefault(chunk_id, _Candidate(chunk=chunk))
        if source == "sparse":
            candidate.sparse_score = max(candidate.sparse_score, score)
        else:
            candidate.exact_score = max(candidate.exact_score, score)
        candidate.evidence.add(source)
        candidate.ranks[source] = min(candidate.ranks.get(source, rank), rank)


def _prune_candidates_to_type_quotas(
    candidates: dict[str, _Candidate],
    quotas: Mapping[str, int],
) -> None:
    counts: defaultdict[str, int] = defaultdict(int)
    keep: set[str] = set()
    ordered = sorted(candidates.values(), key=lambda item: (-item.pre_score, item.chunk.chunk_id))
    for candidate in ordered:
        limit = quotas.get(candidate.chunk.chunk_type, max(quotas.values(), default=20))
        if counts[candidate.chunk.chunk_type] >= limit:
            continue
        counts[candidate.chunk.chunk_type] += 1
        keep.add(candidate.chunk.chunk_id)
    for chunk_id in list(candidates):
        if chunk_id not in keep:
            del candidates[chunk_id]


def _aggregate_objects(
    candidates: Mapping[str, _Candidate],
    *,
    objects: Sequence[SchemaObject],
    top_k: int,
) -> list[RetrievedSchemaObject]:
    object_by_id = {obj.object_id: obj for obj in objects}
    family_by_table = _family_lookup(objects)
    scores: defaultdict[str, float] = defaultdict(float)
    chunks_by_object: defaultdict[str, list[RetrievedChunk]] = defaultdict(list)
    ordered_candidates = sorted(
        candidates.values(),
        key=lambda item: (-item.final_score, item.chunk.chunk_id),
    )

    for candidate_rank, candidate in enumerate(ordered_candidates, start=1):
        chunk = candidate.chunk
        contribution_targets = _aggregation_targets(chunk, family_by_table)
        retrieved_chunk = RetrievedChunk(
            chunk=chunk,
            rank=candidate_rank,
            score=candidate.pre_score,
        )
        for object_id, weight in contribution_targets.items():
            if object_id not in object_by_id:
                continue
            scores[object_id] += candidate.final_score * weight
            chunks_by_object[object_id].append(retrieved_chunk)

    ranked_ids = sorted(scores, key=lambda object_id: (-scores[object_id], object_id))[:top_k]
    return [
        RetrievedSchemaObject(
            schema_object=object_by_id[object_id],
            chunks=chunks_by_object[object_id][:5],
            rank=rank,
            score=scores[object_id],
        )
        for rank, object_id in enumerate(ranked_ids, start=1)
    ]


def _aggregation_targets(
    chunk: RetrievalChunk,
    family_by_table: Mapping[str, list[str]],
) -> dict[str, float]:
    direct_weight = 1.3 if chunk.chunk_type in {"table", "table_family"} else 1.0
    targets: dict[str, float] = {chunk.object_id: direct_weight}
    parent_weight = 0.9 if chunk.chunk_type == "join_candidate" else 0.75
    if chunk.chunk_type in {"column", "column_group", "join_candidate", "sample_value"}:
        for parent_id in chunk.parent_object_ids:
            if parent_id.startswith("table:"):
                targets[parent_id] = max(targets.get(parent_id, 0.0), parent_weight)
                for family_id in family_by_table.get(parent_id.removeprefix("table:"), []):
                    targets[family_id] = max(targets.get(family_id, 0.0), parent_weight * 0.8)
            elif chunk.chunk_type != "sample_value":
                targets[parent_id] = max(targets.get(parent_id, 0.0), 0.55)
    return targets


def _family_lookup(objects: Sequence[SchemaObject]) -> dict[str, list[str]]:
    lookup: defaultdict[str, list[str]] = defaultdict(list)
    for obj in objects:
        if obj.object_type != "family":
            continue
        for table_name in _string_list(obj.metadata.get("member_table_refs")):
            lookup[table_name].append(obj.object_id)
    return dict(lookup)


def _diagnostics(
    query: RetrievalQuery,
    candidates: Mapping[str, _Candidate],
    retrieved_objects: Sequence[RetrievedSchemaObject],
    *,
    sparse_hits: Sequence[tuple[str, float]],
    exact_hits: Sequence[tuple[str, float]],
) -> dict[str, object]:
    candidate_types = Counter(candidate.chunk.chunk_type for candidate in candidates.values())
    return {
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
        "hit_counts": {
            "sparse": len(sparse_hits),
            "exact": len(exact_hits),
        },
        "candidate_counts": {
            "merged": len(candidates),
            "by_type": dict(sorted(candidate_types.items())),
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
        "candidate_evidence": [
            {
                "chunk_id": candidate.chunk.chunk_id,
                "chunk_type": candidate.chunk.chunk_type,
                "sources": sorted(candidate.evidence),
                "score": candidate.final_score,
            }
            for candidate in sorted(
                candidates.values(),
                key=lambda item: (-item.final_score, item.chunk.chunk_id),
            )[:25]
        ],
    }


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


def _flatten_strings(value: object) -> list[str]:
    if isinstance(value, Mapping):
        return [
            item
            for nested_value in value.values()
            for item in _flatten_strings(nested_value)
            if item
        ]
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [item for nested_value in value for item in _flatten_strings(nested_value) if item]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


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


def _string_list(value: object) -> list[str]:
    if isinstance(value, Mapping):
        return [str(item).strip() for item in value.values() if str(item).strip()]
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value).strip()] if str(value).strip() else []
