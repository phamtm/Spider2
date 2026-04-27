"""Rank and expand the table set that should be shown for one task question."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from sol01.index import CACHE_PATH, build_index_cache
from sol01.logging import get_logger
from sol01.models import SchemaSelection, TableSchema

TOKEN_RE = re.compile(r"[a-z0-9_]+")
KEY_SUFFIXES = ("_id", "_code", "_code_prefix", "_name")
MIN_SELECTION_SCORE = 1.5
STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "find",
    "for",
    "from",
    "give",
    "highest",
    "in",
    "least",
    "list",
    "lowest",
    "me",
    "most",
    "name",
    "names",
    "of",
    "on",
    "show",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}
logger = get_logger(__name__)


def retrieve_schema(
    question: str,
    db: str,
    *,
    max_tables: int = 4,
    max_expanded_tables: int = 6,
    cache_path: Path = CACHE_PATH,
) -> SchemaSelection:
    """Return the best-fit table set for a question within one database."""

    if max_tables < 1:
        raise ValueError("max_tables must be at least 1")
    if max_expanded_tables < max_tables:
        raise ValueError("max_expanded_tables must be at least max_tables")

    db_index = load_db_index(db, cache_path=cache_path)
    ranked_tables = _rank_tables(question, db_index)
    selected_tables = _pick_selected_tables(ranked_tables, max_tables=max_tables)
    expanded_tables = _expand_neighbors(
        selected_tables,
        db_index,
        ranked_tables,
        max_expanded_tables=max_expanded_tables,
    )
    logger.info(
        "schema retrieval complete",
        db=db,
        selected_tables=selected_tables,
        expanded_tables=expanded_tables,
        confidence=_confidence(ranked_tables, selected_tables),
    )

    return SchemaSelection(
        db=db,
        selected_tables=selected_tables,
        expanded_tables=expanded_tables,
        rationale=_build_rationale(question, ranked_tables, selected_tables, expanded_tables),
        confidence=_confidence(ranked_tables, selected_tables),
    )


def load_db_index(db: str, *, cache_path: Path = CACHE_PATH) -> dict[str, TableSchema]:
    """Load one database index from the cache, building it when needed."""

    payload = load_index_cache(cache_path=cache_path)
    try:
        return payload[db]
    except KeyError as exc:
        raise KeyError(f"Unknown database: {db}") from exc


def load_index_cache(
    *,
    cache_path: Path = CACHE_PATH,
) -> dict[str, dict[str, TableSchema]]:
    """Load the cached schema index, or build it if the cache is missing."""

    if not cache_path.exists():
        return build_index_cache(cache_path=cache_path)

    raw_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    return {
        db_name: {
            table_name: TableSchema.model_validate(table_schema)
            for table_name, table_schema in tables.items()
        }
        for db_name, tables in raw_payload.items()
    }


def _rank_tables(
    question: str, db_index: dict[str, TableSchema]
) -> list[tuple[str, float, list[str]]]:
    """Score tables by overlap with the question across names, columns, and examples."""

    question_terms = _question_terms(question)
    ranked: list[tuple[str, float, list[str]]] = []

    for table_name, table in db_index.items():
        name_terms = _token_set(table.name)
        column_terms = _token_set(" ".join(column.name for column in table.columns))
        description_terms = _token_set(
            " ".join(column.description or "" for column in table.columns)
        )
        sample_terms = _token_set(_sample_text(table))
        matches: list[str] = []
        score = 0.0

        for term in sorted(question_terms):
            term_score = 0.0
            if term in name_terms:
                term_score += 6.0
            if term in column_terms:
                term_score += 3.0
            if term in description_terms:
                term_score += 1.5
            if term in sample_terms:
                term_score += 0.25
            if term_score > 0:
                matches.append(term)
                score += term_score

        if _phrase_match(question, table.name):
            score += 4.0
        if any(_phrase_match(question, column.name) for column in table.columns):
            score += 2.0

        ranked.append((table_name, score, matches))

    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def _pick_selected_tables(
    ranked_tables: list[tuple[str, float, list[str]]],
    *,
    max_tables: int,
) -> list[str]:
    """Keep the top lexical hits, falling back to one best table when overlap is weak."""

    selected = [
        table_name for table_name, score, _ in ranked_tables if score >= MIN_SELECTION_SCORE
    ][:max_tables]
    if selected:
        return selected
    return [ranked_tables[0][0]]


def _expand_neighbors(
    selected_tables: list[str],
    db_index: dict[str, TableSchema],
    ranked_tables: list[tuple[str, float, list[str]]],
    *,
    max_expanded_tables: int,
) -> list[str]:
    """Add a small number of likely join neighbors based on shared key columns."""

    neighbor_scores = _neighbor_scores(db_index)
    ranking_position = {table_name: index for index, (table_name, _, _) in enumerate(ranked_tables)}
    lexical_scores = {table_name: score for table_name, score, _ in ranked_tables}
    match_counts = {table_name: len(matches) for table_name, _, matches in ranked_tables}
    expanded_tables = list(selected_tables)
    candidate_scores: dict[str, float] = defaultdict(float)
    candidate_links: dict[str, int] = defaultdict(int)
    for selected in selected_tables:
        for neighbor, score in neighbor_scores.get(selected, {}).items():
            if neighbor in expanded_tables:
                continue
            candidate_scores[neighbor] += score
            candidate_links[neighbor] += 1

    candidates = [
        (
            score,
            candidate_links[neighbor],
            lexical_scores.get(neighbor, 0.0),
            match_counts.get(neighbor, 0),
            ranking_position.get(neighbor, 10_000),
            neighbor,
        )
        for neighbor, score in candidate_scores.items()
    ]
    for _, _, _, _, _, neighbor in sorted(
        candidates,
        key=lambda item: (-item[0], -item[1], -item[2], -item[3], item[4], item[5]),
    ):
        if neighbor in expanded_tables:
            continue
        expanded_tables.append(neighbor)
        if len(expanded_tables) == max_expanded_tables:
            break

    return expanded_tables


def _neighbor_scores(db_index: dict[str, TableSchema]) -> dict[str, dict[str, float]]:
    """Find conservative table links from exact shared key-like column names."""

    column_map: dict[str, list[str]] = defaultdict(list)
    for table_name, table in db_index.items():
        for column in table.columns:
            normalized = _normalize_column_name(column.name)
            if _is_join_like_column(normalized):
                column_map[normalized].append(table_name)

    neighbor_scores: dict[str, dict[str, float]] = defaultdict(dict)
    for column_name, tables in column_map.items():
        if len(tables) < 2:
            continue

        # Shared keys that connect many tables are less specific, so we dampen them.
        weight = _join_weight(column_name) / (len(tables) - 1)
        for left in tables:
            for right in tables:
                if left == right:
                    continue
                neighbor_scores[left][right] = neighbor_scores[left].get(right, 0.0) + weight

    return neighbor_scores


def _build_rationale(
    question: str,
    ranked_tables: list[tuple[str, float, list[str]]],
    selected_tables: list[str],
    expanded_tables: list[str],
) -> str:
    """Explain why the selected tables won and whether extra neighbors were added."""

    top_hits: list[str] = []
    for table_name, _score, matches in ranked_tables:
        if table_name not in selected_tables:
            continue
        if matches:
            top_hits.append(f"{table_name} ({', '.join(matches[:3])})")
        else:
            top_hits.append(f"{table_name} (best fallback)")

    rationale = (
        f"Ranked tables for '{question}' using overlap across table names, columns, and examples: "
        + ", ".join(top_hits)
        + "."
    )
    if expanded_tables != selected_tables:
        added = [table for table in expanded_tables if table not in selected_tables]
        rationale += f" Added join neighbors with shared key columns: {', '.join(added)}."
    return rationale


def _confidence(
    ranked_tables: list[tuple[str, float, list[str]]],
    selected_tables: list[str],
) -> float:
    """Turn the lexical ranking margin into a bounded confidence score."""

    if not ranked_tables or not selected_tables:
        return 0.0

    selected_scores = [
        score for table_name, score, _ in ranked_tables if table_name in selected_tables
    ]
    best_score = selected_scores[0]
    next_score = next(
        (score for table_name, score, _ in ranked_tables if table_name not in selected_tables),
        0.0,
    )
    margin = max(best_score - next_score, 0.0)
    confidence = 0.45 + min(best_score, 12.0) / 30.0 + min(margin, 6.0) / 20.0
    return min(round(confidence, 3), 0.95)


def _question_terms(question: str) -> set[str]:
    """Extract simple singularized tokens from the question text."""

    return _token_set(question)


def _token_set(text: str) -> set[str]:
    """Split text into normalized words and simple singular forms."""

    terms: set[str] = set()
    for raw_token in TOKEN_RE.findall(text.lower()):
        if not raw_token:
            continue
        parts = [part for part in raw_token.split("_") if part]
        for part in parts or [raw_token]:
            if len(part) < 2:
                continue
            singular = _singularize(part)
            if singular in STOPWORDS:
                continue
            terms.add(part)
            if singular != part:
                terms.add(singular)
    return terms


def _phrase_match(question: str, name: str) -> bool:
    """Check whether a whole table or column name appears as a phrase."""

    normalized_question = _normalize_phrase_text(question)
    normalized_name = _normalize_phrase_text(name)
    return normalized_name in normalized_question


def _normalize_phrase_text(text: str) -> str:
    """Normalize text into a stable space-separated phrase string."""

    parts: list[str] = []
    for raw_token in TOKEN_RE.findall(text.lower()):
        if not raw_token:
            continue
        for part in raw_token.split("_"):
            if len(part) < 2:
                continue
            singular = _singularize(part)
            if singular in STOPWORDS:
                continue
            parts.append(singular)
    return " ".join(parts)


def _normalize_column_name(name: str) -> str:
    """Normalize a column name before using it for neighbor matching."""

    return "_".join(TOKEN_RE.findall(name.lower()))


def _sample_text(table: TableSchema) -> str:
    """Collect example values without mixing them into stronger schema signals."""

    parts: list[str] = []
    for column in table.columns:
        parts.extend(column.sample_values)
    for row in table.sample_rows:
        parts.extend(str(value) for value in row.values() if value is not None)
    return " ".join(parts)


def _singularize(token: str) -> str:
    """Apply a small singularization rule for common plural forms."""

    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _is_join_like_column(column_name: str) -> bool:
    """Keep neighbor expansion tight by only using key-like shared columns."""

    return column_name.endswith(KEY_SUFFIXES)


def _join_weight(column_name: str) -> float:
    """Weight stronger shared keys higher than softer shared names."""

    if column_name.endswith("_id"):
        return 3.0
    if column_name.endswith("_code") or column_name.endswith("_code_prefix"):
        return 2.0
    return 1.0
