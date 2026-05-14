"""Shared renderers for LargeSchemaSummary facts.

Both planning chunks and SQL reference context render the same summary fields.
This module is the single source of truth for summary wording and structure.
"""

from __future__ import annotations

from sol01.schema.large_schema_summaries import LargeSchemaSummary


def render_summary_text(summaries: list[LargeSchemaSummary]) -> str:
    """Sentence-based summary block for planning chunks (comma-joined field lists)."""

    lines: list[str] = []
    for summary in summaries:
        lines.extend(
            [
                f"Large-schema summary: {summary.summary_id}.",
                f"Purpose: {summary.purpose}",
                f"Grain: {summary.grain}",
                _field_list("Stable exact columns", summary.stable_columns),
                _field_list("Repeated or partition column rules", summary.repeated_column_rules),
                _field_list("Inclusive ranges", summary.inclusive_ranges),
                _field_list("Quote and spelling rules", summary.quote_spelling_rules),
                _field_list("Exact safe examples", summary.examples),
                _field_list("Aliases", summary.aliases),
            ]
        )
    return _join_sentences(lines)


def render_summary_lines(summary: LargeSchemaSummary) -> list[str]:
    """Line-list summary block for SQL reference prompts (bullet-point field sections).

    Does not include the table-name header line — callers prepend it.
    """

    lines = [
        f"Large-schema summary: {summary.summary_id}",
        f"Purpose: {summary.purpose}",
        f"Grain: {summary.grain}",
        "Use only exact names from these references or names confirmed by validation.",
    ]
    lines.extend(_section("Stable exact columns", summary.stable_columns))
    lines.extend(_section("Repeated or partition column rules", summary.repeated_column_rules))
    lines.extend(_section("Inclusive ranges", summary.inclusive_ranges))
    lines.extend(_section("Quote and spelling rules", summary.quote_spelling_rules))
    lines.extend(_section("Exact safe examples", summary.examples))
    return lines


def render_summary_payload(summary: LargeSchemaSummary) -> dict[str, object]:
    """Dict payload stored in chunk metadata."""

    return {
        "summary_id": summary.summary_id,
        "purpose": summary.purpose,
        "grain": summary.grain,
        "stable_columns": summary.stable_columns,
        "repeated_column_rules": summary.repeated_column_rules,
        "inclusive_ranges": summary.inclusive_ranges,
        "quote_spelling_rules": summary.quote_spelling_rules,
        "examples": summary.examples,
        "aliases": summary.aliases,
    }


def render_summary_search_terms(summaries: list[LargeSchemaSummary]) -> list[str]:
    """Flat list of search-index terms extracted from all summaries."""

    terms: list[str] = []
    for summary in summaries:
        terms.extend(
            [
                summary.summary_id,
                summary.purpose,
                summary.grain,
                *summary.stable_columns,
                *summary.repeated_column_rules,
                *summary.inclusive_ranges,
                *summary.quote_spelling_rules,
                *summary.examples,
                *summary.aliases,
            ]
        )
    return terms


def _field_list(label: str, values: list[str]) -> str:
    if not values:
        return ""
    return f"{label}: {', '.join(values)}."


def _join_sentences(parts: list[str]) -> str:
    return " ".join(part.strip() for part in parts if part.strip())


def _section(title: str, values: list[str]) -> list[str]:
    if not values:
        return []
    return [f"{title}:", *[f"- {value}" for value in values]]
