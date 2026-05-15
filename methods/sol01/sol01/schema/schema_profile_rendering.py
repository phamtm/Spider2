"""Shared renderers for generated schema-profile facts."""

from __future__ import annotations

from sol01.models import FamilyProfile, TableProfile

SchemaProfile = TableProfile | FamilyProfile


def render_schema_profile_text(profiles: list[SchemaProfile]) -> str:
    """Sentence-based summary block for planner-facing schema evidence."""

    lines: list[str] = []
    for profile in profiles:
        lines.extend(
            [
                f"Schema profile: {profile.profile_id}.",
                f"Abstraction kind: {profile.abstraction_kind}.",
                f"Summary: {profile.compact_semantic_summary}",
                f"Grain: {profile.grain_hint}",
                _field_list("Key columns", profile.key_columns),
                _field_list("Time columns", profile.time_columns),
                _field_list("Measure columns", profile.measure_columns),
                _field_list("Dimension columns", profile.dimension_columns),
                _field_list("Repeated templates", profile.repeated_column_templates),
                _field_list("Join anchors", profile.join_anchors),
                _field_list("Naming rules", profile.naming_rules),
                _field_list("Aliases", profile.aliases),
                _field_list("Theme terms", profile.theme_terms),
                _field_list("Caveats", profile.caveats),
            ]
        )
    return _join_sentences(lines)


def render_schema_profile_lines(profile: SchemaProfile) -> list[str]:
    """Line-list summary block for SQL reference prompts."""

    lines = [
        f"Schema profile: {profile.profile_id}",
        f"Abstraction kind: {profile.abstraction_kind}",
        f"Summary: {profile.compact_semantic_summary}",
        f"Grain: {profile.grain_hint}",
        "Use exact table and column spellings from the cards below.",
    ]
    lines.extend(_section("Key columns", profile.key_columns))
    lines.extend(_section("Time columns", profile.time_columns))
    lines.extend(_section("Measure columns", profile.measure_columns))
    lines.extend(_section("Dimension columns", profile.dimension_columns))
    lines.extend(_section("Repeated templates", profile.repeated_column_templates))
    lines.extend(_section("Join anchors", profile.join_anchors))
    lines.extend(_section("Naming rules", profile.naming_rules))
    lines.extend(_section("Caveats", profile.caveats))
    return lines


def render_schema_profile_payload(profile: SchemaProfile) -> dict[str, object]:
    """Dict payload stored in schema object metadata."""

    return {
        "profile_id": profile.profile_id,
        "abstraction_kind": profile.abstraction_kind,
        "covered_tables": profile.covered_tables,
        "grain_hint": profile.grain_hint,
        "key_columns": profile.key_columns,
        "time_columns": profile.time_columns,
        "measure_columns": profile.measure_columns,
        "dimension_columns": profile.dimension_columns,
        "repeated_column_templates": profile.repeated_column_templates,
        "join_anchors": profile.join_anchors,
        "naming_rules": profile.naming_rules,
        "compact_semantic_summary": profile.compact_semantic_summary,
        "aliases": profile.aliases,
        "theme_terms": profile.theme_terms,
        "confidence": profile.confidence,
        "caveats": profile.caveats,
    }


def render_schema_profile_search_terms(profiles: list[SchemaProfile]) -> list[str]:
    """Flat search-index terms extracted from generated profiles."""

    terms: list[str] = []
    for profile in profiles:
        terms.extend(
            [
                profile.profile_id,
                profile.abstraction_kind,
                profile.compact_semantic_summary,
                profile.grain_hint,
                *profile.key_columns,
                *profile.time_columns,
                *profile.measure_columns,
                *profile.dimension_columns,
                *profile.repeated_column_templates,
                *profile.join_anchors,
                *profile.naming_rules,
                *profile.aliases,
                *profile.theme_terms,
                *profile.caveats,
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
