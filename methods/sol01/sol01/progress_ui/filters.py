from __future__ import annotations

import pandas as pd


def apply_frame_filters(
    frame: pd.DataFrame,
    *,
    search: str = "",
    selected_status: list[str] | None = None,
    selected_tiers: list[int] | None = None,
    selected_tags: list[str] | None = None,
) -> pd.DataFrame:
    filtered = frame

    if search:
        search_scope, search_term = _parse_search_scope(search)
        search_columns = _search_columns_for_scope(filtered, search_scope)
        if not search_columns:
            return filtered.iloc[0:0]
        search_mask = (
            filtered[search_columns]
            .astype(str)
            .apply(lambda col: col.str.contains(search_term, case=False, na=False))
            .any(axis=1)
        )
        filtered = filtered[search_mask]

    if selected_status:
        filtered = filtered[filtered["status"].isin(selected_status)]

    category_filters_active = bool(selected_tiers or selected_tags)
    if category_filters_active and "category_available" in filtered.columns:
        filtered = filtered[filtered["category_available"]]

    if selected_tiers:
        filtered = filtered[filtered["primary_tier"].isin(selected_tiers)]

    if selected_tags:
        filtered = filtered[
            filtered["tags"].apply(lambda tags: all(tag in (tags or []) for tag in selected_tags))
        ]

    return filtered


def should_show_all_questions(
    selected_tiers: list[int] | None, selected_tags: list[str] | None
) -> bool:
    return bool(selected_tiers or selected_tags)


def _parse_search_scope(search: str) -> tuple[str | None, str]:
    text = search.strip()
    if ":" not in text:
        return None, text

    scope, term = text.split(":", 1)
    scope = scope.strip().lower()
    term = term.strip()
    if scope in {"id", "db", "instruction", "note", "tier", "tags", "notes"} and term:
        return scope, term
    return None, text


def _search_columns_for_scope(frame: pd.DataFrame, scope: str | None) -> list[str]:
    if scope == "id":
        return [column for column in ("instance_id",) if column in frame.columns]
    if scope == "db":
        return [column for column in ("db",) if column in frame.columns]
    if scope == "instruction":
        return [column for column in ("instruction",) if column in frame.columns]
    if scope == "note":
        return [column for column in ("note", "difficulty_notes") if column in frame.columns]
    if scope == "notes":
        return [column for column in ("note", "difficulty_notes") if column in frame.columns]
    if scope == "tier":
        return [column for column in ("primary_tier",) if column in frame.columns]
    if scope == "tags":
        return [column for column in ("tags",) if column in frame.columns]

    return [
        column
        for column in (
            "instance_id",
            "db",
            "instruction",
            "note",
            "primary_tier",
            "tags",
            "difficulty_notes",
        )
        if column in frame.columns
    ]
