from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from sol01.progress_ui.constants import QUESTION_STATUS_ORDER, STATUS_LABELS, TABLE_VISIBLE_ROWS
from sol01.progress_ui.models import Record
from sol01.progress_ui.utils import (
    is_missing_value,
    missing_to_na,
    normalize_tag_values,
    tier_display,
    truncate_text,
)


def _record_timestamp_sort_key(timestamp: datetime | None, fallback_index: int) -> datetime:
    if timestamp is None:
        return datetime.fromtimestamp(fallback_index, tz=UTC)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp


def latest_records(records: list[Record]) -> dict[str, Record]:
    latest: dict[str, Record] = {}
    for _index, record in sorted(
        enumerate(records),
        key=lambda pair: (_record_timestamp_sort_key(pair[1].timestamp, pair[0]), pair[0]),
    ):
        latest[record.instance_id] = record
    return latest


def build_status_frame(
    dataset: pd.DataFrame,
    records: list[Record],
    category_rows: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    latest = latest_records(records)
    ids = list(dataset["instance_id"]) if not dataset.empty else sorted(latest)
    id_set = set(ids)
    rows = []
    metadata = dataset.set_index("instance_id").to_dict("index") if not dataset.empty else {}
    category_rows = category_rows or {}

    for instance_id in ids:
        record = latest.get(instance_id)
        info = metadata.get(instance_id, {})
        category = category_rows.get(instance_id)
        rows.append(
            {
                "instance_id": instance_id,
                "status": record.status if record else "unanswered",
                "score": missing_to_na(record.score if record else pd.NA),
                "timestamp": missing_to_na(record.timestamp if record else pd.NA),
                "run_id": missing_to_na(record.run_id if record else pd.NA),
                "db": missing_to_na(
                    (record.db or info.get("db_id")) if record else info.get("db_id")
                ),
                "instruction": info.get("instruction", ""),
                "note": missing_to_na(record.note if record else pd.NA),
                "source_path": missing_to_na(record.source_path if record else pd.NA),
                "diagnostics": missing_to_na(record.diagnostics if record else pd.NA),
                "primary_tier": missing_to_na(category["primary_tier"] if category else pd.NA),
                "tags": list(category["tags"]) if category else [],
                "difficulty_notes": missing_to_na(
                    category["difficulty_notes"] if category else pd.NA
                ),
                "category_available": category is not None,
            }
        )

    extras = [instance_id for instance_id in latest if instance_id not in id_set]
    for instance_id in sorted(extras):
        record = latest[instance_id]
        category = category_rows.get(instance_id)
        rows.append(
            {
                "instance_id": instance_id,
                "status": record.status,
                "score": missing_to_na(record.score),
                "timestamp": missing_to_na(record.timestamp),
                "run_id": missing_to_na(record.run_id),
                "db": missing_to_na(record.db),
                "instruction": "",
                "note": missing_to_na(record.note),
                "source_path": missing_to_na(record.source_path),
                "diagnostics": missing_to_na(record.diagnostics),
                "primary_tier": missing_to_na(category["primary_tier"] if category else pd.NA),
                "tags": list(category["tags"]) if category else [],
                "difficulty_notes": missing_to_na(
                    category["difficulty_notes"] if category else pd.NA
                ),
                "category_available": category is not None,
            }
        )

    return pd.DataFrame(rows)


def prepare_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    if "timestamp" in display.columns:
        display["timestamp"] = display["timestamp"].apply(
            lambda value: "" if is_missing_value(value) else str(value)
        )
    if "tags" in display.columns:
        display["tags"] = display["tags"].apply(
            lambda value: (
                ", ".join(normalize_tag_values(value)) if not is_missing_value(value) else ""
            )
        )
    if "primary_tier" in display.columns:
        display["primary_tier"] = display["primary_tier"].apply(
            lambda value: f"Tier {int(value)}" if pd.notna(value) else ""
        )
    return display


def _tier_sort_value(value: Any) -> int:
    if is_missing_value(value):
        return 9999
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 9999


def prepare_question_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "instance_id",
                "status",
                "primary_tier",
                "tags",
                "db",
                "instruction",
                "note",
                "diagnostics",
                "score",
            ]
        )

    display = frame.copy()
    if "status" not in display.columns:
        display["status"] = ""
    if "primary_tier" not in display.columns:
        display["primary_tier"] = pd.NA
    if "tags" not in display.columns:
        display["tags"] = [[] for _ in range(len(display))]
    if "db" not in display.columns:
        display["db"] = ""
    if "instruction" not in display.columns:
        display["instruction"] = ""
    if "note" not in display.columns:
        display["note"] = ""
    if "diagnostics" not in display.columns:
        display["diagnostics"] = ""
    if "score" not in display.columns:
        display["score"] = pd.NA

    display["status_sort"] = (
        display["status"]
        .astype(str)
        .str.lower()
        .map({status: index for index, status in enumerate(QUESTION_STATUS_ORDER)})
    )
    display["status_sort"] = display["status_sort"].fillna(len(QUESTION_STATUS_ORDER))
    display["tier_sort"] = display["primary_tier"].apply(_tier_sort_value)
    display["status"] = display["status"].astype(str).map(STATUS_LABELS).fillna(display["status"])
    display["primary_tier"] = display["primary_tier"].apply(tier_display)
    display["tags"] = display["tags"].apply(
        lambda value: ", ".join(normalize_tag_values(value)) or "\u2014"
    )
    display["instruction"] = display["instruction"].apply(
        lambda value: truncate_text(value, 120) or "\u2014"
    )
    display["note"] = display["note"].apply(lambda value: truncate_text(value, 80) or "\u2014")
    display["diagnostics"] = display["diagnostics"].apply(
        lambda value: truncate_text(value, 120) or "\u2014"
    )
    display["db"] = display["db"].apply(lambda value: truncate_text(value, 48) or "\u2014")

    display = display.sort_values(
        by=["status_sort", "tier_sort", "instance_id"],
        kind="stable",
    ).reset_index(drop=True)
    return display[
        [
            "instance_id",
            "status",
            "primary_tier",
            "tags",
            "db",
            "instruction",
            "note",
            "diagnostics",
            "score",
        ]
    ]


def dataframe_height(row_count: int) -> int:
    visible_rows = max(1, min(row_count, TABLE_VISIBLE_ROWS))
    return visible_rows * 24 + 48


def make_progress_frame_for_ids(
    records: list[Record],
    total_questions: int,
    selected_instance_ids: set[str] | None = None,
) -> pd.DataFrame:
    if selected_instance_ids is not None:
        records = [record for record in records if record.instance_id in selected_instance_ids]
    if not records:
        return pd.DataFrame(
            columns=[
                "x",
                "answered_pct",
                "correct_pct",
                "answered",
                "correct",
                "incorrect",
            ]
        )

    sorted_records = sorted(
        enumerate(records),
        key=lambda pair: pair[1].timestamp or datetime.fromtimestamp(pair[0], tz=UTC),
    )

    state: dict[str, str] = {}
    rows = []

    answered, correct, incorrect = 0, 0, 0

    for index, (_, record) in enumerate(sorted_records, start=1):
        prev_status = state.get(record.instance_id, "unanswered")
        new_status = record.status

        if prev_status != new_status:
            if prev_status != "unanswered":
                answered -= 1
            if prev_status == "correct":
                correct -= 1
            if prev_status == "incorrect":
                incorrect -= 1

            if new_status != "unanswered":
                answered += 1
            if new_status == "correct":
                correct += 1
            if new_status == "incorrect":
                incorrect += 1

            state[record.instance_id] = new_status

        denominator = max(total_questions, len(state), 1)
        rows.append(
            {
                "x": record.timestamp or index,
                "answered_pct": answered / denominator * 100,
                "correct_pct": correct / denominator * 100,
                "answered": answered,
                "correct": correct,
                "incorrect": incorrect,
            }
        )
    return pd.DataFrame(rows)


def prepare_debug_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "instance_id",
        "status",
        "score",
        "timestamp",
        "run_id",
        "db",
        "instruction",
        "note",
        "diagnostics",
        "source_path",
        "primary_tier",
        "tags",
        "difficulty_notes",
        "category_available",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    debug = frame.copy()
    if "timestamp" in debug.columns:
        debug["timestamp"] = debug["timestamp"].apply(
            lambda value: "" if is_missing_value(value) else str(value)
        )
    if "primary_tier" in debug.columns:
        debug["primary_tier"] = debug["primary_tier"].apply(tier_display)
    if "tags" in debug.columns:
        debug["tags"] = debug["tags"].apply(
            lambda value: ", ".join(normalize_tag_values(value)) or "\u2014"
        )
    for column in ("instruction", "note", "db", "source_path", "difficulty_notes"):
        if column in debug.columns:
            debug[column] = debug[column].apply(
                lambda value: "" if is_missing_value(value) else str(value)
            )
    if "diagnostics" in debug.columns:
        debug["diagnostics"] = debug["diagnostics"].apply(
            lambda value: "" if is_missing_value(value) else truncate_text(value, 120)
        )
    if "category_available" not in debug.columns:
        debug["category_available"] = False

    for column in columns:
        if column not in debug.columns:
            debug[column] = "" if column != "category_available" else False

    return debug[columns]
