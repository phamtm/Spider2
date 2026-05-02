from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import pandas as pd

from sol01.category_metadata import tier_complexity_summary
from sol01.progress_ui.constants import (
    OUTPUTS_ROOT,
    STATUS_COLORS,
    STATUS_LABELS,
)
from sol01.progress_ui.utils import is_missing_value, normalize_tag_values
from sol01.registry import resolve_llm_call_log_path as resolve_task_llm_call_log_path


def _truncate_text(value: Any, limit: int) -> str:
    if is_missing_value(value):
        return ""
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "\u2026"


def _tier_display(value: Any) -> str:
    if is_missing_value(value):
        return "Uncategorized"
    try:
        return f"Tier {int(float(value))}"
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or "Uncategorized"


def _status_dot_label(status: str) -> str:
    label = STATUS_LABELS.get(str(status).lower(), str(status))
    return f"\u25cf {label}" if label else label


def _status_dot_style(status: object) -> str:
    key = str(status).split(" ", 1)[-1].lower()
    color = STATUS_COLORS.get(key, "#9ca3af")
    return f"color: {color}; font-weight: 600;"


def format_question_option(row: pd.Series | dict[str, Any]) -> str:
    data = row if isinstance(row, dict) else row.to_dict()
    parts = [str(data.get("instance_id") or "")]
    status = str(data.get("status") or "")
    if status:
        parts.append(STATUS_LABELS.get(status, status.title()))
    tier = data.get("primary_tier")
    if not is_missing_value(tier):
        parts.append(_tier_display(tier))
    db = data.get("db")
    if not is_missing_value(db):
        parts.append(str(db))
    instruction = _truncate_text(data.get("instruction"), 80)
    if instruction:
        parts.append(instruction)
    return " | ".join(parts)


def select_question_row(frame: pd.DataFrame, instance_id: str | None) -> dict[str, Any] | None:
    if not instance_id or frame.empty or "instance_id" not in frame.columns:
        return None
    matches = frame.loc[frame["instance_id"] == instance_id]
    if matches.empty:
        return None
    row = matches.iloc[0].to_dict()
    row["status_label"] = STATUS_LABELS.get(
        str(row.get("status") or ""), str(row.get("status") or "")
    )
    tier = row.get("primary_tier")
    row["primary_tier_label"] = _tier_display(tier)
    row["tags_label"] = ", ".join(normalize_tag_values(row.get("tags"))) or "\u2014"
    row["instruction"] = (
        "" if is_missing_value(row.get("instruction")) else str(row.get("instruction"))
    )
    row["note"] = "" if is_missing_value(row.get("note")) else str(row.get("note"))
    row["diagnostics"] = (
        "" if is_missing_value(row.get("diagnostics")) else str(row.get("diagnostics"))
    )
    row["difficulty_notes"] = (
        "" if is_missing_value(row.get("difficulty_notes")) else str(row.get("difficulty_notes"))
    )
    row["db"] = "" if is_missing_value(row.get("db")) else str(row.get("db"))
    row["source_path"] = (
        "" if is_missing_value(row.get("source_path")) else str(row.get("source_path"))
    )
    return row


def resolve_selected_llm_call_log_path(row: dict[str, Any]) -> Path | None:
    if is_missing_value(row.get("run_id")) or is_missing_value(row.get("instance_id")):
        return None
    task_row = {
        "run_id": row["run_id"],
        "instance_id": row["instance_id"],
    }
    for key in ("trace_path", "extra_artifacts"):
        if key in row and not is_missing_value(row.get(key)):
            task_row[key] = row[key]
    try:
        return resolve_task_llm_call_log_path(task_row, outputs_root=OUTPUTS_ROOT)
    except ValueError:
        return None


def _format_llm_call_option(row: dict[str, Any]) -> str:
    sequence = (
        row.get("sequence") if row.get("sequence") is not None else row.get("line_number", "\u2014")
    )
    parts = [
        f"#{sequence}",
        str(row.get("prompt_name") or "\u2014"),
        str(row.get("status") or "\u2014"),
        str(row.get("duration") or "\u2014"),
    ]
    call_id = row.get("call_id")
    if call_id and call_id != "\u2014":
        parts.append(str(call_id))
    return " | ".join(parts)


def build_run_command(dataset_path: Path, source_path: Path) -> str:
    return (
        "uv run streamlit run progress_ui.py -- "
        f"--dataset {shlex.quote(str(dataset_path))} "
        f"--source {shlex.quote(str(source_path))}"
    )


def format_tier_summary(selected_tiers: list[int]) -> str:
    return tier_complexity_summary(selected_tiers)
