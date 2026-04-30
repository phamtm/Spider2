"""Local Streamlit dashboard for Spider progress.

Run from methods/sol01:
    uv run streamlit run progress_ui.py

Optional CLI defaults:
    uv run streamlit run progress_ui.py -- --source outputs/registry/latest.json
"""

from __future__ import annotations

import argparse
import html
import json
import shlex
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sol01.category_metadata import CategoryMetadataValidationError, load_category_metadata_map

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "spider2-snow" / "spider2-snow.jsonl"
DEFAULT_SOURCE = ROOT / "methods" / "sol01" / "outputs" / "registry" / "latest.json"

STATUS_ORDER = ("correct", "incorrect", "answered", "unanswered")
STATUS_LABELS = {
    "correct": "Correct",
    "incorrect": "Incorrect",
    "answered": "Answered",
    "unanswered": "Unanswered",
}
STATUS_COLORS = {
    "correct": "#22c55e",
    "incorrect": "#ef4444",
    "answered": "#64748b",
    "unanswered": "#1f2937",
}

CORRECT_COLOR = STATUS_COLORS["correct"]
INCORRECT_COLOR = STATUS_COLORS["incorrect"]
ANSWERED_COLOR = STATUS_COLORS["answered"]

CHART_HEIGHT = 440
TABLE_ROW_HEIGHT = 24
TABLE_VISIBLE_ROWS = 50
TABLE_HEIGHT = TABLE_VISIBLE_ROWS * TABLE_ROW_HEIGHT + 48
SECTION_GAP = 24
QUESTION_STATUS_ORDER = ("unanswered", "incorrect", "answered", "correct")

TIER_COMPLEXITY = {
    1: "Simple lookup or single-step aggregate. Usually one table and one obvious filter.",
    2: "Straightforward multi-step query. Usually one join, modest filtering, or a simple grouped result.",
    3: "Multi-step reasoning. Common examples are ranking, window functions, temporal rollups, cohort logic, or external notes.",
    4: "Hard query. Usually mixes several advanced patterns, such as nested aggregation, geospatial logic, multi-hop joins, or multiple time-based transformations.",
    5: "Harder multi-step query. Usually combines joins, filtering, ranking, or grouped comparisons.",
    6: "Advanced reasoning. Usually adds deeper joins, temporal logic, or cross-table aggregation.",
    7: "Advanced multi-step query. Often needs nested ranking, cohort-style analysis, or multi-scale rollups.",
    8: "Very hard query. Usually mixes multiple advanced patterns such as joins, temporal windows, or nested aggregation.",
    9: "Expert-level query. Often requires layered filters, ranking passes, or compound grouping logic.",
    10: "Expert-level reasoning. Usually adds time-series windows, moving calculations, or external constraints.",
    11: "Very complex query. Often involves hierarchical or recursive style reasoning with several transformations.",
    12: "Hardest queries in the current set. Usually combine several advanced steps, such as nested state, cumulative allocation, or forecasting-style logic.",
}


@dataclass(frozen=True)
class Record:
    instance_id: str
    status: str
    score: float | None
    timestamp: datetime | None
    run_id: str | None
    db: str | None
    note: str | None
    source_path: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    args, _ = parser.parse_known_args()
    return args


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        pass
    return rows


@st.cache_data(ttl=3600)  # Cache dataset, assume it rarely changes
def read_dataset(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        return pd.DataFrame(columns=["instance_id", "instruction", "db_id"])

    try:
        # Vastly faster vectorized loading instead of looping json.loads
        frame = pd.read_json(path_obj, lines=True)
    except ValueError:
        return pd.DataFrame(columns=["instance_id", "instruction", "db_id"])

    if "instance_id" not in frame.columns:
        return pd.DataFrame(columns=["instance_id", "instruction", "db_id"])

    for column in ("instruction", "db_id"):
        if column not in frame.columns:
            frame[column] = ""

    return frame[["instance_id", "instruction", "db_id"]].drop_duplicates("instance_id")


def parse_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=UTC)

    text = str(value).strip()
    formats = (
        "%Y%m%dT%H%M%S.%fZ",
        "%Y%m%dT%H%M%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def as_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_instance_id(item: dict[str, Any], fallback: str | None = None) -> str | None:
    for key in ("instance_id", "id", "question_id", "task_id", "output"):
        value = item.get(key)
        if value:
            return str(value).removesuffix(".csv").removesuffix(".sql")
    return fallback


def classify(item: dict[str, Any]) -> tuple[str, float | None]:
    score = as_float(item.get("score"))
    passed = item.get("passed")
    status = str(item.get("status") or item.get("eval_status") or "").lower()

    if score is not None:
        return ("correct" if score >= 1 else "incorrect"), score
    if passed is True:
        return "correct", 1.0
    if passed is False:
        return "incorrect", 0.0
    if status in {"pass", "passed", "correct", "success"} and item.get("eval_status") != "failed":
        return "correct", 1.0
    if status in {"fail", "failed", "incorrect", "eval_failed", "error"}:
        return "incorrect", 0.0
    if status in {"solver_failed", "not_answered", "missing"} and not (
        item.get("csv_path") or item.get("sql_path")
    ):
        return "unanswered", None
    if item.get("eval_error") or item.get("failure_reason"):
        return "incorrect", 0.0
    if item.get("csv_path") or item.get("sql_path") or item.get("csv_present"):
        return "answered", None
    return "answered", None


def normalize_item(
    item: dict[str, Any], source_path: Path, fallback_id: str | None = None
) -> Record | None:
    instance_id = find_instance_id(item, fallback=fallback_id)
    if not instance_id:
        return None

    status, score = classify(item)
    timestamp = parse_timestamp(item.get("timestamp") or item.get("generated_at"))
    note = item.get("failure_reason") or item.get("eval_error") or item.get("solver_status")
    return Record(
        instance_id=instance_id,
        status=status,
        score=score,
        timestamp=timestamp,
        run_id=item.get("run_id"),
        db=item.get("db") or item.get("db_id"),
        note=str(note) if note else None,
        source_path=str(source_path),
    )


def records_from_json(path: Path) -> list[Record]:
    data = read_json(path)
    rows: list[dict[str, Any]]
    if isinstance(data, list):
        rows = [row for row in data if isinstance(row, dict)]
    elif isinstance(data, dict):
        if isinstance(data.get("task_results"), list):
            rows = data["task_results"]
        elif isinstance(data.get("per_instance"), list):
            rows = data["per_instance"]
        elif isinstance(data.get("instance_scores"), dict):
            rows = [
                {
                    "instance_id": instance_id,
                    "score": score,
                    "generated_at": data.get("generated_at"),
                }
                for instance_id, score in data["instance_scores"].items()
            ]
        else:
            rows = [data]
    else:
        rows = []
    return [record for row in rows if (record := normalize_item(row, path))]


def records_from_csv(path: Path) -> list[Record]:
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []

    if frame.empty:
        return []

    if len(frame.columns) == 1 and frame.columns[0] == "output":
        frame["score"] = 1

    records = []
    for item in frame.to_dict("records"):
        records.append(normalize_item(item, path))
    return [record for record in records if record]


def records_from_file(path: Path) -> list[Record]:
    if path.suffix == ".jsonl":
        return [record for row in read_jsonl(path) if (record := normalize_item(row, path))]
    if path.suffix == ".json":
        return records_from_json(path)
    if path.suffix == ".csv":
        return records_from_csv(path)
    if path.suffix in {".sql"}:
        return [
            Record(
                instance_id=path.stem,
                status="answered",
                score=None,
                timestamp=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
                run_id=None,
                db=None,
                note="SQL present, not evaluated",
                source_path=str(path),
            )
        ]
    return []


def discover_result_files(path: Path) -> list[Path]:
    preferred = [
        path / "registry" / "latest.json",
        path / "registry" / "task_results.jsonl",
        path / "eval" / "summary.json",
        path / "eval" / "per_instance.jsonl",
        path / "summary.json",
        path / "per_instance.jsonl",
    ]
    found = [candidate for candidate in preferred if candidate.exists()]
    if found:
        return found

    patterns = (
        "**/eval/summary.json",
        "**/eval/per_instance.jsonl",
        "*.json",
        "*.jsonl",
        "*.csv",
        "*.sql",
    )
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path.glob(pattern))
    return sorted(set(files))


@st.cache_data(ttl=5)  # Cache but refresh every 5 seconds to track live runs
def load_records(source_str: str) -> list[Record]:
    source = Path(source_str)
    if not source.exists():
        return []
    if source.is_dir():
        records: list[Record] = []
        for file_path in discover_result_files(source):
            records.extend(records_from_file(file_path))
        return records
    return records_from_file(source)


@st.cache_data(ttl=3600)  # Cache metadata alongside the dataset for a stable sidebar
def load_category_metadata_rows(dataset_path: str) -> dict[str, dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        return {}

    try:
        metadata_map = load_category_metadata_map(dataset_path=path)
    except (CategoryMetadataValidationError, FileNotFoundError, OSError, ValueError):
        return {}

    return {
        instance_id: {
            "primary_tier": record.primary_tier,
            "tags": list(record.tags),
            "difficulty_notes": record.difficulty_notes,
        }
        for instance_id, record in metadata_map.items()
    }


def latest_records(records: list[Record]) -> dict[str, Record]:
    latest: dict[str, Record] = {}
    for index, record in enumerate(records):
        current = latest.get(record.instance_id)
        current_time = current.timestamp if current else None
        record_time = record.timestamp or datetime.fromtimestamp(index, tz=UTC)
        if current is None or current_time is None or record_time >= current_time:
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
                "score": record.score if record else None,
                "timestamp": record.timestamp if record else None,
                "run_id": record.run_id if record else None,
                "db": record.db or info.get("db_id") if record else info.get("db_id"),
                "instruction": info.get("instruction", ""),
                "note": record.note if record else None,
                "source_path": record.source_path if record else None,
                "primary_tier": category["primary_tier"] if category else None,
                "tags": list(category["tags"]) if category else [],
                "difficulty_notes": category["difficulty_notes"] if category else None,
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
                "score": record.score,
                "timestamp": record.timestamp,
                "run_id": record.run_id,
                "db": record.db,
                "instruction": "",
                "note": record.note,
                "source_path": record.source_path,
                "primary_tier": category["primary_tier"] if category else None,
                "tags": list(category["tags"]) if category else [],
                "difficulty_notes": category["difficulty_notes"] if category else None,
                "category_available": category is not None,
            }
        )

    return pd.DataFrame(rows)


def _status_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = {"correct": 0, "incorrect": 0, "answered": 0, "unanswered": 0}
    if "status" not in frame.columns:
        return counts

    value_counts = frame["status"].astype(str).str.lower().value_counts(dropna=False)
    for status in counts:
        counts[status] = int(value_counts.get(status, 0))
    return counts


def _summary_rates(total: int, answered: int, correct: int, incorrect: int, unanswered: int) -> dict[str, float]:
    coverage_pct = answered / total * 100 if total else 0.0
    accuracy_pct = correct / answered * 100 if answered else 0.0
    return {
        "coverage_pct": coverage_pct,
        "accuracy_pct": accuracy_pct,
        "answered_pct": answered / total * 100 if total else 0.0,
        "correct_pct": correct / total * 100 if total else 0.0,
        "incorrect_pct": incorrect / total * 100 if total else 0.0,
        "unanswered_pct": unanswered / total * 100 if total else 0.0,
    }


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, (bool, int)) else False


def compute_overall_summary(frame: pd.DataFrame) -> dict[str, Any]:
    total = int(len(frame))
    counts = _status_counts(frame)
    answered = total - counts["unanswered"]
    summary = {
        "total": total,
        "answered": answered,
        "correct": counts["correct"],
        "incorrect": counts["incorrect"],
        "unanswered": counts["unanswered"],
    }
    return summary | _summary_rates(
        total,
        answered,
        counts["correct"],
        counts["incorrect"],
        counts["unanswered"],
    )


def _summary_columns(key_name: str, value_name: str) -> list[str]:
    return [
        key_name,
        value_name,
        "total",
        "answered",
        "correct",
        "incorrect",
        "unanswered",
        "coverage_pct",
        "accuracy_pct",
        "answered_pct",
        "correct_pct",
        "incorrect_pct",
        "unanswered_pct",
    ]


def _build_summary_row(label: Any, value_name: str, group: pd.DataFrame) -> dict[str, Any]:
    counts = _status_counts(group)
    total = int(len(group))
    answered = total - counts["unanswered"]
    rates = _summary_rates(
        total,
        answered,
        counts["correct"],
        counts["incorrect"],
        counts["unanswered"],
    )
    return {
        value_name: str(label),
        "total": total,
        "answered": answered,
        "correct": counts["correct"],
        "incorrect": counts["incorrect"],
        "unanswered": counts["unanswered"],
        **rates,
    }


def compute_tier_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_summary_columns("primary_tier", "tier_label"))

    rows: list[dict[str, Any]] = []
    tier_series = (
        frame["primary_tier"]
        if "primary_tier" in frame.columns
        else pd.Series([pd.NA] * len(frame), index=frame.index)
    )
    for tier_value, group in frame.assign(_tier=tier_series).groupby("_tier", dropna=False):
        if pd.isna(tier_value):
            rows.append({"primary_tier": pd.NA, **_build_summary_row("Uncategorized", "tier_label", group)})
            continue

        try:
            tier_number = int(tier_value)
        except (TypeError, ValueError):
            tier_number = tier_value
        rows.append({"primary_tier": tier_number, **_build_summary_row(f"Tier {tier_number}", "tier_label", group)})

    result = pd.DataFrame(rows, columns=_summary_columns("primary_tier", "tier_label"))
    if result.empty:
        return result

    sort_key = result["primary_tier"].apply(
        lambda value: (1, 0, "") if pd.isna(value) else (0, 0, int(value)) if str(value).isdigit() else (0, 1, str(value))
    )
    result = result.assign(_tier_sort=sort_key).sort_values(
        by=["_tier_sort", "tier_label"],
        kind="stable",
    )
    return result.drop(columns="_tier_sort").reset_index(drop=True)


def _normalize_tag_values(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(tag) for tag in value if str(tag)]
    if _is_missing_value(value):
        return []
    return [str(value)]


def compute_tag_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_summary_columns("tag", "tag_label"))

    tag_frame = frame.copy()
    if "tags" not in tag_frame.columns:
        tag_frame["tags"] = [[] for _ in range(len(tag_frame))]

    tag_frame["tag"] = tag_frame["tags"].apply(_normalize_tag_values)
    tag_frame["tag"] = tag_frame["tag"].apply(lambda tags: tags or ["(no tags)"])
    tag_frame = tag_frame.explode("tag", ignore_index=True)

    rows = []
    for tag_value, group in tag_frame.groupby("tag", dropna=False):
        rows.append({"tag": str(tag_value), **_build_summary_row(tag_value, "tag_label", group)})

    result = pd.DataFrame(rows, columns=_summary_columns("tag", "tag_label"))
    if result.empty:
        return result

    return result.sort_values(
        by=["unanswered", "incorrect", "total", "tag_label"],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def recommend_focus(frame: pd.DataFrame) -> dict[str, Any]:
    summary = compute_overall_summary(frame)
    if summary["total"] == 0:
        return {
            "kind": "empty",
            "title": "No questions loaded",
            "detail": "Load results to get a focus recommendation.",
            "count": 0,
            "primary_tier": None,
            "tag": None,
            "coverage_pct": 0.0,
            "accuracy_pct": 0.0,
        }

    if summary["answered"] < 10:
        tier_summary = compute_tier_summary(frame)
        low_tier_unanswered = tier_summary[
            tier_summary["primary_tier"].isin([1, 2, 3]) & (tier_summary["unanswered"] > 0)
        ]
        if not low_tier_unanswered.empty:
            row = low_tier_unanswered.sort_values(
                by=["primary_tier", "unanswered", "total"],
                ascending=[True, False, False],
                kind="stable",
            ).iloc[0]
            tier_value = None if pd.isna(row["primary_tier"]) else int(row["primary_tier"])
            return {
                "kind": "unanswered",
                "title": f"Clear tier {row['tier_label']}",
                "detail": (
                    f"{int(row['unanswered'])} questions in {row['tier_label']} are still unanswered."
                ),
                "count": int(row["unanswered"]),
                "primary_tier": tier_value,
                "tag": None,
                "coverage_pct": float(row["coverage_pct"]),
                "accuracy_pct": float(row["accuracy_pct"]),
            }

        return {
            "kind": "baseline",
            "title": "Build a baseline",
            "detail": (
                f"Only {summary['answered']} of {summary['total']} questions are answered. "
                "Get more coverage before tuning accuracy."
            ),
            "count": summary["answered"],
            "primary_tier": None,
            "tag": None,
            "coverage_pct": summary["coverage_pct"],
            "accuracy_pct": summary["accuracy_pct"],
        }

    tier_summary = compute_tier_summary(frame)
    incorrect_tiers = tier_summary[tier_summary["incorrect"] > 0]
    if not incorrect_tiers.empty:
        row = incorrect_tiers.sort_values(
            by=["incorrect", "unanswered", "primary_tier"],
            ascending=[False, False, True],
            kind="stable",
        ).iloc[0]
        tier_value = None if pd.isna(row["primary_tier"]) else int(row["primary_tier"])
        return {
            "kind": "incorrect",
            "title": f"Fix tier {row['tier_label']} answers",
            "detail": (
                f"{int(row['incorrect'])} incorrect answers remain in {row['tier_label']}. "
                "Start there."
            ),
            "count": int(row["incorrect"]),
            "primary_tier": tier_value,
            "tag": None,
            "coverage_pct": float(row["coverage_pct"]),
            "accuracy_pct": float(row["accuracy_pct"]),
        }

    unanswered_tiers = tier_summary[tier_summary["unanswered"] > 0]
    if not unanswered_tiers.empty:
        row = unanswered_tiers.sort_values(
            by=["primary_tier", "unanswered", "total"],
            ascending=[True, False, False],
            kind="stable",
        ).iloc[0]
        tier_value = None if pd.isna(row["primary_tier"]) else int(row["primary_tier"])
        return {
            "kind": "unanswered",
            "title": f"Clear tier {row['tier_label']}",
            "detail": (
                f"{int(row['unanswered'])} questions in {row['tier_label']} are still unanswered."
            ),
            "count": int(row["unanswered"]),
            "primary_tier": tier_value,
            "tag": None,
            "coverage_pct": float(row["coverage_pct"]),
            "accuracy_pct": float(row["accuracy_pct"]),
        }

    return {
        "kind": "complete",
        "title": "No urgent work left",
        "detail": "All questions in the current slice are answered and correct.",
        "count": 0,
        "primary_tier": None,
        "tag": None,
        "coverage_pct": summary["coverage_pct"],
        "accuracy_pct": summary["accuracy_pct"],
    }


def prepare_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    if "timestamp" in display.columns:
        display["timestamp"] = display["timestamp"].apply(lambda value: "" if _is_missing_value(value) else str(value))
    if "tags" in display.columns:
        display["tags"] = display["tags"].apply(
            lambda value: ", ".join(_normalize_tag_values(value)) if not _is_missing_value(value) else ""
        )
    if "primary_tier" in display.columns:
        display["primary_tier"] = display["primary_tier"].apply(
            lambda value: f"Tier {int(value)}" if pd.notna(value) else ""
        )
    return display


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
        debug["timestamp"] = debug["timestamp"].apply(lambda value: "" if _is_missing_value(value) else str(value))
    if "primary_tier" in debug.columns:
        debug["primary_tier"] = debug["primary_tier"].apply(_tier_display)
    if "tags" in debug.columns:
        debug["tags"] = debug["tags"].apply(lambda value: ", ".join(_normalize_tag_values(value)) or "—")
    for column in ("instruction", "note", "db", "source_path", "difficulty_notes"):
        if column in debug.columns:
            debug[column] = debug[column].apply(lambda value: "" if _is_missing_value(value) else str(value))
    if "category_available" not in debug.columns:
        debug["category_available"] = False

    for column in columns:
        if column not in debug.columns:
            debug[column] = "" if column != "category_available" else False

    return debug[columns]


def _truncate_text(value: Any, limit: int) -> str:
    if _is_missing_value(value):
        return ""
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _tier_display(value: Any) -> str:
    if _is_missing_value(value):
        return "Uncategorized"
    try:
        return f"Tier {int(float(value))}"
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or "Uncategorized"


def _tier_sort_value(value: Any) -> int:
    if _is_missing_value(value):
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
    if "score" not in display.columns:
        display["score"] = pd.NA

    display["status_sort"] = display["status"].astype(str).str.lower().map(
        {status: index for index, status in enumerate(QUESTION_STATUS_ORDER)}
    )
    display["status_sort"] = display["status_sort"].fillna(len(QUESTION_STATUS_ORDER))
    display["tier_sort"] = display["primary_tier"].apply(_tier_sort_value)
    display["status"] = display["status"].astype(str).map(STATUS_LABELS).fillna(display["status"])
    display["primary_tier"] = display["primary_tier"].apply(_tier_display)
    display["tags"] = display["tags"].apply(
        lambda value: ", ".join(_normalize_tag_values(value)) or "—"
    )
    display["instruction"] = display["instruction"].apply(lambda value: _truncate_text(value, 120) or "—")
    display["note"] = display["note"].apply(lambda value: _truncate_text(value, 80) or "—")
    display["db"] = display["db"].apply(lambda value: _truncate_text(value, 48) or "—")

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
            "score",
        ]
    ]


def format_question_option(row: pd.Series | dict[str, Any]) -> str:
    data = row if isinstance(row, dict) else row.to_dict()
    parts = [str(data.get("instance_id") or "")]
    status = str(data.get("status") or "")
    if status:
        parts.append(STATUS_LABELS.get(status, status.title()))
    tier = data.get("primary_tier")
    if not _is_missing_value(tier):
        parts.append(_tier_display(tier))
    db = data.get("db")
    if db:
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
    row["status_label"] = STATUS_LABELS.get(str(row.get("status") or ""), str(row.get("status") or ""))
    tier = row.get("primary_tier")
    row["primary_tier_label"] = _tier_display(tier)
    row["tags_label"] = ", ".join(_normalize_tag_values(row.get("tags"))) or "—"
    row["instruction"] = str(row.get("instruction") or "")
    row["note"] = str(row.get("note") or "")
    row["difficulty_notes"] = str(row.get("difficulty_notes") or "")
    row["db"] = str(row.get("db") or "")
    row["source_path"] = str(row.get("source_path") or "")
    return row


def render_status_legend() -> None:
    items = "".join(
        f"""
        <span class="status-chip">
            <span class="status-swatch" style="background:{STATUS_COLORS[status]}"></span>
            {STATUS_LABELS[status]}
        </span>
        """
        for status in STATUS_ORDER
    )
    st.markdown(
        f"""
        <style>
        .status-legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 0 0 12px 0;
        }}
        .status-chip {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.9rem;
        }}
        .status-swatch {{
            width: 10px;
            height: 10px;
            border-radius: 999px;
            display: inline-block;
        }}
        </style>
        <div class="status-legend">{items}</div>
        """,
        unsafe_allow_html=True,
    )


def render_question_detail(row: dict[str, Any] | None) -> None:
    if not row:
        st.info("Select a question to see its details.")
        return

    st.subheader("Selected question")
    detail_cols = st.columns(4)
    detail_cols[0].metric("Status", row["status_label"])
    detail_cols[1].metric("Tier", row["primary_tier_label"])
    detail_cols[2].metric("DB", row["db"] or "—")
    detail_cols[3].metric("Score", "" if pd.isna(row.get("score")) else f"{row['score']}")

    st.markdown(f"**Question**\n\n{row.get('instruction') or '—'}")
    st.markdown(f"**Tags**\n\n{row['tags_label']}")

    extra_cols = st.columns(2)
    extra_cols[0].markdown(f"**Note**\n\n{row['note'] or '—'}")
    extra_cols[1].markdown(f"**Difficulty notes**\n\n{row['difficulty_notes'] or '—'}")

    st.markdown(f"**Source**: `{row['source_path'] or '—'}`")


def build_run_command(dataset_path: Path, source_path: Path) -> str:
    return (
        "uv run streamlit run progress_ui.py -- "
        f"--dataset {shlex.quote(str(dataset_path))} "
        f"--source {shlex.quote(str(source_path))}"
    )


def should_show_all_questions(selected_tiers: list[int] | None, selected_tags: list[str] | None) -> bool:
    return bool(selected_tiers or selected_tags)


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
        search_columns = [
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
            if column in filtered.columns
        ]
        search_mask = (
            filtered[search_columns]
            .astype(str)
            .apply(lambda col: col.str.contains(search, case=False, na=False))
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


def format_tier_summary(selected_tiers: list[int]) -> str:
    if not selected_tiers:
        return "Tier is the question complexity score. Higher tiers usually mean more reasoning steps, joins, or transformations."

    selected = []
    for tier in selected_tiers:
        description = TIER_COMPLEXITY.get(tier)
        if description:
            selected.append(f"Tier {tier}: {description}")

    if not selected:
        return "Tier is the question complexity score. Higher tiers usually mean more reasoning steps, joins, or transformations."

    return "Selected tier complexity: " + " ".join(selected)


def render_tier_guide(selected_tiers: list[int]) -> None:
    st.caption(format_tier_summary(selected_tiers))
    with st.expander("Tier guide", expanded=bool(selected_tiers)):
        for tier in sorted(TIER_COMPLEXITY):
            st.markdown(f"- **Tier {tier}**: {TIER_COMPLEXITY[tier]}")


def make_progress_frame(records: list[Record], total_questions: int) -> pd.DataFrame:
    return make_progress_frame_for_ids(records, total_questions)


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

    # O(N) approach: Maintain running totals instead of recalculating on every row
    answered, correct, incorrect = 0, 0, 0

    for index, (_, record) in enumerate(sorted_records, start=1):
        prev_status = state.get(record.instance_id, "unanswered")
        new_status = record.status

        # Only update totals if the status actually changed
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


def render_chart(progress: pd.DataFrame, *, empty_message: str) -> None:
    if progress.empty:
        st.info(empty_message)
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=progress["x"],
            y=progress["answered_pct"],
            name="Answered",
            mode="lines+markers",
            line={"color": ANSWERED_COLOR, "width": 3, "shape": "hv"},
            fill="tozeroy",
            fillcolor="rgba(100, 116, 139, 0.18)",
            hovertemplate="Answered: %{customdata[0]}<br>%{y:.1f}%<extra></extra>",
            customdata=progress[["answered"]],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=progress["x"],
            y=progress["correct_pct"],
            name="Correct",
            mode="lines+markers",
            line={"color": CORRECT_COLOR, "width": 3, "shape": "hv"},
            fill="tozeroy",
            fillcolor="rgba(34, 197, 94, 0.18)",
            hovertemplate="Correct: %{customdata[0]}<br>%{y:.1f}%<extra></extra>",
            customdata=progress[["correct"]],
        )
    )
    fig.update_layout(
        height=CHART_HEIGHT,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        paper_bgcolor="#080808",
        plot_bgcolor="#080808",
        font={"color": "#d7d7d7"},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.05, "x": 0},
        yaxis={
            "range": [0, 100],
            "ticksuffix": "%",
            "gridcolor": "rgba(255,255,255,0.08)",
            "zeroline": False,
        },
        xaxis={"gridcolor": "rgba(255,255,255,0.04)", "zeroline": False},
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_grid(frame: pd.DataFrame) -> None:
    tiles = []
    for row in frame.to_dict("records"):
        status = row["status"]
        title = f"{row['instance_id']} | {STATUS_LABELS[status]}"
        if row.get("db"):
            title += f" | {row['db']}"
        if row.get("note"):
            title += f" | {row['note']}"
        if row.get("instruction"):
            title += f" | {row['instruction']}"
        tiles.append(
            '<div class="tile" '
            f'title="{html.escape(title)}" '
            f'style="background:{STATUS_COLORS[status]}"></div>'
        )

    st.markdown(
        f"""
        <style>
        .status-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(10px, 1fr));
            gap: 3px;
            width: 100%;
            align-items: center;
        }}
        .status-grid-wrap {{
            padding-bottom: 16px;
        }}
        .tile {{
            aspect-ratio: 1 / 1;
            border: 1px solid rgba(0, 0, 0, 0.65);
            min-width: 10px;
        }}
        .tile:hover {{
            transform: scale(1.8);
            outline: 1px solid rgba(255, 255, 255, 0.75);
            z-index: 2;
        }}
        </style>
        <div class="status-grid-wrap">
            <div class="status-grid">{"".join(tiles)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #080808;
            color: #f3f4f6;
        }
        [data-testid="stHeader"] {
            background: rgba(8, 8, 8, 0.85);
        }
        [data-testid="stMetricValue"] {
            color: #f8fafc;
        }
        [data-testid="stSidebar"] {
            background: #101010;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(255,255,255,0.08);
        }
        .section-spacer {
            height: 24px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="Spider Progress", layout="wide")
    apply_page_style()

    dataset_path = resolve_path(args.dataset)
    source_path = resolve_path(args.source)
    with st.sidebar:
        st.title("Progress UI")
        metadata_rows = load_category_metadata_rows(str(dataset_path))
        available_tiers = sorted({row["primary_tier"] for row in metadata_rows.values()})
        available_tags = sorted({tag for row in metadata_rows.values() for tag in row["tags"]})
        search = st.text_input("Search", value="")
        selected = st.multiselect(
            "Status",
            options=list(STATUS_ORDER),
            default=list(STATUS_ORDER),
            format_func=lambda value: STATUS_LABELS[value],
        )
        selected_tiers = st.multiselect(
            "Tier",
            options=available_tiers,
            default=[],
            help="Leave empty to include every tier.",
        )
        selected_tags = st.multiselect(
            "Tags",
            options=available_tags,
            default=[],
            help="Multiple tags use AND semantics.",
        )

    # Pass strings to the cached functions, as Streamlit handles string hashing perfectly
    dataset = read_dataset(str(dataset_path))
    records = load_records(str(source_path))
    full_frame = build_status_frame(dataset, records, metadata_rows)

    frame = apply_frame_filters(
        full_frame,
        search=search,
        selected_status=selected,
        selected_tiers=selected_tiers,
        selected_tags=selected_tags,
    )

    summary = compute_overall_summary(frame)
    total = summary["total"]
    answered = summary["answered"]
    correct = summary["correct"]
    incorrect = summary["incorrect"]
    answered_score = summary["coverage_pct"]
    correct_score = summary["accuracy_pct"]

    overview_tab, questions_tab, debug_tab = st.tabs(["Overview", "Questions", "Debug"])

    with overview_tab:
        st.header("Spider2-Snowflake Progress")
        st.caption(f"Dataset: `{dataset_path}`  |  Results: `{source_path}`")

        cols = st.columns(4)
        cols[0].metric("Coverage", f"{answered_score:.1f}%", f"{answered:,} answered")
        cols[1].metric("Accuracy", f"{correct_score:.1f}%", f"{correct:,} correct")
        cols[2].metric("Total", f"{total:,}")
        cols[3].metric("Answered", f"{answered:,}")

        cols = st.columns(3)
        cols[0].metric("Correct", f"{correct:,}")
        cols[1].metric("Incorrect", f"{incorrect:,}")
        cols[2].metric("Unanswered", f"{summary['unanswered']:,}")

        focus = recommend_focus(frame)
        focus_cols = st.columns([2, 1])
        with focus_cols[0]:
            st.subheader("Recommended focus")
            st.info(f"{focus['title']}\n\n{focus['detail']}")
        with focus_cols[1]:
            st.metric("Focus count", f"{focus['count']:,}")
            st.metric("Coverage", f"{focus['coverage_pct']:.1f}%")
            st.metric("Accuracy", f"{focus['accuracy_pct']:.1f}%")

        tier_summary = compute_tier_summary(frame)
        tag_summary = compute_tag_summary(frame)

        st.subheader("Tier progress")
        if tier_summary.empty:
            st.info("No tier data available for the current slice.")
        else:
            st.dataframe(
                tier_summary[
                    [
                        "tier_label",
                        "total",
                        "answered",
                        "correct",
                        "incorrect",
                        "unanswered",
                        "coverage_pct",
                        "accuracy_pct",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

        st.subheader("Tag progress")
        if tag_summary.empty:
            st.info("No tag data available for the current slice.")
        else:
            st.dataframe(
                tag_summary[
                    [
                        "tag_label",
                        "total",
                        "answered",
                        "correct",
                        "incorrect",
                        "unanswered",
                        "coverage_pct",
                        "accuracy_pct",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

    with questions_tab:
        render_status_legend()
        question_columns = [
            "instance_id",
            "status",
            "primary_tier",
            "tags",
            "db",
            "instruction",
            "note",
        ]

        st.subheader("Filtered questions")
        if frame.empty:
            st.info("No questions match the current filters.")
        else:
            question_frame = prepare_question_table(frame)
            st.dataframe(
                question_frame[question_columns],
                width="stretch",
                height=TABLE_HEIGHT,
                row_height=TABLE_ROW_HEIGHT,
                hide_index=True,
            )

            st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

            selection_col, detail_col = st.columns([1, 2])
            with selection_col:
                selected_instance_id = st.selectbox(
                    "Selected question",
                    options=list(question_frame["instance_id"]),
                    format_func=lambda instance_id: format_question_option(
                        question_frame.loc[question_frame["instance_id"] == instance_id].iloc[0]
                    ),
                    label_visibility="visible",
                )
            with detail_col:
                render_question_detail(select_question_row(frame, selected_instance_id))

        if should_show_all_questions(selected_tiers, selected_tags):
            st.subheader("All questions")
            full_question_frame = prepare_question_table(full_frame)
            if full_question_frame.empty:
                st.info("No questions are available for the current dataset.")
            else:
                st.dataframe(
                    full_question_frame[question_columns],
                    width="stretch",
                    height=TABLE_HEIGHT,
                    row_height=TABLE_ROW_HEIGHT,
                    hide_index=True,
                )

    with debug_tab:
        st.header("Debug")
        st.caption("Operational details for triage and empty-state debugging.")

        run_command = build_run_command(dataset_path, source_path)
        debug_cols = st.columns(4)
        debug_cols[0].metric("Loaded records", f"{len(records):,}")
        debug_cols[1].metric("Raw rows", f"{len(full_frame):,}")
        debug_cols[2].metric("Metadata tiers", f"{len(available_tiers):,}")
        debug_cols[3].metric("Metadata tags", f"{len(available_tags):,}")

        detail_cols = st.columns(2)
        with detail_cols[0]:
            st.markdown("**Dataset path**")
            st.code(str(dataset_path))
            st.markdown("**Results source**")
            st.code(str(source_path))
        with detail_cols[1]:
            st.markdown("**Run command**")
            st.code(run_command, language="bash")
            st.markdown("**Metadata state**")
            if metadata_rows:
                st.success(f"Category metadata available for {len(metadata_rows):,} questions.")
            else:
                st.warning("Category metadata is missing for this dataset.")

        st.subheader("Raw rows")
        debug_frame = prepare_debug_frame(full_frame)
        st.dataframe(debug_frame, width="stretch", height=TABLE_HEIGHT, hide_index=True)


if __name__ == "__main__":
    main()
