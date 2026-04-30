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
            filtered["tags"].apply(
                lambda tags: all(tag in (tags or []) for tag in selected_tags)
            )
        ]

    return filtered


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

    source_default = args.source if resolve_path(args.source).exists() else ""
    with st.sidebar:
        st.title("Progress UI")
        dataset_path = resolve_path(
            st.text_input(
                "Dataset JSONL",
                value=args.dataset,
                help="Used for the full question list.",
            )
        )
        source_path = resolve_path(
            st.text_input(
                "Results source",
                value=source_default,
                help=(
                    "File or directory: latest.json, task_results.jsonl, "
                    "summary.json, per_instance.jsonl, CSV, SQL dir."
                ),
            )
        )
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
        if not available_tiers or not available_tags:
            st.caption("Category metadata is unavailable for the current dataset.")
        st.caption("Run: `uv run streamlit run progress_ui.py`")

    # Pass strings to the cached functions, as Streamlit handles string hashing perfectly
    dataset = read_dataset(str(dataset_path))
    records = load_records(str(source_path))
    frame = build_status_frame(dataset, records, metadata_rows)

    frame = apply_frame_filters(
        frame,
        search=search,
        selected_status=selected,
        selected_tiers=selected_tiers,
        selected_tags=selected_tags,
    )
    selected_instance_ids = set(frame["instance_id"]) if not frame.empty else set()

    totals = frame["status"].value_counts().to_dict()
    total = len(frame)
    answered = total - totals.get("unanswered", 0)
    correct = totals.get("correct", 0)
    incorrect = totals.get("incorrect", 0)
    answered_score = answered / total * 100 if total else 0
    correct_score = correct / answered * 100 if answered else 0

    st.title(f"{answered} of {total} questions answered, {correct} correct ({correct_score:.1f}%)")
    st.caption(f"Dataset: `{dataset_path}`  |  Results: `{source_path}`")

    cols = st.columns(5)
    cols[0].metric("Questions", f"{total:,}")
    cols[1].metric("Answered", f"{answered:,}", f"{answered_score:.1f}%")
    cols[2].metric("Correct", f"{correct:,}")
    cols[3].metric("Incorrect", f"{incorrect:,}")
    cols[4].metric("Unanswered", f"{totals.get('unanswered', 0):,}")

    st.subheader("Progress Chart")
    chart_empty_message = (
        "No questions match the current filters."
        if frame.empty
        else "No progress records found for the current slice."
    )
    render_chart(
        make_progress_frame_for_ids(records, len(frame), selected_instance_ids),
        empty_message=chart_empty_message,
    )

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    with st.expander("Question Grid", expanded=True):
        render_grid(frame)

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    with st.expander("Rows"):
        display = frame.copy()
        display["timestamp"] = display["timestamp"].astype(str)
        st.dataframe(
            display[
                [
                    "instance_id",
                    "status",
                    "score",
                    "primary_tier",
                    "tags",
                    "difficulty_notes",
                    "db",
                    "timestamp",
                    "run_id",
                    "note",
                    "instruction",
                    "source_path",
                ]
            ],
            width="stretch",
            height=TABLE_HEIGHT,
            row_height=TABLE_ROW_HEIGHT,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
