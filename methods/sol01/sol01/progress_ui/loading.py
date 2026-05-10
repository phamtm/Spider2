from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from sol01.infra.logging import get_logger
from sol01.loading.category_metadata import (
    CategoryMetadataValidationError,
    load_category_metadata_map,
)
from sol01.progress_ui.models import Record
from sol01.progress_ui.parsing import (
    normalize_item,
    read_json,
    read_jsonl,
)

logger = get_logger(__name__)


@st.cache_data(ttl=3600)
def read_dataset(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        return pd.DataFrame(columns=["instance_id", "instruction", "db_id"])

    try:
        frame = pd.read_json(path_obj, lines=True)
    except ValueError:
        return pd.DataFrame(columns=["instance_id", "instruction", "db_id"])

    if "instance_id" not in frame.columns:
        return pd.DataFrame(columns=["instance_id", "instruction", "db_id"])

    for column in ("instruction", "db_id"):
        if column not in frame.columns:
            frame[column] = ""

    return frame[["instance_id", "instruction", "db_id"]].drop_duplicates("instance_id")


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
    except Exception as exc:
        logger.warning("skipping corrupted csv result file", path=str(path), error=str(exc))
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
                diagnostics=None,
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


@st.cache_data(ttl=5)
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


@st.cache_data(ttl=3600)
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
