from datetime import UTC, datetime

import pandas as pd

from progress_ui import (
    Record,
    apply_frame_filters,
    build_status_frame,
    make_progress_frame_for_ids,
)


def test_build_status_frame_joins_category_metadata():
    dataset = pd.DataFrame(
        [
            {"instance_id": "sf_a", "instruction": "q1", "db_id": "DB_A"},
            {"instance_id": "sf_b", "instruction": "q2", "db_id": "DB_B"},
        ]
    )
    records = [
        Record(
            instance_id="sf_a",
            status="correct",
            score=1.0,
            timestamp=datetime(2026, 4, 30, tzinfo=UTC),
            run_id="run-1",
            db="DB_A",
            note=None,
            source_path="/tmp/result.json",
        )
    ]
    category_rows = {
        "sf_a": {
            "primary_tier": 3,
            "tags": ["aggregation", "temporal"],
            "difficulty_notes": "needs a time window",
        }
    }

    frame = build_status_frame(dataset, records, category_rows)

    row_a = frame.loc[frame["instance_id"] == "sf_a"].iloc[0]
    row_b = frame.loc[frame["instance_id"] == "sf_b"].iloc[0]

    assert row_a["primary_tier"] == 3
    assert row_a["tags"] == ["aggregation", "temporal"]
    assert row_a["difficulty_notes"] == "needs a time window"
    assert bool(row_a["category_available"]) is True

    assert pd.isna(row_b["primary_tier"])
    assert row_b["tags"] == []
    assert pd.isna(row_b["difficulty_notes"])
    assert bool(row_b["category_available"]) is False


def test_apply_frame_filters_uses_and_tags_and_skips_uncategorized_only_when_needed():
    frame = pd.DataFrame(
        [
            {
                "instance_id": "sf_a",
                "status": "correct",
                "primary_tier": 3,
                "tags": ["aggregation", "temporal"],
                "category_available": True,
                "db": "DB_A",
                "instruction": "alpha",
                "note": None,
                "difficulty_notes": "note",
            },
            {
                "instance_id": "sf_b",
                "status": "correct",
                "primary_tier": 3,
                "tags": ["aggregation"],
                "category_available": True,
                "db": "DB_B",
                "instruction": "beta",
                "note": None,
                "difficulty_notes": None,
            },
            {
                "instance_id": "sf_c",
                "status": "correct",
                "primary_tier": None,
                "tags": [],
                "category_available": False,
                "db": "DB_C",
                "instruction": "gamma",
                "note": None,
                "difficulty_notes": None,
            },
        ]
    )

    filtered = apply_frame_filters(
        frame,
        selected_status=["correct"],
        selected_tiers=[3],
        selected_tags=["aggregation", "temporal"],
    )
    assert list(filtered["instance_id"]) == ["sf_a"]

    filtered = apply_frame_filters(frame, selected_status=["correct"])
    assert list(filtered["instance_id"]) == ["sf_a", "sf_b", "sf_c"]


def test_make_progress_frame_filters_records_to_selected_questions():
    records = [
        Record(
            instance_id="sf_other",
            status="correct",
            score=1.0,
            timestamp=datetime(2026, 4, 29, tzinfo=UTC),
            run_id="run-0",
            db="DB",
            note=None,
            source_path="/tmp/other.json",
        ),
        Record(
            instance_id="sf_keep",
            status="correct",
            score=1.0,
            timestamp=datetime(2026, 4, 30, tzinfo=UTC),
            run_id="run-1",
            db="DB",
            note=None,
            source_path="/tmp/keep.json",
        ),
    ]

    progress = make_progress_frame_for_ids(records, total_questions=1, selected_instance_ids={"sf_keep"})

    assert list(progress["answered"]) == [1]
    assert list(progress["correct_pct"]) == [100.0]


def test_make_progress_frame_returns_empty_for_empty_slice():
    progress = make_progress_frame_for_ids([], total_questions=0, selected_instance_ids=set())

    assert progress.empty
