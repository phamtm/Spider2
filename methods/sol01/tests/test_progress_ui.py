from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sol01.progress_ui import (
    Record,
    apply_frame_filters,
    build_run_command,
    build_status_frame,
    latest_records,
    load_records,
    make_progress_frame_for_ids,
    prepare_debug_frame,
    prepare_question_table,
    select_question_row,
    should_show_all_questions,
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


def test_build_status_frame_uses_nullable_missing_values():
    dataset = pd.DataFrame([{"instance_id": "sf_a", "instruction": "q1"}])

    frame = build_status_frame(dataset, [], None)

    row = frame.loc[frame["instance_id"] == "sf_a"].iloc[0]

    assert pd.isna(row["score"])
    assert pd.isna(row["timestamp"])
    assert pd.isna(row["run_id"])
    assert pd.isna(row["db"])
    assert pd.isna(row["note"])
    assert pd.isna(row["source_path"])
    assert pd.isna(row["primary_tier"])
    assert pd.isna(row["difficulty_notes"])


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


def test_apply_frame_filters_supports_scoped_search():
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
                "note": "first",
                "difficulty_notes": "note",
            },
            {
                "instance_id": "sf_b",
                "status": "correct",
                "primary_tier": 4,
                "tags": ["comparison"],
                "category_available": True,
                "db": "DB_B",
                "instruction": "beta",
                "note": "second",
                "difficulty_notes": None,
            },
        ]
    )

    filtered = apply_frame_filters(frame, search="id:sf_b")

    assert list(filtered["instance_id"]) == ["sf_b"]


def test_apply_frame_filters_excludes_missing_metadata_when_category_filters_are_active():
    frame = pd.DataFrame(
        [
            {
                "instance_id": "sf_a",
                "status": "correct",
                "primary_tier": 3,
                "tags": ["aggregation"],
                "category_available": True,
                "db": "DB_A",
                "instruction": "alpha",
                "note": None,
                "difficulty_notes": None,
            },
            {
                "instance_id": "sf_missing",
                "status": "correct",
                "primary_tier": None,
                "tags": [],
                "category_available": False,
                "db": "DB_B",
                "instruction": "beta",
                "note": None,
                "difficulty_notes": None,
            },
        ]
    )

    filtered = apply_frame_filters(frame, selected_tiers=[3])

    assert list(filtered["instance_id"]) == ["sf_a"]


def test_latest_records_prefers_sorted_timestamp_order_over_input_order():
    records = [
        Record(
            instance_id="sf_a",
            status="incorrect",
            score=0.0,
            timestamp=datetime(2026, 4, 29, tzinfo=UTC),
            run_id="run-1",
            db="DB",
            note=None,
            source_path="/tmp/older.json",
        ),
        Record(
            instance_id="sf_a",
            status="correct",
            score=1.0,
            timestamp=datetime(2026, 4, 30, tzinfo=UTC),
            run_id="run-2",
            db="DB",
            note=None,
            source_path="/tmp/newer.json",
        ),
    ]

    latest = latest_records(records)

    assert latest["sf_a"].status == "correct"
    assert latest["sf_a"].source_path == "/tmp/newer.json"


def test_build_status_frame_handles_empty_results_and_missing_metadata():
    dataset = pd.DataFrame(
        [
            {"instance_id": "sf_a", "instruction": "q1", "db_id": "DB_A"},
            {"instance_id": "sf_b", "instruction": "q2", "db_id": "DB_B"},
        ]
    )

    frame = build_status_frame(dataset, [], None)

    assert list(frame["instance_id"]) == ["sf_a", "sf_b"]
    assert list(frame["status"]) == ["unanswered", "unanswered"]
    assert list(frame["tags"]) == [[], []]
    assert list(frame["category_available"]) == [False, False]


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

    progress = make_progress_frame_for_ids(
        records, total_questions=1, selected_instance_ids={"sf_keep"}
    )

    assert list(progress["answered"]) == [1]
    assert list(progress["correct_pct"]) == [100.0]


def test_make_progress_frame_returns_empty_for_empty_slice():
    progress = make_progress_frame_for_ids([], total_questions=0, selected_instance_ids=set())

    assert progress.empty


def test_prepare_question_table_sorts_unanswered_first_and_truncates_instruction():
    frame = pd.DataFrame(
        [
            {
                "instance_id": "sf_answered",
                "status": "correct",
                "primary_tier": 3,
                "tags": ["aggregation"],
                "db": "DB_B",
                "instruction": "short prompt",
                "note": "done",
                "diagnostics": "validation: ok",
                "score": 1.0,
            },
            {
                "instance_id": "sf_unanswered",
                "status": "unanswered",
                "primary_tier": 1,
                "tags": ["temporal"],
                "db": "DB_A",
                "instruction": "x" * 160,
                "note": None,
                "score": None,
            },
        ]
    )

    table = prepare_question_table(frame)

    assert list(table["instance_id"]) == ["sf_unanswered", "sf_answered"]
    assert table.loc[0, "status"] == "Unanswered"
    assert table.loc[0, "primary_tier"] == "Tier 1"
    assert table.loc[0, "instruction"].endswith("…")
    assert table.loc[1, "note"] == "done"
    assert table.loc[1, "diagnostics"] == "validation: ok"


def test_should_show_all_questions_turns_on_for_tier_or_tag_filters():
    assert should_show_all_questions([1], []) is True
    assert should_show_all_questions([], ["aggregation"]) is True
    assert should_show_all_questions([], []) is False


def test_prepare_question_table_keeps_all_rows_available():
    frame = pd.DataFrame(
        [
            {"instance_id": "sf_a", "status": "correct", "primary_tier": 1, "tags": []},
            {"instance_id": "sf_b", "status": "incorrect", "primary_tier": 2, "tags": []},
            {"instance_id": "sf_c", "status": "unanswered", "primary_tier": 3, "tags": []},
        ]
    )

    table = prepare_question_table(frame)

    assert list(table["instance_id"]) == ["sf_c", "sf_b", "sf_a"]


def test_select_question_row_returns_full_detail_fields():
    frame = pd.DataFrame(
        [
            {
                "instance_id": "sf_1",
                "status": "incorrect",
                "primary_tier": 4,
                "tags": ["ranking", "joins"],
                "db": "DB_A",
                "instruction": "find the top seller",
                "note": "missing join",
                "diagnostics": "ranking: validation=-180.0",
                "difficulty_notes": "needs ranking",
                "source_path": "/tmp/result.json",
                "score": 0.0,
            }
        ]
    )

    row = select_question_row(frame, "sf_1")

    assert row is not None
    assert row["status_label"] == "Incorrect"
    assert row["primary_tier_label"] == "Tier 4"
    assert row["tags_label"] == "ranking, joins"
    assert row["source_path"] == "/tmp/result.json"
    assert row["diagnostics"] == "ranking: validation=-180.0"


def test_prepare_debug_frame_keeps_operational_fields_visible():
    frame = pd.DataFrame(
        [
            {
                "instance_id": "sf_1",
                "status": "correct",
                "score": 1.0,
                "timestamp": datetime(2026, 4, 30, tzinfo=UTC),
                "run_id": "run-1",
                "db": "DB_A",
                "instruction": "question text",
                "note": "note text",
                "diagnostics": "validation: missing table",
                "source_path": "/tmp/result.json",
                "primary_tier": 3,
                "tags": ["aggregation", "temporal"],
                "difficulty_notes": "needs a time window",
                "category_available": True,
            }
        ]
    )

    debug = prepare_debug_frame(frame)

    assert list(debug.columns) == [
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
    assert debug.loc[0, "timestamp"].startswith("2026-04-30")
    assert debug.loc[0, "primary_tier"] == "Tier 3"
    assert debug.loc[0, "tags"] == "aggregation, temporal"
    assert debug.loc[0, "diagnostics"] == "validation: missing table"


def test_prepare_debug_frame_returns_empty_schema_for_missing_results():
    debug = prepare_debug_frame(pd.DataFrame())

    assert debug.empty
    assert "instance_id" in debug.columns


def test_build_run_command_includes_dataset_and_source_paths():
    command = build_run_command(
        Path("/tmp/dataset.jsonl"),
        Path("/tmp/results/latest.json"),
    )

    assert command == (
        "uv run streamlit run sol01/progress_ui/app.py -- "
        "--dataset /tmp/dataset.jsonl "
        "--source /tmp/results/latest.json"
    )


def test_load_records_extracts_trace_diagnostics_from_trace_json(tmp_path: Path):
    trace_dir = tmp_path / "run"
    trace_dir.mkdir()
    (trace_dir / "trace.json").write_text(
        """
        {
          "instance_id": "sf_1",
          "status": "failed",
          "db": "DB_A",
          "question": "Find a customer.",
          "schema_selection": {
            "diagnostics": {
              "prompt_budget": {
                "planning_prompt_chars": 1200,
                "sql_reference_context_chars": 900,
                "max_schema_prompt_chars": 24000
              }
            }
          },
          "attempts": [
            {
              "validation": {"ok": false, "errors": ["missing grouped key StyleID"]},
              "execution_result": {"ok": false, "error": "Validation failed before execution."},
              "shape_report": {"violations": ["missing grouped key StyleID"]},
              "filter_grounding_report": {
                "reason": "Empty result but probe values suggest a stored label variant.",
                "exact_filters": ["country = 'Russia'"],
                "value_rewrites": [
                  {
                    "filter": "country = 'Russia'",
                    "rewrite": "Russian Federation"
                  }
                ]
              },
              "critic": {
                "should_repair": true,
                "repair_focus": "add the grouped key",
                "issues": ["Result is missing the customer breakdown."]
              },
              "score_breakdown": {
                "execution_status": -1000.0,
                "validation": -180.0,
                "shape": -28.0,
                "filter_grounding": 16.0,
                "confidence_tiebreaker": 0.01
              }
            }
          ]
        }
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    records = load_records(str(trace_dir))

    assert len(records) == 1
    assert records[0].diagnostics == (
        "prompt budget: planning=1200/24000, context=900/24000 | "
        "validation: missing grouped key StyleID | "
        "execution: Validation failed before execution. | "
        "shape: missing grouped key StyleID | "
        "filter grounding: country = 'Russia' -> Russian Federation | "
        "filters: country = 'Russia' | "
        "critic: Result is missing the customer breakdown. | "
        "ranking: execution_status=-1000, validation=-180, shape=-28"
    )
