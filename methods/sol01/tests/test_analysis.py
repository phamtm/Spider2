"""Tests for run-level failure analysis."""

from __future__ import annotations

import json
from pathlib import Path

from sol01.analysis import analyze_run
from sol01.output import ensure_run_paths, write_trace


def test_analyze_run_groups_failures_and_writes_reports(tmp_path: Path):
    """Synthetic traces should land in the expected failure buckets."""

    run_paths = ensure_run_paths("analysis-run", outputs_root=tmp_path)
    run_paths.eval_dir.mkdir(parents=True, exist_ok=True)
    (run_paths.eval_dir / "summary.json").write_text(
        json.dumps(
            {
                "attempted_local_tasks": 6,
                "correct_local_tasks": 2,
                "missing_csv_count": 1,
                "missing_instance_ids": ["local005"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    local001_csv = run_paths.csv_dir / "local001.csv"
    local001_csv.write_text("answer\n1\n", encoding="utf-8")
    write_trace(
        run_paths,
        instance_id="local001",
        trace={
            "instance_id": "local001",
            "db": "db_alpha",
            "question": "Show one answer.",
            "status": "success",
            "csv_path": str(local001_csv),
            "final_execution": {"ok": True, "row_count": 1},
            "attempts": [
                {
                    "sql": "SELECT 1 AS answer",
                    "validation": {"ok": True, "errors": []},
                    "execution_result": {"ok": True, "error": None},
                }
            ],
        },
    )

    write_trace(
        run_paths,
        instance_id="local002",
        trace={
            "instance_id": "local002",
            "db": "db_alpha",
            "question": "Broken table lookup.",
            "status": "failed",
            "csv_path": None,
            "attempts": [
                {
                    "sql": "SELECT * FROM missing_table",
                    "validation": {"ok": False, "errors": ["Table missing_table is not allowed."]},
                    "execution_result": {
                        "ok": False,
                        "error": "Validation failed before execution.",
                    },
                }
            ],
        },
    )

    write_trace(
        run_paths,
        instance_id="local003",
        trace={
            "instance_id": "local003",
            "db": "db_beta",
            "question": "Read a missing column.",
            "status": "failed",
            "csv_path": None,
            "attempts": [
                {
                    "sql": "SELECT missing_column FROM sales",
                    "validation": {"ok": True, "errors": []},
                    "execution_result": {"ok": False, "error": "no such column: missing_column"},
                }
            ],
        },
    )

    local004_csv = run_paths.csv_dir / "local004.csv"
    local004_csv.write_text("answer\n", encoding="utf-8")
    write_trace(
        run_paths,
        instance_id="local004",
        trace={
            "instance_id": "local004",
            "db": "db_beta",
            "question": "Return no rows.",
            "status": "success",
            "csv_path": str(local004_csv),
            "final_execution": {"ok": True, "row_count": 0},
            "attempts": [
                {
                    "sql": "SELECT * FROM sales WHERE 1 = 0",
                    "validation": {"ok": True, "errors": []},
                    "execution_result": {"ok": True, "error": None},
                }
            ],
        },
    )

    missing_csv_path = run_paths.csv_dir / "local005.csv"
    write_trace(
        run_paths,
        instance_id="local005",
        trace={
            "instance_id": "local005",
            "db": "db_gamma",
            "question": "CSV disappeared after success.",
            "status": "success",
            "csv_path": str(missing_csv_path),
            "final_execution": {"ok": True, "row_count": 2},
            "attempts": [
                {
                    "sql": "SELECT 1 UNION ALL SELECT 2",
                    "validation": {"ok": True, "errors": []},
                    "execution_result": {"ok": True, "error": None},
                }
            ],
        },
    )

    write_trace(
        run_paths,
        instance_id="local006",
        trace={
            "instance_id": "local006",
            "db": "db_gamma",
            "question": "Try a repair after the critic complains.",
            "status": "failed",
            "csv_path": None,
            "attempts": [
                {
                    "stage": "initial_1",
                    "sql": "SELECT amount FROM sales",
                    "validation": {"ok": True, "errors": []},
                    "execution_result": {"ok": True, "error": None},
                    "critic": {
                        "confidence": 0.2,
                        "issues": ["Result is missing the customer breakdown."],
                        "should_repair": True,
                        "repair_focus": "add the customer column",
                    },
                },
                {
                    "stage": "critic_repair",
                    "sql": "SELECT missing_column FROM sales",
                    "validation": {"ok": True, "errors": []},
                    "execution_result": {"ok": False, "error": "no such column: missing_column"},
                },
            ],
        },
    )

    report = analyze_run("analysis-run", outputs_root=tmp_path)

    assert report["status_counts"] == {"success": 3, "failed": 3, "skipped": 0}
    assert [item["instance_id"] for item in report["by_category"]["validation"]] == ["local002"]
    assert [item["instance_id"] for item in report["by_category"]["execution"]] == [
        "local003",
        "local006",
    ]
    assert [item["instance_id"] for item in report["by_category"]["empty_result"]] == ["local004"]
    assert [item["instance_id"] for item in report["by_category"]["critic"]] == ["local006"]
    assert [item["instance_id"] for item in report["by_category"]["missing_csv"]] == ["local005"]
    assert [item["instance_id"] for item in report["by_category"]["retrieval_miss"]] == ["local002"]

    summary_text = (run_paths.analysis_dir / "summary.md").read_text(encoding="utf-8")
    assert "Analysis for analysis-run" in summary_text
    assert "- validation: 1 (local002)" in summary_text
    assert "- missing_csv: 1 (local005)" in summary_text

    failures = json.loads((run_paths.analysis_dir / "failures.json").read_text(encoding="utf-8"))
    assert failures["run_id"] == "analysis-run"
    assert failures["category_counts"]["critic"] == 1
    assert failures["by_database"]["db_gamma"]["category_counts"]["missing_csv"] == 1


def test_analyze_run_does_not_label_success_keywords_as_failures(tmp_path: Path):
    """Success traces should not fall into failure-only heuristic buckets."""

    run_paths = ensure_run_paths("success-keywords", outputs_root=tmp_path)
    csv_path = run_paths.csv_dir / "local101.csv"
    csv_path.write_text("day,total\n2024-01-01,10\n", encoding="utf-8")
    write_trace(
        run_paths,
        instance_id="local101",
        trace={
            "instance_id": "local101",
            "db": "db_dates",
            "question": "Show total sales by date.",
            "status": "success",
            "csv_path": str(csv_path),
            "final_execution": {"ok": True, "row_count": 1},
            "attempts": [
                {
                    "sql": "SELECT order_date, SUM(amount) FROM sales GROUP BY order_date",
                    "validation": {"ok": True, "errors": []},
                    "execution_result": {"ok": True, "error": None},
                    "critic": {
                        "confidence": 0.9,
                        "issues": ["Looks fine."],
                        "should_repair": False,
                        "repair_focus": None,
                    },
                }
            ],
        },
    )

    report = analyze_run("success-keywords", outputs_root=tmp_path)

    assert report["by_category"]["critic"] == []
    assert report["by_category"]["aggregation_issue"] == []
    assert report["by_category"]["date_filter_issue"] == []
    assert report["by_category"]["retrieval_miss"] == []


def test_analyze_run_uses_manifest_for_missing_csv_without_eval_summary(tmp_path: Path):
    """Partial runs should still report missing CSVs before eval has run."""

    run_paths = ensure_run_paths("partial-run", outputs_root=tmp_path)
    run_paths.manifest_path.write_text(
        json.dumps({"task_ids": ["local201", "local202"]}) + "\n",
        encoding="utf-8",
    )
    csv_path = run_paths.csv_dir / "local201.csv"
    csv_path.write_text("answer\n1\n", encoding="utf-8")
    write_trace(
        run_paths,
        instance_id="local201",
        trace={
            "instance_id": "local201",
            "db": "db_partial",
            "question": "Completed task.",
            "status": "success",
            "csv_path": str(csv_path),
            "final_execution": {"ok": True, "row_count": 1},
            "attempts": [
                {
                    "sql": "SELECT 1 AS answer",
                    "validation": {"ok": True, "errors": []},
                    "execution_result": {"ok": True, "error": None},
                }
            ],
        },
    )

    report = analyze_run("partial-run", outputs_root=tmp_path)

    assert [item["instance_id"] for item in report["by_category"]["missing_csv"]] == ["local202"]
