"""Tests for run-level failure analysis."""

from __future__ import annotations

import json
from pathlib import Path

from sol01.analysis import analyze_run
from sol01.output import ensure_run_paths, write_trace


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_analyze_run_groups_failures_and_writes_reports(tmp_path: Path):
    """Synthetic traces should land in the expected failure buckets."""

    run_paths = ensure_run_paths("analysis-run", outputs_root=tmp_path)
    run_paths.eval_dir.mkdir(parents=True, exist_ok=True)
    (run_paths.eval_dir / "summary.json").write_text(
        json.dumps(
            {
                "attempted_tasks": 6,
                "correct_tasks": 2,
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
            "schema_selection": {
                "selected_tables": ["known_table"],
                "expanded_tables": ["known_table"],
            },
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
    assert report["by_category"]["validation"][0]["evidence"] == [
        "Table missing_table is not allowed."
    ]
    assert report["by_category"]["retrieval_miss"][0]["evidence"] == [
        "Validation reported a table outside the selected schema."
    ]
    assert report["by_category"]["execution"][1]["hints"] == []

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
    assert "aggregation_issue" not in report["by_category"]
    assert "date_filter_issue" not in report["by_category"]
    assert report["by_category"]["retrieval_miss"] == []


def test_analyze_run_keeps_speculative_signals_as_hints(tmp_path: Path):
    """Weak date and aggregation signals should not become failure categories."""

    run_paths = ensure_run_paths("failure-hints", outputs_root=tmp_path)
    write_trace(
        run_paths,
        instance_id="local151",
        trace={
            "instance_id": "local151",
            "db": "db_dates",
            "question": "Show total sales by month.",
            "status": "failed",
            "csv_path": None,
            "attempts": [
                {
                    "sql": (
                        "SELECT DATE_TRUNC('month', order_date), SUM(amount) FROM sales GROUP BY 1"
                    ),
                    "validation": {"ok": True, "errors": []},
                    "execution_result": {"ok": False, "error": "invalid timestamp"},
                }
            ],
        },
    )

    report = analyze_run("failure-hints", outputs_root=tmp_path)

    execution_record = report["by_category"]["execution"][0]
    assert execution_record["instance_id"] == "local151"
    assert execution_record["hints"] == [
        "possible_aggregation_issue",
        "possible_date_filter_issue",
    ]
    assert "aggregation_issue" not in report["by_category"]
    assert "date_filter_issue" not in report["by_category"]


def test_analyze_run_requires_schema_evidence_for_retrieval_miss(tmp_path: Path):
    """Table validation errors alone should not be called retrieval misses."""

    run_paths = ensure_run_paths("schema-evidence", outputs_root=tmp_path)
    write_trace(
        run_paths,
        instance_id="local175",
        trace={
            "instance_id": "local175",
            "db": "db_schema",
            "question": "Read a missing table.",
            "status": "failed",
            "csv_path": None,
            "attempts": [
                {
                    "sql": "SELECT * FROM missing_table",
                    "validation": {"ok": False, "errors": ["Table missing_table is not allowed."]},
                    "execution_result": {"ok": False, "error": "Validation failed."},
                }
            ],
        },
    )

    report = analyze_run("schema-evidence", outputs_root=tmp_path)

    assert [item["instance_id"] for item in report["by_category"]["validation"]] == ["local175"]
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


def test_analyze_run_groups_results_by_primary_tier_and_tag(tmp_path: Path):
    """Category summaries should track pass/fail counts by tier and tag."""

    dataset_path = tmp_path / "spider2-snow.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "instance_id": "local001",
                "instruction": "q",
                "db_id": "DB",
                "external_knowledge": None,
            },
            {
                "instance_id": "local002",
                "instruction": "q",
                "db_id": "DB",
                "external_knowledge": None,
            },
            {
                "instance_id": "local003",
                "instruction": "q",
                "db_id": "DB",
                "external_knowledge": None,
            },
        ],
    )
    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    _write_jsonl(
        batch_dir / "batch_01.jsonl",
        [
            {"instance_id": "local001", "primary_tier": 1, "tags": ["aggregation", "temporal"]},
            {"instance_id": "local002", "primary_tier": 3, "tags": ["aggregation"]},
            {"instance_id": "local003", "primary_tier": 3, "tags": ["comparison", "temporal"]},
        ],
    )

    run_paths = ensure_run_paths("category-analysis", outputs_root=tmp_path)
    run_paths.eval_dir.mkdir(parents=True, exist_ok=True)
    (run_paths.eval_dir / "summary.json").write_text(
        json.dumps(
            {
                "attempted_tasks": 3,
                "correct_tasks": 1,
                "missing_csv_count": 0,
                "per_instance": [
                    {
                        "instance_id": "local001",
                        "score": 1,
                        "passed": True,
                        "csv_present": True,
                        "failure_reason": None,
                    },
                    {
                        "instance_id": "local002",
                        "score": 0,
                        "passed": False,
                        "csv_present": True,
                        "failure_reason": "official_fail",
                    },
                    {
                        "instance_id": "local003",
                        "score": 0,
                        "passed": False,
                        "csv_present": True,
                        "failure_reason": "official_fail",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    for instance_id in ("local001", "local002", "local003"):
        write_trace(
            run_paths,
            instance_id=instance_id,
            trace={
                "instance_id": instance_id,
                "db": "db_alpha",
                "question": f"Question {instance_id}",
                "status": "success" if instance_id == "local001" else "failed",
                "csv_path": str(run_paths.csv_dir / f"{instance_id}.csv"),
                "final_execution": {"ok": True, "row_count": 1},
                "attempts": [
                    {
                        "sql": "SELECT 1",
                        "validation": {"ok": True, "errors": []},
                        "execution_result": {"ok": True, "error": None},
                    }
                ],
            },
        )

    report = analyze_run(
        "category-analysis",
        outputs_root=tmp_path,
        dataset_path=dataset_path,
        batch_dir=batch_dir,
    )

    assert report["by_primary_tier"][1]["passed"] == 1
    assert report["by_primary_tier"][1]["failed"] == 0
    assert report["by_primary_tier"][3]["passed"] == 0
    assert report["by_primary_tier"][3]["failed"] == 2
    assert report["by_tag"]["aggregation"]["passed"] == 1
    assert report["by_tag"]["aggregation"]["failed"] == 1
    assert report["by_tag"]["temporal"]["total"] == 2

    summary_text = (run_paths.analysis_dir / "summary.md").read_text(encoding="utf-8")
    assert "## By Primary Tier" in summary_text
    assert "- tier 3: passed 0, failed 2, total 2" in summary_text
    assert "## By Tag" in summary_text


def test_analyze_run_records_category_metadata_provenance(tmp_path: Path):
    """Category analysis should expose the exact metadata snapshot it used."""

    dataset_path = tmp_path / "spider2-snow.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "instance_id": "local001",
                "instruction": "q",
                "db_id": "DB",
                "external_knowledge": None,
            }
        ],
    )
    run_paths = ensure_run_paths("category-provenance", outputs_root=tmp_path)
    run_paths.eval_dir.mkdir(parents=True, exist_ok=True)
    (run_paths.eval_dir / "summary.json").write_text(
        json.dumps(
            {
                "attempted_tasks": 1,
                "correct_tasks": 1,
                "missing_csv_count": 0,
                "per_instance": [
                    {
                        "instance_id": "local001",
                        "score": 1,
                        "passed": True,
                        "csv_present": True,
                        "failure_reason": None,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = run_paths.csv_dir / "local001.csv"
    csv_path.write_text("answer\n1\n", encoding="utf-8")
    write_trace(
        run_paths,
        instance_id="local001",
        trace={
            "instance_id": "local001",
            "db": "db_alpha",
            "question": "Question local001",
            "status": "success",
            "csv_path": str(csv_path),
            "final_execution": {"ok": True, "row_count": 1},
            "attempts": [
                {
                    "sql": "SELECT 1",
                    "validation": {"ok": True, "errors": []},
                    "execution_result": {"ok": True, "error": None},
                }
            ],
        },
    )

    batch_dir_one = tmp_path / "batches_one"
    batch_dir_one.mkdir()
    _write_jsonl(
        batch_dir_one / "batch_01.jsonl",
        [{"instance_id": "local001", "primary_tier": 1, "tags": ["aggregation"]}],
    )
    first_report = analyze_run(
        "category-provenance",
        outputs_root=tmp_path,
        dataset_path=dataset_path,
        batch_dir=batch_dir_one,
    )

    batch_dir_two = tmp_path / "batches_two"
    batch_dir_two.mkdir()
    _write_jsonl(
        batch_dir_two / "batch_01.jsonl",
        [{"instance_id": "local001", "primary_tier": 1, "tags": ["comparison"]}],
    )
    second_report = analyze_run(
        "category-provenance",
        outputs_root=tmp_path,
        dataset_path=dataset_path,
        batch_dir=batch_dir_two,
    )

    first_metadata = first_report["category_metadata"]
    second_metadata = second_report["category_metadata"]
    assert first_metadata["dataset_path"] == str(dataset_path)
    assert first_metadata["batch_dir"] == str(batch_dir_one)
    assert first_metadata["record_count"] == 1
    assert first_metadata["sha256"] != second_metadata["sha256"]
    assert second_metadata["batch_dir"] == str(batch_dir_two)

    summary_text = (run_paths.analysis_dir / "summary.md").read_text(encoding="utf-8")
    assert "## Category Metadata" in summary_text
    assert f"- batches: {batch_dir_two}" in summary_text


def test_analyze_run_reports_unmapped_category_coverage(tmp_path: Path):
    """Missing category rows should stay visible in coverage counts."""

    dataset_path = tmp_path / "spider2-snow.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "instance_id": "local001",
                "instruction": "q",
                "db_id": "DB",
                "external_knowledge": None,
            },
            {
                "instance_id": "local002",
                "instruction": "q",
                "db_id": "DB",
                "external_knowledge": None,
            },
        ],
    )
    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    _write_jsonl(
        batch_dir / "batch_01.jsonl",
        [{"instance_id": "local001", "primary_tier": 1, "tags": ["aggregation"]}],
    )

    run_paths = ensure_run_paths("category-coverage", outputs_root=tmp_path)
    run_paths.eval_dir.mkdir(parents=True, exist_ok=True)
    (run_paths.eval_dir / "summary.json").write_text(
        json.dumps(
            {
                "attempted_tasks": 2,
                "correct_tasks": 1,
                "missing_csv_count": 0,
                "per_instance": [
                    {
                        "instance_id": "local001",
                        "score": 1,
                        "passed": True,
                        "csv_present": True,
                        "failure_reason": None,
                    },
                    {
                        "instance_id": "local002",
                        "score": 0,
                        "passed": False,
                        "csv_present": True,
                        "failure_reason": "official_fail",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    for instance_id in ("local001", "local002"):
        csv_path = run_paths.csv_dir / f"{instance_id}.csv"
        csv_path.write_text("answer\n1\n", encoding="utf-8")
        write_trace(
            run_paths,
            instance_id=instance_id,
            trace={
                "instance_id": instance_id,
                "db": "db_alpha",
                "question": f"Question {instance_id}",
                "status": "success" if instance_id == "local001" else "failed",
                "csv_path": str(csv_path),
                "final_execution": {"ok": True, "row_count": 1},
                "attempts": [
                    {
                        "sql": "SELECT 1",
                        "validation": {"ok": True, "errors": []},
                        "execution_result": {"ok": True, "error": None},
                    }
                ],
            },
        )

    report = analyze_run(
        "category-coverage",
        outputs_root=tmp_path,
        dataset_path=dataset_path,
        batch_dir=batch_dir,
    )

    assert report["category_coverage"] == {"mapped": 1, "unmapped": 1, "total": 2}
    assert report["by_primary_tier"][1]["total"] == 1
    assert "local002" not in report["by_primary_tier"][1]["instance_ids"]

    summary_text = (run_paths.analysis_dir / "summary.md").read_text(encoding="utf-8")
    assert "## Category Coverage" in summary_text
    assert "- mapped: 1, unmapped: 1, total 2" in summary_text


def test_analyze_run_keeps_empty_per_instance_as_source_of_truth(tmp_path: Path):
    """An empty official eval row set should not fall back to trace status."""

    dataset_path = tmp_path / "spider2-snow.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "instance_id": "local101",
                "instruction": "q",
                "db_id": "DB",
                "external_knowledge": None,
            }
        ],
    )
    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    _write_jsonl(
        batch_dir / "batch_01.jsonl",
        [{"instance_id": "local101", "primary_tier": 2, "tags": ["temporal"]}],
    )

    run_paths = ensure_run_paths("empty-per-instance", outputs_root=tmp_path)
    run_paths.eval_dir.mkdir(parents=True, exist_ok=True)
    (run_paths.eval_dir / "summary.json").write_text(
        json.dumps(
            {
                "attempted_tasks": 1,
                "correct_tasks": 0,
                "missing_csv_count": 0,
                "per_instance": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = run_paths.csv_dir / "local101.csv"
    csv_path.write_text("answer\n1\n", encoding="utf-8")
    write_trace(
        run_paths,
        instance_id="local101",
        trace={
            "instance_id": "local101",
            "db": "db_beta",
            "question": "Question local101",
            "status": "success",
            "csv_path": str(csv_path),
            "final_execution": {"ok": True, "row_count": 1},
            "attempts": [
                {
                    "sql": "SELECT 1",
                    "validation": {"ok": True, "errors": []},
                    "execution_result": {"ok": True, "error": None},
                }
            ],
        },
    )

    report = analyze_run(
        "empty-per-instance",
        outputs_root=tmp_path,
        dataset_path=dataset_path,
        batch_dir=batch_dir,
    )

    assert report["category_rows"]["source"] == "eval_summary.per_instance"
    assert report["category_rows"]["source_row_count"] == 0
    assert report["by_primary_tier"] == {}
    assert report["by_tag"] == {}

    summary_text = (run_paths.analysis_dir / "summary.md").read_text(encoding="utf-8")
    assert "source: eval summary per_instance (0 rows)" in summary_text
    assert "- mapped: 0, unmapped: 0, total 0" in summary_text
