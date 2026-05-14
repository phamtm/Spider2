"""Tests for offline schema context coverage evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from sol01 import cli
from sol01.cli import schema_context as cli_schema_context
from sol01.infra.config import SchemaContextConfig
from sol01.models import ColumnSchema, SchemaContextChunk, SchemaObject, TableSchema, Task
from sol01.schema.schema_context_cache import SchemaContextCache
from sol01.schema.schema_context_eval import (
    load_gold_tables,
    run_schema_context_eval,
    write_schema_context_eval_report,
)


def test_load_gold_tables_reads_offline_jsonl(tmp_path: Path):
    """Gold-table labels should be read only from the explicit offline file."""

    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(
        "\n".join(
            [
                json.dumps({"instance_id": "sf001", "gold_tables": ["DB.PUBLIC.ORDERS"]}),
                json.dumps({"instance_id": "sf002", "gold_tables": ["DB.PUBLIC.CUSTOMERS"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_gold_tables(gold_path) == {
        "sf001": ["DB.PUBLIC.ORDERS"],
        "sf002": ["DB.PUBLIC.CUSTOMERS"],
    }


def test_schema_context_eval_reports_gold_coverage_family_success_and_failures():
    """The evaluator should measure pre/post resolver recall and failure evidence."""

    db_index = _db_index()
    index = _schema_context_cache()
    tasks = [
        Task(
            instance_id="sf_family",
            db="DB",
            question="Show every historical sales table.",
        ),
        Task(
            instance_id="sf_missing",
            db="DB",
            question="Show customer sales details.",
        ),
    ]

    report = run_schema_context_eval(
        tasks,
        gold_tables_by_instance={
            "sf_family": ["DB.PUBLIC.SALES_2022", "DB.PUBLIC.SALES_2023"],
            "sf_missing": ["DB.PUBLIC.CUSTOMERS"],
        },
        config=SchemaContextConfig(object_cutoff=3),
        db_index_loader=lambda db: db_index,
        schema_context_cache_loader=lambda db, db_index, config: index,
    )

    assert report.task_count == 2
    assert report.covered_task_count == 0
    assert report.pre_resolver_gold_recall == 0.5
    assert report.pre_resolver_any_gold_recall == 0.5
    assert report.post_resolver_gold_recall == 0.5
    assert report.post_resolver_all_gold_recall == 0.5
    assert report.family_expansion_success == 1.0
    assert report.average_prompt_reduction < 1.0
    assert [item["instance_id"] for item in report.recall_regressions] == ["sf_missing"]
    assert report.failures == [
        {
            "instance_id": "sf_missing",
            "db": "DB",
            "missing_gold_tables": ["DB.PUBLIC.CUSTOMERS"],
            "top_evidence": report.failures[0]["top_evidence"],
        }
    ]
    assert report.failures[0]["top_evidence"][0]["chunk_id"]


def test_schema_context_eval_filters_covered_schemas_and_compares_baseline(monkeypatch):
    """Covered-schema runs should call out recall drops against a previous report."""

    class Summary:
        summary_id = "covered_schema"

    class FakeRegistry:
        def match_table_ref(self, table_ref: str) -> list[Summary]:
            return [Summary()] if table_ref.startswith("DB.PUBLIC.") else []

    monkeypatch.setattr(
        "sol01.schema.schema_context_eval.load_large_schema_summary_registry",
        lambda: FakeRegistry(),
    )

    report = run_schema_context_eval(
        [
            Task(instance_id="sf_family", db="DB", question="Show every historical sales table."),
            Task(instance_id="sf_missing", db="DB", question="Show customer sales details."),
        ],
        gold_tables_by_instance={
            "sf_family": ["DB.PUBLIC.SALES_2022", "DB.PUBLIC.SALES_2023"],
            "sf_missing": ["DB.PUBLIC.CUSTOMERS"],
        },
        config=SchemaContextConfig(object_cutoff=3),
        db_index_loader=lambda db: _db_index(),
        schema_context_cache_loader=lambda db, db_index, config: _schema_context_cache(),
        covered_only=True,
        baseline_tasks={
            "sf_family": {
                "pre_resolver_gold_recall": 1.0,
                "post_resolver_gold_recall": 1.0,
            },
            "sf_missing": {
                "pre_resolver_gold_recall": 1.0,
                "post_resolver_gold_recall": 1.0,
            },
        },
    )

    assert report.task_count == 2
    assert report.covered_task_count == 2
    assert report.tasks[0]["covered_summary_ids"] == ["covered_schema"]
    assert report.recall_regressions[0]["instance_id"] == "sf_missing"
    assert report.recall_regressions[0]["baseline_post_resolver_gold_recall"] == 1.0


def test_schema_context_eval_persists_report_and_hallucinated_column_failures(tmp_path: Path):
    """Persisted schema context eval artifacts should include trace-derived column failures."""

    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "sf_bad.json").write_text(
        json.dumps(
            {
                "instance_id": "sf_bad",
                "db": "DB",
                "status": "failed",
                "attempts": [
                    {
                        "validation": {
                            "ok": False,
                            "errors": ["Unknown column fake_metric on DB.PUBLIC.SALES_2022."],
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = run_schema_context_eval(
        [Task(instance_id="sf_family", db="DB", question="Show every historical sales table.")],
        gold_tables_by_instance={
            "sf_family": ["DB.PUBLIC.SALES_2022", "DB.PUBLIC.SALES_2023"],
        },
        config=SchemaContextConfig(object_cutoff=3),
        db_index_loader=lambda db: _db_index(),
        schema_context_cache_loader=lambda db, db_index, config: _schema_context_cache(),
        trace_dirs=[trace_dir],
    )
    output_dir = write_schema_context_eval_report(report, tmp_path / "schema_context_eval")

    assert report.hallucinated_column_failures[0]["instance_id"] == "sf_bad"
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "tasks.jsonl").exists()
    assert "Hallucinated Column Failures" in (output_dir / "summary.md").read_text(encoding="utf-8")


def test_schema_context_eval_cli_command_dispatches(monkeypatch, tmp_path: Path):
    """The CLI should expose schema-context-eval without using runtime solving paths."""

    runner = CliRunner()
    called: dict[str, Any] = {}
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text('{"instance_id": "sf001", "gold_tables": ["DB.PUBLIC.ORDERS"]}\n')

    class DummyReport:
        task_count = 1
        object_cutoff = 4
        covered_task_count = 1
        pre_resolver_gold_recall = 1.0
        pre_resolver_any_gold_recall = 1.0
        post_resolver_gold_recall = 1.0
        post_resolver_all_gold_recall = 1.0
        family_expansion_success = None
        average_prompt_reduction = 0.25
        failures: list[dict[str, Any]] = []
        tasks: list[dict[str, Any]] = []
        recall_regressions: list[dict[str, Any]] = []
        prompt_size_wins: list[dict[str, Any]] = []
        hallucinated_column_failures: list[dict[str, Any]] = []

        def summary(self) -> dict[str, Any]:
            return {"task_count": self.task_count}

        def payload(self) -> dict[str, Any]:
            return {"summary": self.summary(), "tasks": [], "failures": []}

    def fake_handle_schema_context_eval(**kwargs: Any) -> DummyReport:
        called.update(kwargs)
        return DummyReport()

    monkeypatch.setattr(
        cli_schema_context,
        "handle_schema_context_eval",
        fake_handle_schema_context_eval,
    )

    result = runner.invoke(
        cli.app,
        [
            "schema-context-eval",
            "sf001",
            "--gold-path",
            str(gold_path),
            "--object-cutoff",
            "4",
        ],
    )

    assert result.exit_code == 0
    assert called["selectors"] == ["sf001"]
    assert called["gold_path"] == gold_path
    assert called["object_cutoff"] == 4
    assert called["covered_only"] is False
    assert called["baseline_path"] is None
    assert called["trace_run_ids"] == []
    assert "Pre-resolver gold recall: 100.0% (1.000)" in result.output
    assert "Pre-resolver any-gold recall: 100.0% (1.000)" in result.output


def _db_index() -> dict[str, TableSchema]:
    return {
        "DB.PUBLIC.SALES_2022": _table("SALES_2022"),
        "DB.PUBLIC.SALES_2023": _table("SALES_2023"),
    }


def _table(name: str) -> TableSchema:
    return TableSchema(
        name=name,
        database_name="DB",
        schema_name="PUBLIC",
        full_name=f"DB.PUBLIC.{name}",
        ddl=f"CREATE TABLE {name} (ORDER_ID TEXT, AMOUNT NUMBER);",
        columns=[
            ColumnSchema(name="ORDER_ID", type="TEXT"),
            ColumnSchema(name="AMOUNT", type="NUMBER"),
        ],
        searchable_text=f"{name} sales amount",
    )


def _schema_context_cache() -> SchemaContextCache:
    objects = [
        SchemaObject(
            object_id="table:DB.PUBLIC.SALES_2022",
            object_type="table",
            name="SALES_2022",
            db="DB",
            table_name="DB.PUBLIC.SALES_2022",
            searchable_text="sales 2022 amount",
        ),
        SchemaObject(
            object_id="table:DB.PUBLIC.SALES_2023",
            object_type="table",
            name="SALES_2023",
            db="DB",
            table_name="DB.PUBLIC.SALES_2023",
            searchable_text="sales 2023 amount",
        ),
        SchemaObject(
            object_id="family:DB.PUBLIC:exact_sales:11111111",
            object_type="family",
            name="DB.PUBLIC.sales table family",
            db="DB",
            table_name="DB.PUBLIC.SALES_2022",
            searchable_text="historical sales table family",
            metadata={
                "canonical_member": "DB.PUBLIC.SALES_2022",
                "member_table_refs": ["DB.PUBLIC.SALES_2022", "DB.PUBLIC.SALES_2023"],
                "common_columns": ["ORDER_ID", "AMOUNT"],
            },
        ),
    ]
    chunks = [
        SchemaContextChunk(
            chunk_id="family:DB.PUBLIC:exact_sales:11111111::table_family",
            object_id="family:DB.PUBLIC:exact_sales:11111111",
            chunk_type="table_family",
            parent_object_ids=[
                "table:DB.PUBLIC.SALES_2022",
                "table:DB.PUBLIC.SALES_2023",
            ],
            evidence_text="historical sales table family 2022 2023",
            prompt_text="Sales table family with annual physical members.",
        ),
        SchemaContextChunk(
            chunk_id="table:DB.PUBLIC.SALES_2022::table",
            object_id="table:DB.PUBLIC.SALES_2022",
            chunk_type="table",
            evidence_text="sales table 2022 amount",
            prompt_text="Sales table for 2022.",
        ),
        SchemaContextChunk(
            chunk_id="table:DB.PUBLIC.SALES_2023::table",
            object_id="table:DB.PUBLIC.SALES_2023",
            chunk_type="table",
            evidence_text="sales table 2023 amount",
            prompt_text="Sales table for 2023.",
        ),
    ]
    return SchemaContextCache(
        db="DB",
        cache_key="test",
        cache_dir=Path("/tmp/test-schema-context-eval"),
        manifest={},
        objects=objects,
        chunks=chunks,
    )
