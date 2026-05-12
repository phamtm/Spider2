"""Tests for offline schema retrieval coverage evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from sol01 import cli
from sol01.infra.config import SchemaRetrievalConfig
from sol01.models import ColumnSchema, RetrievalChunk, SchemaObject, TableSchema, Task
from sol01.schema.retrieval_eval import load_gold_tables, run_retrieval_eval
from sol01.schema.retrieval_index import SchemaRetrievalIndex, _build_sparse_index


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


def test_retrieval_eval_reports_gold_coverage_family_success_and_failures():
    """The evaluator should measure pre/post resolver recall and failure evidence."""

    db_index = _db_index()
    index = _retrieval_index()
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

    report = run_retrieval_eval(
        tasks,
        gold_tables_by_instance={
            "sf_family": ["DB.PUBLIC.SALES_2022", "DB.PUBLIC.SALES_2023"],
            "sf_missing": ["DB.PUBLIC.CUSTOMERS"],
        },
        config=SchemaRetrievalConfig(object_top_k=3),
        db_index_loader=lambda db: db_index,
        retrieval_index_loader=lambda db, db_index, config: index,
    )

    assert report.task_count == 2
    assert report.pre_resolver_any_gold_recall == 0.5
    assert report.post_resolver_all_gold_recall == 0.5
    assert report.family_expansion_success == 1.0
    assert report.average_prompt_reduction < 1.0
    assert report.failures == [
        {
            "instance_id": "sf_missing",
            "db": "DB",
            "missing_gold_tables": ["DB.PUBLIC.CUSTOMERS"],
            "top_evidence": report.failures[0]["top_evidence"],
        }
    ]
    assert report.failures[0]["top_evidence"][0]["chunk_id"]


def test_retrieval_eval_cli_command_dispatches(monkeypatch, tmp_path: Path):
    """The CLI should expose retrieval-eval without using runtime solving paths."""

    runner = CliRunner()
    called: dict[str, Any] = {}
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text('{"instance_id": "sf001", "gold_tables": ["DB.PUBLIC.ORDERS"]}\n')

    class DummyReport:
        task_count = 1
        object_cutoff = 4
        pre_resolver_any_gold_recall = 1.0
        post_resolver_all_gold_recall = 1.0
        family_expansion_success = None
        average_prompt_reduction = 0.25
        failures: list[dict[str, Any]] = []
        tasks: list[dict[str, Any]] = []

        def summary(self) -> dict[str, Any]:
            return {"task_count": self.task_count}

    def fake_handle_retrieval_eval(**kwargs: Any) -> DummyReport:
        called.update(kwargs)
        return DummyReport()

    monkeypatch.setattr(cli, "handle_retrieval_eval", fake_handle_retrieval_eval)

    result = runner.invoke(
        cli.app,
        [
            "retrieval-eval",
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


def _retrieval_index() -> SchemaRetrievalIndex:
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
        RetrievalChunk(
            chunk_id="family:DB.PUBLIC:exact_sales:11111111::table_family",
            object_id="family:DB.PUBLIC:exact_sales:11111111",
            chunk_type="table_family",
            parent_object_ids=[
                "table:DB.PUBLIC.SALES_2022",
                "table:DB.PUBLIC.SALES_2023",
            ],
            bm25_text="historical sales table family 2022 2023",
            prompt_text="Sales table family with annual physical members.",
        ),
        RetrievalChunk(
            chunk_id="table:DB.PUBLIC.SALES_2022::table",
            object_id="table:DB.PUBLIC.SALES_2022",
            chunk_type="table",
            bm25_text="sales table 2022 amount",
            prompt_text="Sales table for 2022.",
        ),
        RetrievalChunk(
            chunk_id="table:DB.PUBLIC.SALES_2023::table",
            object_id="table:DB.PUBLIC.SALES_2023",
            chunk_type="table",
            bm25_text="sales table 2023 amount",
            prompt_text="Sales table for 2023.",
        ),
    ]
    return SchemaRetrievalIndex(
        db="DB",
        cache_key="test",
        cache_dir=Path("/tmp/test-retrieval-eval"),
        manifest={},
        objects=objects,
        chunks=chunks,
        sparse=_build_sparse_index(chunks),
    )
