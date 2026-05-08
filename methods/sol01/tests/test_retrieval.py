"""Tests for schema retrieval utility helpers."""

from __future__ import annotations

from sol01.models import ColumnSchema, TableSchema
from sol01.schema.retrieval import _db_schema_summary, _sanitize_llm_tables

CUSTOMERS = "E_COMMERCE.E_COMMERCE.CUSTOMERS"
ORDERS = "E_COMMERCE.E_COMMERCE.ORDERS"


def _index() -> dict[str, TableSchema]:
    return {
        ORDERS: TableSchema(
            name=ORDERS,
            full_name=ORDERS,
            ddl="",
            columns=[
                ColumnSchema(name="ORDER_ID", type="NUMBER"),
                ColumnSchema(name="STATUS", type="TEXT", sample_values=["complete", "open"]),
            ],
            searchable_text="orders",
        ),
        CUSTOMERS: TableSchema(
            name=CUSTOMERS,
            full_name=CUSTOMERS,
            ddl="",
            columns=[ColumnSchema(name="CUSTOMER_ID", type="NUMBER", description="Customer key")],
            searchable_text="customers",
        ),
    }


def test_sanitize_llm_tables_keeps_valid_unique_names_and_suffix_matches():
    selected = _sanitize_llm_tables(
        ["orders", "missing", "E_COMMERCE.CUSTOMERS", "orders"],
        _index(),
    )

    assert selected == [ORDERS, CUSTOMERS]


def test_db_schema_summary_keeps_column_types_docs_and_samples():
    summary = _db_schema_summary(_index())

    assert f"Table {ORDERS}:" in summary
    assert "ORDER_ID [NUMBER]" in summary
    assert "STATUS [TEXT] - sample values: complete, open" in summary
    assert "CUSTOMER_ID [NUMBER] - Customer key" in summary
