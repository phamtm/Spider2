"""Focused tests for filter-grounding probe heuristics."""

from __future__ import annotations

from sol01.candidates.filter_grounding import _probe_columns_for_table
from sol01.models import ColumnSchema, TableSchema


def test_probe_columns_prefer_exact_column_name_match() -> None:
    table = TableSchema(
        name="CUSTOMERS",
        full_name="DB.PUBLIC.CUSTOMERS",
        ddl="CREATE TABLE CUSTOMERS (COUNTRY TEXT, COUNTRY_CODE TEXT, NOTES TEXT)",
        columns=[
            ColumnSchema(name="COUNTRY", type="TEXT"),
            ColumnSchema(name="COUNTRY_CODE", type="TEXT"),
            ColumnSchema(name="NOTES", type="TEXT"),
        ],
        searchable_text="customers country",
    )

    assert _probe_columns_for_table(table, "country", "United States") == ["COUNTRY"]


def test_probe_columns_fall_back_to_related_column_names() -> None:
    table = TableSchema(
        name="COUNTRY_REF",
        full_name="DB.PUBLIC.COUNTRY_REF",
        ddl="CREATE TABLE COUNTRY_REF (COUNTRY_CODE TEXT, COUNTRY_NAME TEXT, NOTES TEXT)",
        columns=[
            ColumnSchema(name="COUNTRY_CODE", type="TEXT"),
            ColumnSchema(name="COUNTRY_NAME", type="TEXT"),
            ColumnSchema(name="NOTES", type="TEXT"),
        ],
        searchable_text="country ref",
    )

    assert _probe_columns_for_table(table, "country", "United States") == [
        "COUNTRY_CODE",
        "COUNTRY_NAME",
    ]
