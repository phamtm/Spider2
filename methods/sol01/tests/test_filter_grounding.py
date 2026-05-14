"""Focused tests for filter-grounding probe heuristics."""

from __future__ import annotations

from sol01.candidates.filter_grounding import (
    _column_looks_key_like,
    _probe_columns_for_table,
    _table_looks_like_lookup_name,
)
from sol01.models import ColumnSchema, TableSchema


def test_lookup_table_detection_uses_name_parts_instead_of_substrings() -> None:
    assert _table_looks_like_lookup_name("DB.PUBLIC.COUNTRY_REF")
    assert not _table_looks_like_lookup_name("DB.PUBLIC.DISCOUNTS")


def test_key_column_detection_does_not_treat_grid_as_an_id() -> None:
    assert _column_looks_key_like("ORDER_ID")
    assert not _column_looks_key_like("GRID")


def test_probe_columns_prefer_lookup_labels_and_keys() -> None:
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
