"""Tests for generated per-database schema-profile artifacts."""

from __future__ import annotations

import json

import pytest

from sol01.models import FamilyProfile, SchemaProfileCatalog, TableProfile
from sol01.schema.schema_profiles import (
    build_schema_profile,
    compact_table_keys_for_profiles,
    load_schema_profile_catalog,
)


def test_schema_profile_models_reject_question_leakage_and_benchmark_ids():
    with pytest.raises(ValueError, match="forbidden hint token"):
        TableProfile(
            profile_id="bad_profile",
            abstraction_kind="table",
            table_name="DB.PUBLIC.ORDERS",
            covered_tables=["DB.PUBLIC.ORDERS"],
            grain_hint="One row per order.",
            naming_rules=["Use exact names."],
            compact_semantic_summary="This came from sf_bq089.",
            aliases=["orders"],
            theme_terms=["orders"],
            confidence=0.9,
            provenance_inputs=["raw/orders.json"],
            source_column_count=3,
            source_sample_row_count=0,
        )

    with pytest.raises(ValueError, match="declarative"):
        TableProfile(
            profile_id="question_profile",
            abstraction_kind="table",
            table_name="DB.PUBLIC.ORDERS",
            covered_tables=["DB.PUBLIC.ORDERS"],
            grain_hint="One row per order.",
            naming_rules=["Use exact names."],
            compact_semantic_summary="What is the right order table?",
            aliases=["orders"],
            theme_terms=["orders"],
            confidence=0.9,
            provenance_inputs=["raw/orders.json"],
            source_column_count=3,
            source_sample_row_count=0,
        )


def test_compact_table_keys_cover_wide_tables_and_families():
    catalog = SchemaProfileCatalog(
        db="DB",
        source_schema_hash="schema-hash-v1",
        table_profiles=[
            TableProfile(
                profile_id="orders_profile",
                abstraction_kind="wide_table",
                table_name="DB.PUBLIC.ORDERS",
                covered_tables=["DB.PUBLIC.ORDERS"],
                grain_hint="One row per order.",
                naming_rules=["Use exact names."],
                compact_semantic_summary="Order facts.",
                aliases=["orders"],
                theme_terms=["orders"],
                confidence=0.9,
                provenance_inputs=["raw/orders.json"],
                source_column_count=80,
                source_sample_row_count=0,
            )
        ],
        family_profiles=[
            FamilyProfile(
                profile_id="sales_family",
                abstraction_kind="exact_family",
                family_selector="stem=sales;kind=exact_family",
                covered_tables=["DB.PUBLIC.SALES_2022", "DB.PUBLIC.SALES_2023"],
                canonical_table="DB.PUBLIC.SALES_2022",
                grain_hint="One row per sale.",
                naming_rules=["Use exact names."],
                compact_semantic_summary="Sales family.",
                aliases=["sales"],
                theme_terms=["sales"],
                confidence=0.95,
                provenance_inputs=["raw/sales_2022.json", "raw/sales_2023.json"],
                member_count=2,
            )
        ],
    )

    assert compact_table_keys_for_profiles(catalog) == {
        "DB.PUBLIC.ORDERS",
        "DB.PUBLIC.SALES_2022",
        "DB.PUBLIC.SALES_2023",
    }


def test_build_schema_profile_writes_per_db_catalog_and_manifest(tmp_path):
    metadata_root = tmp_path / "metadata"
    db_dir = metadata_root / "TEST_DB" / "PUBLIC"
    db_dir.mkdir(parents=True)
    (db_dir / "DDL.csv").write_text(
        "table_name,DDL\nPUBLIC.ORDERS,CREATE TABLE ORDERS (ORDER_ID TEXT, STATUS TEXT)\n",
        encoding="utf-8",
    )
    (db_dir / "ORDERS.json").write_text(
        json.dumps(
            {
                "table_name": "PUBLIC.ORDERS",
                "table_fullname": "TEST_DB.PUBLIC.ORDERS",
                "column_names": ["ORDER_ID", "STATUS"],
                "column_types": ["TEXT", "TEXT"],
                "description": ["Primary key.", "Order status."],
                "sample_rows": [{"ORDER_ID": "o1", "STATUS": "open"}],
            }
        ),
        encoding="utf-8",
    )

    result = build_schema_profile(
        "TEST_DB",
        metadata_root=metadata_root,
        profile_root=tmp_path / "profiles",
    )

    assert result.changed is True
    assert result.catalog_path.name == "catalog.json"
    assert result.manifest_path.name == "manifest.json"
    assert load_schema_profile_catalog("TEST_DB", profile_root=tmp_path / "profiles") is not None
    assert result.catalog.table_profiles[0].table_name == "TEST_DB.PUBLIC.ORDERS"
    assert result.manifest.artifact_hash


def test_build_schema_profile_skips_rewrite_when_inputs_are_unchanged(tmp_path):
    metadata_root = tmp_path / "metadata"
    db_dir = metadata_root / "TEST_DB" / "PUBLIC"
    db_dir.mkdir(parents=True)
    (db_dir / "DDL.csv").write_text(
        "table_name,DDL\nPUBLIC.ORDERS,CREATE TABLE ORDERS (ORDER_ID TEXT)\n",
        encoding="utf-8",
    )
    (db_dir / "ORDERS.json").write_text(
        json.dumps(
            {
                "table_name": "PUBLIC.ORDERS",
                "table_fullname": "TEST_DB.PUBLIC.ORDERS",
                "column_names": ["ORDER_ID"],
                "column_types": ["TEXT"],
                "description": ["Primary key."],
                "sample_rows": [{"ORDER_ID": "o1"}],
            }
        ),
        encoding="utf-8",
    )

    first = build_schema_profile(
        "TEST_DB",
        metadata_root=metadata_root,
        profile_root=tmp_path / "profiles",
    )
    second = build_schema_profile(
        "TEST_DB",
        metadata_root=metadata_root,
        profile_root=tmp_path / "profiles",
    )

    assert first.changed is True
    assert second.changed is False
    assert first.manifest.artifact_hash == second.manifest.artifact_hash
