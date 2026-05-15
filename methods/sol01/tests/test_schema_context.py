"""Tests for deterministic schema context availability."""

from __future__ import annotations

from pathlib import Path

from sol01.infra.fs_cache import stable_hash, write_json
from sol01.models import (
    ColumnSchema,
    SchemaObject,
    SchemaProfileCatalog,
    SchemaProfileManifest,
    TableProfile,
    TableSchema,
)
from sol01.schema.schema_context import build_available_schema_context
from sol01.schema.schema_context_cache import SchemaContextCache, build_schema_context_cache
from sol01.schema.schema_profiles import (
    SCHEMA_PROFILE_BUILDER_VERSION,
    SCHEMA_PROFILE_SUMMARIZER_VERSION,
    SCHEMA_PROFILE_TEMPLATE_VERSION,
    schema_profile_catalog_path,
    schema_profile_manifest_path,
)


def test_full_metadata_context_returns_all_objects_in_stable_order():
    cache = _fake_cache()

    objects, diagnostics = build_available_schema_context(cache)

    selected_ids = {obj.schema_object.object_id for obj in objects}
    assert selected_ids == {
        "family:DB.PUBLIC:orders_family:deadbeef",
        "table:DB.PUBLIC.CUSTOMERS",
        "table:DB.PUBLIC.ORDERS",
        "column:DB.PUBLIC.ORDERS#AMOUNT",
        "column:DB.PUBLIC.ORDERS#STATUS",
        "join_candidate:DB.PUBLIC.ORDERS#CUSTOMER_ID->DB.PUBLIC.CUSTOMERS#CUSTOMER_ID:abcdef12",
        "sample_value:DB.PUBLIC.ORDERS#STATUS:11111111",
    }
    assert diagnostics["context_mode"] == "full_metadata"
    assert diagnostics["context_counts"] == {
        "objects_total": 7,
        "available_objects": 7,
    }
    assert [obj.schema_object.object_id for obj in objects] == [
        "family:DB.PUBLIC:orders_family:deadbeef",
        "table:DB.PUBLIC.CUSTOMERS",
        "table:DB.PUBLIC.ORDERS",
        "column:DB.PUBLIC.ORDERS#AMOUNT",
        "column:DB.PUBLIC.ORDERS#STATUS",
        "join_candidate:DB.PUBLIC.ORDERS#CUSTOMER_ID->DB.PUBLIC.CUSTOMERS#CUSTOMER_ID:abcdef12",
        "sample_value:DB.PUBLIC.ORDERS#STATUS:11111111",
    ]


def test_adaptive_profile_context_uses_profile_text_without_hiding_unprofiled_tables(tmp_path):
    profile_root = tmp_path / "schema_profiles"
    _write_profile_artifact(profile_root)
    covered_table = TableSchema(
        name="_20240103",
        database_name="GITHUB_REPOS_DATE",
        schema_name="DAY",
        full_name="GITHUB_REPOS_DATE.DAY._20240103",
        ddl="CREATE TABLE _20240103 (SECRET_DDL_MARKER TEXT);",
        columns=[
            ColumnSchema(name="public", type="BOOLEAN"),
            ColumnSchema(name="actor", type="VARIANT"),
            ColumnSchema(name="created_at", type="TIMESTAMP"),
            ColumnSchema(name="type", type="TEXT"),
            ColumnSchema(name="repo", type="VARIANT"),
            ColumnSchema(name="payload", type="VARIANT"),
            ColumnSchema(name="id", type="TEXT"),
            ColumnSchema(name="other", type="VARIANT"),
            ColumnSchema(name="org", type="VARIANT"),
        ],
        sample_rows=[],
        searchable_text="github events",
    )
    uncovered_table = TableSchema(
        name="REPOSITORIES",
        database_name="GITHUB_REPOS_DATE",
        schema_name="DAY",
        full_name="GITHUB_REPOS_DATE.DAY.REPOSITORIES",
        ddl="CREATE TABLE REPOSITORIES (ID TEXT, NAME TEXT);",
        columns=[ColumnSchema(name="ID", type="TEXT"), ColumnSchema(name="NAME", type="TEXT")],
        sample_rows=[],
        searchable_text="repository metadata",
    )
    cache = build_schema_context_cache(
        "GITHUB_REPOS_DATE",
        db_index={
            "GITHUB_REPOS_DATE.DAY._20240103": covered_table,
            "GITHUB_REPOS_DATE.DAY.REPOSITORIES": uncovered_table,
        },
        cache_root=tmp_path,
        schema_profile_root=profile_root,
    )

    schema_context_objects, diagnostics = build_available_schema_context(cache)

    assert diagnostics["context_mode"] == "adaptive_profiles"
    schema_context_ids = [item.schema_object.object_id for item in schema_context_objects]
    assert "table:GITHUB_REPOS_DATE.DAY._20240103" in schema_context_ids
    assert "column:GITHUB_REPOS_DATE.DAY._20240103#payload" not in schema_context_ids
    assert "table:GITHUB_REPOS_DATE.DAY.REPOSITORIES" in schema_context_ids
    assert "column:GITHUB_REPOS_DATE.DAY.REPOSITORIES#NAME" in schema_context_ids
    profile_object = next(
        item
        for item in schema_context_objects
        if item.schema_object.object_id == "table:GITHUB_REPOS_DATE.DAY._20240103"
    )
    uncovered_object = next(
        item
        for item in schema_context_objects
        if item.schema_object.object_id == "table:GITHUB_REPOS_DATE.DAY.REPOSITORIES"
    )
    assert "Schema profile: github_day_profile." in profile_object.planning_text
    assert "SECRET_DDL_MARKER" not in profile_object.planning_text
    assert "Columns: ID [TEXT], NAME [TEXT]." in uncovered_object.planning_text


def _write_profile_artifact(profile_root: Path) -> None:
    schema_profile_catalog_path("GITHUB_REPOS_DATE", profile_root=profile_root).parent.mkdir(
        parents=True, exist_ok=True
    )
    catalog = SchemaProfileCatalog(
        db="GITHUB_REPOS_DATE",
        source_schema_hash="schema-hash-v1",
        table_profiles=[
            TableProfile(
                profile_id="github_day_profile",
                abstraction_kind="wide_table",
                table_name="GITHUB_REPOS_DATE.DAY._20240103",
                covered_tables=["GITHUB_REPOS_DATE.DAY._20240103"],
                grain_hint="One row per github event.",
                time_columns=["created_at"],
                naming_rules=["Use exact table name GITHUB_REPOS_DATE.DAY._20240103."],
                compact_semantic_summary="Daily github archive event table.",
                aliases=["github day events"],
                theme_terms=["github", "events"],
                confidence=0.9,
                provenance_inputs=["raw/day/_20240103.json"],
                source_column_count=9,
                source_sample_row_count=0,
            )
        ],
    )
    manifest = SchemaProfileManifest(
        db="GITHUB_REPOS_DATE",
        source_schema_hash="schema-hash-v1",
        builder_version=SCHEMA_PROFILE_BUILDER_VERSION,
        summarizer_version=SCHEMA_PROFILE_SUMMARIZER_VERSION,
        prompt_template_version=SCHEMA_PROFILE_TEMPLATE_VERSION,
        generated_at="2026-05-15T00:00:00+00:00",
        artifact_hash=stable_hash(catalog.model_dump(mode="json")),
        table_profile_count=1,
        family_profile_count=0,
    )
    write_json(
        schema_profile_catalog_path("GITHUB_REPOS_DATE", profile_root=profile_root),
        catalog.model_dump(mode="json"),
    )
    write_json(
        schema_profile_manifest_path("GITHUB_REPOS_DATE", profile_root=profile_root),
        manifest.model_dump(mode="json"),
    )


def _fake_cache() -> SchemaContextCache:
    objects = [
        SchemaObject(
            object_id="table:DB.PUBLIC.ORDERS",
            object_type="table",
            name="ORDERS",
            db="DB",
            table_name="DB.PUBLIC.ORDERS",
            searchable_text="orders status customer",
        ),
        SchemaObject(
            object_id="table:DB.PUBLIC.CUSTOMERS",
            object_type="table",
            name="CUSTOMERS",
            db="DB",
            table_name="DB.PUBLIC.CUSTOMERS",
            searchable_text="customers",
        ),
        SchemaObject(
            object_id="column:DB.PUBLIC.ORDERS#STATUS",
            object_type="column",
            name="STATUS",
            db="DB",
            table_name="DB.PUBLIC.ORDERS",
            column_name="STATUS",
            searchable_text="status closed open",
        ),
        SchemaObject(
            object_id="column:DB.PUBLIC.ORDERS#AMOUNT",
            object_type="column",
            name="AMOUNT",
            db="DB",
            table_name="DB.PUBLIC.ORDERS",
            column_name="AMOUNT",
            searchable_text="amount revenue",
        ),
        SchemaObject(
            object_id="sample_value:DB.PUBLIC.ORDERS#STATUS:11111111",
            object_type="sample_value",
            name="closed",
            db="DB",
            table_name="DB.PUBLIC.ORDERS",
            column_name="STATUS",
            searchable_text="closed",
        ),
        SchemaObject(
            object_id="join_candidate:DB.PUBLIC.ORDERS#CUSTOMER_ID->DB.PUBLIC.CUSTOMERS#CUSTOMER_ID:abcdef12",
            object_type="join_candidate",
            name="ORDERS_CUSTOMERS",
            db="DB",
            searchable_text="orders customers customer_id",
        ),
        SchemaObject(
            object_id="family:DB.PUBLIC:orders_family:deadbeef",
            object_type="family",
            name="orders_family",
            db="DB",
            searchable_text="orders table family",
            metadata={"member_table_refs": ["DB.PUBLIC.ORDERS"]},
        ),
    ]
    return SchemaContextCache(
        db="DB",
        cache_key="test",
        cache_dir=Path("/tmp/test-schema-index"),
        manifest={},
        objects=objects,
    )
