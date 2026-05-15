"""Tests for versioned schema metadata context caching."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sol01.models import ColumnSchema, TableSchema
from sol01.schema.large_schema_summaries import load_large_schema_summary_registry
from sol01.schema.schema_context_cache import (
    SchemaContextCacheError,
    SchemaContextCacheLockTimeout,
    _covered_table_keys,
    _version_dir,
    build_schema_context_cache,
    load_current_schema_context_cache,
    schema_context_cache_key,
    schema_source_hash,
)


def _db_index(*, extra_column: bool = False) -> dict[str, TableSchema]:
    """Return a compact schema index with table, column, join, and sample chunks."""

    order_columns = [
        ColumnSchema(name="ORDER_ID", type="TEXT"),
        ColumnSchema(name="CUSTOMER_ID", type="TEXT"),
        ColumnSchema(name="STATUS", type="TEXT"),
    ]
    if extra_column:
        order_columns.append(ColumnSchema(name="ORDER_TOTAL", type="NUMBER"))
    return {
        "DB.PUBLIC.ORDERS": TableSchema(
            name="ORDERS",
            database_name="DB",
            schema_name="PUBLIC",
            full_name="DB.PUBLIC.ORDERS",
            ddl="",
            columns=order_columns,
            sample_rows=[{"STATUS": "open"}, {"STATUS": "closed"}, {"STATUS": "open"}],
            searchable_text="orders",
        ),
        "DB.PUBLIC.CUSTOMERS": TableSchema(
            name="CUSTOMERS",
            database_name="DB",
            schema_name="PUBLIC",
            full_name="DB.PUBLIC.CUSTOMERS",
            ddl="",
            columns=[ColumnSchema(name="CUSTOMER_ID", type="TEXT")],
            sample_rows=[],
            searchable_text="customers",
        ),
    }


def _build(tmp_path: Path, **kwargs):
    return build_schema_context_cache(
        "DB",
        db_index=_db_index(),
        cache_root=tmp_path,
        lock_timeout_seconds=0.1,
        lock_poll_seconds=0.01,
        **kwargs,
    )


def _write_custom_summary_registry(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "summary_id": "orders_custom_summary",
                        "schema_copies": [
                            {"database": "DB", "schema_name": "PUBLIC"},
                        ],
                        "match": {"table_names": ["ORDERS"]},
                        "purpose": "Custom order lifecycle fact table.",
                        "grain": "One row per order in the custom registry.",
                        "stable_columns": ["ORDER_ID", "CUSTOMER_ID", "STATUS"],
                        "repeated_column_rules": ["No repeated physical table family."],
                        "quote_spelling_rules": ["Use ORDERS exactly as spelled."],
                        "examples": ["ORDERS", "ORDERS", "ORDERS"],
                        "aliases": ["custom order registry", "order lifecycle"],
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_builds_loads_and_validates_schema_context_cache_for_one_database(tmp_path):
    index = _build(tmp_path)

    assert index.cache_dir.exists()
    assert {path.name for path in index.cache_dir.iterdir()} == {"manifest.json", "objects.jsonl"}

    loaded = load_current_schema_context_cache("DB", cache_root=tmp_path)

    assert loaded.cache_key == index.cache_key
    assert loaded.manifest["source_schema_hash"] == schema_source_hash(_db_index())
    assert loaded.manifest["curated_summary_registry_hash"]
    assert loaded.manifest["curated_summary_registry_version"] == "large-schema-summaries-v1"
    assert loaded.manifest["context_mode"] == "full_metadata"


def test_cache_key_changes_for_schema_versions_model_metadata_family_threshold_and_summaries():
    source_hash = schema_source_hash(_db_index())
    base = {
        "db": "DB",
        "source_schema_hash": source_hash,
        "family_similarity_threshold": 0.82,
        "curated_summary_registry_hash": "summary-hash-v1",
    }

    baseline = schema_context_cache_key(**base)

    assert schema_context_cache_key(**{**base, "source_schema_hash": "different"}) != baseline
    assert schema_context_cache_key(**{**base, "family_similarity_threshold": 0.9}) != baseline
    assert (
        schema_context_cache_key(**{**base, "curated_summary_registry_hash": "summary-hash-v2"})
        != baseline
    )


def test_stale_missing_cache_artifact_is_rebuilt_under_same_key(tmp_path):
    first = _build(tmp_path)
    (first.cache_dir / "objects.jsonl").unlink()

    rebuilt = _build(tmp_path)

    assert rebuilt.cache_key == first.cache_key
    assert (rebuilt.cache_dir / "objects.jsonl").exists()
    assert list(rebuilt.cache_dir.parent.glob(".*.invalid.*"))


def test_build_lock_waits_bounded_time_when_no_current_cache_exists(tmp_path):
    lock_path = tmp_path / "DB" / "build.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text('{"token": "other"}\n', encoding="utf-8")

    with pytest.raises(SchemaContextCacheLockTimeout):
        _build(tmp_path)


def test_build_lock_reloads_existing_cache_instead_of_racing(tmp_path):
    first = _build(tmp_path)
    lock_path = tmp_path / "DB" / "build.lock"
    lock_path.write_text('{"token": "other"}\n', encoding="utf-8")

    loaded = _build(tmp_path)

    assert loaded.cache_key == first.cache_key


def test_missing_current_pointer_is_reported(tmp_path):
    with pytest.raises(
        SchemaContextCacheError, match="missing current schema context cache pointer"
    ):
        load_current_schema_context_cache("DB", cache_root=tmp_path)


def test_changed_schema_source_publishes_separate_version_directory(tmp_path):
    first = _build(tmp_path)
    second = build_schema_context_cache(
        "DB",
        db_index=_db_index(extra_column=True),
        cache_root=tmp_path,
        lock_timeout_seconds=0.1,
        lock_poll_seconds=0.01,
    )

    assert first.cache_key != second.cache_key
    assert _version_dir(tmp_path, "DB", first.cache_key).exists()
    assert _version_dir(tmp_path, "DB", second.cache_key).exists()


def test_changed_summary_registry_content_publishes_separate_version_directory(tmp_path):
    registry_path = tmp_path / "large_schema_summaries.json"
    registry_path.write_text('{"summaries": []}\n', encoding="utf-8")
    cache_root = tmp_path / "cache"

    first = _build(cache_root, curated_summary_registry_path=registry_path)
    _write_custom_summary_registry(registry_path)
    second = _build(cache_root, curated_summary_registry_path=registry_path)

    assert first.cache_key != second.cache_key
    assert (
        first.manifest["curated_summary_registry_hash"]
        != second.manifest["curated_summary_registry_hash"]
    )
    assert first.manifest["curated_summary_registry_version"] == "large-schema-summaries-v1"
    assert second.manifest["curated_summary_registry_version"] == "large-schema-summaries-v1"
    assert _version_dir(cache_root, "DB", first.cache_key).exists()
    assert _version_dir(cache_root, "DB", second.cache_key).exists()


def test_custom_summary_registry_drives_coverage_and_object_metadata(tmp_path):
    registry_path = _write_custom_summary_registry(tmp_path / "custom_summaries.json")
    cache = _build(tmp_path / "cache", curated_summary_registry_path=registry_path)

    object_ids = {obj.object_id for obj in cache.objects}
    orders_object = next(obj for obj in cache.objects if obj.object_id == "table:DB.PUBLIC.ORDERS")

    assert cache.manifest["context_mode"] == "compact_catalog"
    assert "column:DB.PUBLIC.ORDERS#STATUS" not in object_ids
    assert "table:DB.PUBLIC.CUSTOMERS" in object_ids
    assert "column:DB.PUBLIC.CUSTOMERS#CUSTOMER_ID" not in object_ids
    assert orders_object.metadata["summary_ids"] == ["orders_custom_summary"]
    assert orders_object.metadata["large_schema_summaries"][0]["text"].startswith(
        "Large-schema summary: orders_custom_summary."
    )


def test_compact_catalog_keeps_uncovered_tables_without_join_candidates(tmp_path):
    covered_table = TableSchema(
        name="_20240103",
        database_name="GITHUB_REPOS_DATE",
        schema_name="DAY",
        full_name="GITHUB_REPOS_DATE.DAY._20240103",
        ddl="",
        columns=[
            ColumnSchema(name="public", type="BOOLEAN"),
            ColumnSchema(name="actor", type="VARIANT"),
            ColumnSchema(name="created_at", type="TIMESTAMP"),
        ],
        sample_rows=[],
        searchable_text="github events",
    )
    uncovered_table = TableSchema(
        name="REPOSITORIES",
        database_name="GITHUB_REPOS_DATE",
        schema_name="DAY",
        full_name="GITHUB_REPOS_DATE.DAY.REPOSITORIES",
        ddl="",
        columns=[ColumnSchema(name="ID", type="TEXT"), ColumnSchema(name="NAME", type="TEXT")],
        sample_rows=[],
        searchable_text="repository metadata",
    )
    db_index = {
        "GITHUB_REPOS_DATE.DAY._20240103": covered_table,
        "GITHUB_REPOS_DATE.DAY.REPOSITORIES": uncovered_table,
    }
    cache = build_schema_context_cache(
        "GITHUB_REPOS_DATE",
        db_index=db_index,
        cache_root=tmp_path,
        lock_timeout_seconds=0.1,
    )

    object_ids = {obj.object_id for obj in cache.objects}
    object_types = {obj.object_type for obj in cache.objects}

    assert cache.manifest["context_mode"] == "compact_catalog"
    assert "table:GITHUB_REPOS_DATE.DAY._20240103" in object_ids
    assert "table:GITHUB_REPOS_DATE.DAY.REPOSITORIES" in object_ids
    assert not any(oid.startswith("column:") for oid in object_ids)
    assert not any(oid.startswith("join_candidate:") for oid in object_ids)
    assert "table" in object_types

    summary_object = next(
        obj for obj in cache.objects if obj.object_id == "table:GITHUB_REPOS_DATE.DAY._20240103"
    )
    assert summary_object.metadata.get("large_schema_summaries")


def test_compact_catalog_preserves_table_object_for_every_input_table(tmp_path):
    registry_path = _write_custom_summary_registry(tmp_path / "custom_summaries.json")
    cache = _build(tmp_path / "cache", curated_summary_registry_path=registry_path)

    visible_tables = {
        obj.table_name
        for obj in cache.objects
        if obj.object_type == "table" and obj.table_name is not None
    }

    assert visible_tables == {"DB.PUBLIC.CUSTOMERS", "DB.PUBLIC.ORDERS"}


def test_covered_table_keys_matches_tables_against_summary_registry():
    db_index = {
        "GITHUB_REPOS_DATE.DAY._20240103": TableSchema(
            name="_20240103",
            database_name="GITHUB_REPOS_DATE",
            schema_name="DAY",
            full_name="GITHUB_REPOS_DATE.DAY._20240103",
            ddl="",
            columns=[ColumnSchema(name="public", type="BOOLEAN")],
            sample_rows=[],
            searchable_text="github events",
        ),
        "GITHUB_REPOS_DATE.DAY.REPOSITORIES": TableSchema(
            name="REPOSITORIES",
            database_name="GITHUB_REPOS_DATE",
            schema_name="DAY",
            full_name="GITHUB_REPOS_DATE.DAY.REPOSITORIES",
            ddl="",
            columns=[ColumnSchema(name="ID", type="TEXT")],
            sample_rows=[],
            searchable_text="repositories",
        ),
    }
    registry = load_large_schema_summary_registry()

    covered = _covered_table_keys(db_index, registry)

    assert covered == {"GITHUB_REPOS_DATE.DAY._20240103"}
