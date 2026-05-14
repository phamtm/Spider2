"""Tests for versioned schema metadata context caching."""

from __future__ import annotations

from pathlib import Path

import pytest

from sol01.models import ColumnSchema, TableSchema
from sol01.schema.schema_context_cache import (
    SchemaContextCacheError,
    SchemaContextCacheLockTimeout,
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


def test_builds_loads_and_validates_schema_context_cache_for_one_database(tmp_path):
    index = _build(tmp_path)

    assert index.cache_dir.exists()
    assert {path.name for path in index.cache_dir.iterdir()} == {
        "manifest.json",
        "objects.jsonl",
        "chunks.jsonl",
    }
    assert not (index.cache_dir / "embeddings.npy").exists()
    assert index.chunks

    loaded = load_current_schema_context_cache("DB", cache_root=tmp_path)

    assert loaded.cache_key == index.cache_key
    assert loaded.manifest["source_schema_hash"] == schema_source_hash(_db_index())
    assert loaded.manifest["curated_summary_registry_hash"]
    assert loaded.manifest["curated_summary_registry_version"] == "large-schema-summaries-v1"


def test_cache_key_changes_for_schema_versions_model_metadata_family_threshold_and_summaries():
    source_hash = schema_source_hash(_db_index())
    base = {
        "db": "DB",
        "source_schema_hash": source_hash,
        "object_builder_version": "objects-v1",
        "chunk_render_version": "chunks-v1",
        "family_similarity_threshold": 0.82,
        "curated_summary_registry_hash": "summary-hash-v1",
        "curated_summary_registry_version": "summaries-v1",
    }

    baseline = schema_context_cache_key(**base)

    assert schema_context_cache_key(**{**base, "source_schema_hash": "different"}) != baseline
    assert schema_context_cache_key(**{**base, "object_builder_version": "objects-v2"}) != baseline
    assert schema_context_cache_key(**{**base, "chunk_render_version": "chunks-v2"}) != baseline
    assert schema_context_cache_key(**{**base, "family_similarity_threshold": 0.9}) != baseline
    assert (
        schema_context_cache_key(**{**base, "curated_summary_registry_hash": "summary-hash-v2"})
        != baseline
    )
    assert (
        schema_context_cache_key(**{**base, "curated_summary_registry_version": "summaries-v2"})
        != baseline
    )


def test_stale_missing_cache_artifact_is_rebuilt_under_same_key(tmp_path):
    first = _build(tmp_path)
    (first.cache_dir / "chunks.jsonl").unlink()

    rebuilt = _build(tmp_path)

    assert rebuilt.cache_key == first.cache_key
    assert (rebuilt.cache_dir / "chunks.jsonl").exists()
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


def test_changed_builder_version_publishes_separate_version_directory(tmp_path):
    first = _build(tmp_path, object_builder_version="objects-v1")
    second = _build(tmp_path, object_builder_version="objects-v2")

    assert first.cache_key != second.cache_key
    assert _version_dir(tmp_path, "DB", first.cache_key).exists()
    assert _version_dir(tmp_path, "DB", second.cache_key).exists()


def test_changed_summary_registry_version_publishes_separate_version_directory(tmp_path):
    registry_path = tmp_path / "large_schema_summaries.json"
    registry_path.write_text('{"summaries": []}\n', encoding="utf-8")
    cache_root = tmp_path / "cache"

    first = _build(
        cache_root,
        curated_summary_registry_path=registry_path,
        curated_summary_registry_version="summaries-v1",
    )
    second = _build(
        cache_root,
        curated_summary_registry_path=registry_path,
        curated_summary_registry_version="summaries-v2",
    )

    assert first.cache_key != second.cache_key
    assert (
        first.manifest["curated_summary_registry_hash"]
        == second.manifest["curated_summary_registry_hash"]
    )
    assert first.manifest["curated_summary_registry_version"] == "summaries-v1"
    assert second.manifest["curated_summary_registry_version"] == "summaries-v2"
    assert _version_dir(cache_root, "DB", first.cache_key).exists()
    assert _version_dir(cache_root, "DB", second.cache_key).exists()
