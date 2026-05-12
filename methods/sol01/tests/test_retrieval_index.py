"""Tests for versioned schema retrieval index caching."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sol01.models import ColumnSchema, TableSchema
from sol01.schema.retrieval_index import (
    RetrievalIndexError,
    RetrievalIndexLockTimeout,
    _publish_version_directory,
    _version_dir,
    _write_current_pointer,
    build_retrieval_index,
    load_current_retrieval_index,
    retrieval_index_cache_key,
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
    return build_retrieval_index(
        "DB",
        db_index=_db_index(),
        cache_root=tmp_path,
        lock_timeout_seconds=0.1,
        lock_poll_seconds=0.01,
        **kwargs,
    )


def test_builds_loads_and_validates_retrieval_index_for_one_database(tmp_path):
    index = _build(tmp_path)

    assert index.cache_dir.exists()
    assert {path.name for path in index.cache_dir.iterdir()} == {
        "manifest.json",
        "objects.jsonl",
        "chunks.jsonl",
        "sparse.json",
    }
    assert not (index.cache_dir / "embeddings.npy").exists()
    assert index.sparse["chunk_ids"] == [chunk.chunk_id for chunk in index.chunks]
    assert max(index.sparse["document_frequency"].values()) <= len(index.chunks)

    loaded = load_current_retrieval_index("DB", cache_root=tmp_path)

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

    baseline = retrieval_index_cache_key(**base)

    assert retrieval_index_cache_key(**{**base, "source_schema_hash": "different"}) != baseline
    assert retrieval_index_cache_key(**{**base, "object_builder_version": "objects-v2"}) != baseline
    assert retrieval_index_cache_key(**{**base, "chunk_render_version": "chunks-v2"}) != baseline
    assert retrieval_index_cache_key(**{**base, "family_similarity_threshold": 0.9}) != baseline
    assert (
        retrieval_index_cache_key(**{**base, "curated_summary_registry_hash": "summary-hash-v2"})
        != baseline
    )
    assert (
        retrieval_index_cache_key(**{**base, "curated_summary_registry_version": "summaries-v2"})
        != baseline
    )


def test_final_dir_already_exists_is_not_overwritten(tmp_path):
    temp_dir = tmp_path / "temp-version"
    final_dir = tmp_path / "final-version"
    temp_dir.mkdir()
    final_dir.mkdir()
    (final_dir / "marker.txt").write_text("keep", encoding="utf-8")

    published = _publish_version_directory(temp_dir, final_dir)

    assert published is False
    assert not temp_dir.exists()
    assert (final_dir / "marker.txt").read_text(encoding="utf-8") == "keep"


def test_current_pointer_update_uses_os_replace(tmp_path, monkeypatch):
    calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def spy_replace(source, destination):
        calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", spy_replace)
    current_path = tmp_path / "DB" / "current.json"
    cache_dir = tmp_path / "DB" / "versions" / "abc"

    _write_current_pointer(current_path, db="DB", cache_key="abc", cache_dir=cache_dir)

    assert calls
    assert calls[0][1] == current_path
    assert json.loads(current_path.read_text(encoding="utf-8")) == {
        "cache_dir": str(cache_dir),
        "cache_key": "abc",
        "db": "DB",
    }
    assert not list(current_path.parent.glob("*.tmp"))


def test_stale_missing_cache_artifact_is_rebuilt_under_same_key(tmp_path):
    first = _build(tmp_path)
    (first.cache_dir / "sparse.json").unlink()

    rebuilt = _build(tmp_path)

    assert rebuilt.cache_key == first.cache_key
    assert (rebuilt.cache_dir / "sparse.json").exists()
    assert list(rebuilt.cache_dir.parent.glob(".*.invalid.*"))


def test_embedding_era_cache_artifacts_are_not_reused(tmp_path):
    first = _build(tmp_path)
    (first.cache_dir / "embeddings.npy").write_bytes(b"old embedding matrix")

    rebuilt = _build(tmp_path)

    assert rebuilt.cache_key == first.cache_key
    assert not (rebuilt.cache_dir / "embeddings.npy").exists()
    assert {path.name for path in rebuilt.cache_dir.iterdir()} == {
        "manifest.json",
        "objects.jsonl",
        "chunks.jsonl",
        "sparse.json",
    }
    assert list(rebuilt.cache_dir.parent.glob(".*.invalid.*"))


def test_pre_summary_key_cache_artifacts_are_not_reused(tmp_path):
    first = _build(tmp_path)
    manifest_path = first.cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("curated_summary_registry_hash")
    manifest.pop("curated_summary_registry_version")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rebuilt = _build(tmp_path)

    assert rebuilt.cache_key == first.cache_key
    assert rebuilt.manifest["curated_summary_registry_hash"]
    assert rebuilt.manifest["curated_summary_registry_version"] == "large-schema-summaries-v1"
    assert list(rebuilt.cache_dir.parent.glob(".*.invalid.*"))


def test_build_lock_waits_bounded_time_when_no_current_cache_exists(tmp_path):
    lock_path = tmp_path / "DB" / "build.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text('{"token": "other"}\n', encoding="utf-8")

    with pytest.raises(RetrievalIndexLockTimeout):
        _build(tmp_path)


def test_build_lock_reloads_existing_cache_instead_of_racing(tmp_path):
    first = _build(tmp_path)
    lock_path = tmp_path / "DB" / "build.lock"
    lock_path.write_text('{"token": "other"}\n', encoding="utf-8")

    loaded = _build(tmp_path)

    assert loaded.cache_key == first.cache_key


def test_missing_current_pointer_is_reported(tmp_path):
    with pytest.raises(RetrievalIndexError, match="missing current retrieval index pointer"):
        load_current_retrieval_index("DB", cache_root=tmp_path)


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
