"""Tests for versioned schema metadata context caching."""

from __future__ import annotations

from pathlib import Path

import pytest

from sol01.infra.fs_cache import stable_hash, write_json
from sol01.models import (
    ColumnSchema,
    SchemaProfileCatalog,
    SchemaProfileManifest,
    TableProfile,
    TableSchema,
)
from sol01.schema.schema_context_cache import (
    SCHEMA_CONTEXT_CACHE_KEY_LENGTH,
    SchemaContextCacheError,
    SchemaContextCacheLockTimeout,
    _version_dir,
    build_schema_context_cache,
    load_current_schema_context_cache,
    schema_context_cache_key,
    schema_source_hash,
)
from sol01.schema.schema_profiles import (
    SCHEMA_PROFILE_BUILDER_VERSION,
    SCHEMA_PROFILE_SUMMARIZER_VERSION,
    SCHEMA_PROFILE_TEMPLATE_VERSION,
    compact_table_keys_for_profiles,
    schema_profile_catalog_path,
    schema_profile_manifest_path,
)


def _db_index(*, extra_column: bool = False) -> dict[str, TableSchema]:
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
            sample_rows=[{"STATUS": "open"}, {"STATUS": "closed"}],
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


def _write_schema_profile(
    profile_root: Path,
    *,
    db: str = "DB",
    source_schema_hash: str = "schema-hash-v1",
    summary: str = "Order lifecycle fact table.",
) -> None:
    schema_profile_catalog_path(db, profile_root=profile_root).parent.mkdir(
        parents=True, exist_ok=True
    )
    catalog = SchemaProfileCatalog(
        db=db,
        source_schema_hash=source_schema_hash,
        table_profiles=[
            TableProfile(
                profile_id="orders_profile",
                abstraction_kind="wide_table",
                table_name="DB.PUBLIC.ORDERS",
                covered_tables=["DB.PUBLIC.ORDERS"],
                grain_hint="One row per order.",
                key_columns=["ORDER_ID", "CUSTOMER_ID"],
                dimension_columns=["STATUS"],
                naming_rules=["Use exact table name DB.PUBLIC.ORDERS."],
                compact_semantic_summary=summary,
                aliases=["orders"],
                theme_terms=["orders", "status"],
                confidence=0.9,
                provenance_inputs=["raw/orders.json"],
                source_column_count=3,
                source_sample_row_count=2,
            )
        ],
    )
    artifact_hash = stable_hash(catalog.model_dump(mode="json"))
    manifest = SchemaProfileManifest(
        db=db,
        source_schema_hash=source_schema_hash,
        builder_version=SCHEMA_PROFILE_BUILDER_VERSION,
        summarizer_version=SCHEMA_PROFILE_SUMMARIZER_VERSION,
        prompt_template_version=SCHEMA_PROFILE_TEMPLATE_VERSION,
        generated_at="2026-05-15T00:00:00+00:00",
        artifact_hash=artifact_hash,
        table_profile_count=1,
        family_profile_count=0,
    )
    write_json(
        schema_profile_catalog_path(db, profile_root=profile_root),
        catalog.model_dump(mode="json"),
    )
    write_json(
        schema_profile_manifest_path(db, profile_root=profile_root),
        manifest.model_dump(mode="json"),
    )


def test_builds_loads_and_validates_schema_context_cache_for_one_database_without_profiles(
    tmp_path,
):
    cache = _build(tmp_path)

    assert cache.cache_dir.exists()
    assert {path.name for path in cache.cache_dir.iterdir()} == {"manifest.json", "objects.jsonl"}

    loaded = load_current_schema_context_cache("DB", cache_root=tmp_path)

    assert loaded.cache_key == cache.cache_key
    assert loaded.manifest["source_schema_hash"] == schema_source_hash(_db_index())
    assert loaded.manifest["schema_profile_catalog_hash"] is None
    assert loaded.manifest["schema_profile_builder_version"] == SCHEMA_PROFILE_BUILDER_VERSION
    assert loaded.manifest["context_mode"] == "full_metadata"


def test_cache_key_changes_for_schema_versions_family_threshold_and_profile_hash():
    source_hash = schema_source_hash(_db_index())
    base = {
        "db": "DB",
        "source_schema_hash": source_hash,
        "family_similarity_threshold": 0.82,
        "schema_profile_catalog_hash": "profile-hash-v1",
    }

    baseline = schema_context_cache_key(**base)
    assert len(baseline) == SCHEMA_CONTEXT_CACHE_KEY_LENGTH
    assert schema_context_cache_key(**{**base, "source_schema_hash": "different"}) != baseline
    assert schema_context_cache_key(**{**base, "family_similarity_threshold": 0.9}) != baseline
    assert (
        schema_context_cache_key(**{**base, "schema_profile_catalog_hash": "profile-hash-v2"})
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


def test_changed_profile_artifact_content_publishes_separate_version_directory(tmp_path):
    profile_root = tmp_path / "schema_profiles"
    cache_root = tmp_path / "cache"
    _write_schema_profile(profile_root, summary="Order lifecycle fact table.")

    first = _build(cache_root, schema_profile_root=profile_root)
    _write_schema_profile(
        profile_root,
        summary="Order lifecycle table with explicit status semantics.",
    )
    second = _build(cache_root, schema_profile_root=profile_root)

    assert first.cache_key != second.cache_key
    assert (
        first.manifest["schema_profile_catalog_hash"]
        != second.manifest["schema_profile_catalog_hash"]
    )
    assert _version_dir(cache_root, "DB", first.cache_key).exists()
    assert _version_dir(cache_root, "DB", second.cache_key).exists()


def test_profile_artifact_drives_adaptive_compaction_and_object_metadata(tmp_path):
    profile_root = tmp_path / "schema_profiles"
    _write_schema_profile(profile_root)
    cache = _build(tmp_path / "cache", schema_profile_root=profile_root)

    object_ids = {obj.object_id for obj in cache.objects}
    orders_object = next(obj for obj in cache.objects if obj.object_id == "table:DB.PUBLIC.ORDERS")
    customers_object = next(
        obj for obj in cache.objects if obj.object_id == "table:DB.PUBLIC.CUSTOMERS"
    )

    assert cache.manifest["context_mode"] == "adaptive_profiles"
    assert "column:DB.PUBLIC.ORDERS#STATUS" not in object_ids
    assert "column:DB.PUBLIC.CUSTOMERS#CUSTOMER_ID" in object_ids
    assert orders_object.metadata["schema_profile_ids"] == ["orders_profile"]
    assert orders_object.metadata["schema_profiles"][0]["text"].startswith(
        "Schema profile: orders_profile."
    )
    assert "schema_profiles" not in customers_object.metadata


def test_adaptive_profiles_keep_unprofiled_tables_visible_and_exact(tmp_path):
    profile_root = tmp_path / "schema_profiles"
    catalog = SchemaProfileCatalog(
        db="GITHUB_REPOS_DATE",
        source_schema_hash="schema-hash-v1",
        table_profiles=[],
        family_profiles=[],
    )
    manifest = SchemaProfileManifest(
        db="GITHUB_REPOS_DATE",
        source_schema_hash="schema-hash-v1",
        builder_version=SCHEMA_PROFILE_BUILDER_VERSION,
        summarizer_version=SCHEMA_PROFILE_SUMMARIZER_VERSION,
        prompt_template_version=SCHEMA_PROFILE_TEMPLATE_VERSION,
        generated_at="2026-05-15T00:00:00+00:00",
        artifact_hash=stable_hash(catalog.model_dump(mode="json")),
        table_profile_count=0,
        family_profile_count=0,
    )
    schema_profile_catalog_path("GITHUB_REPOS_DATE", profile_root=profile_root).parent.mkdir(
        parents=True, exist_ok=True
    )
    write_json(
        schema_profile_catalog_path("GITHUB_REPOS_DATE", profile_root=profile_root),
        catalog.model_dump(mode="json"),
    )
    write_json(
        schema_profile_manifest_path("GITHUB_REPOS_DATE", profile_root=profile_root),
        manifest.model_dump(mode="json"),
    )
    db_index = {
        "GITHUB_REPOS_DATE.DAY._20240103": TableSchema(
            name="_20240103",
            database_name="GITHUB_REPOS_DATE",
            schema_name="DAY",
            full_name="GITHUB_REPOS_DATE.DAY._20240103",
            ddl="",
            columns=[ColumnSchema(name="PUBLIC", type="BOOLEAN")],
            sample_rows=[],
            searchable_text="github events",
        ),
        "GITHUB_REPOS_DATE.DAY.REPOSITORIES": TableSchema(
            name="REPOSITORIES",
            database_name="GITHUB_REPOS_DATE",
            schema_name="DAY",
            full_name="GITHUB_REPOS_DATE.DAY.REPOSITORIES",
            ddl="",
            columns=[ColumnSchema(name="ID", type="TEXT"), ColumnSchema(name="NAME", type="TEXT")],
            sample_rows=[],
            searchable_text="repositories",
        ),
    }

    cache = build_schema_context_cache(
        "GITHUB_REPOS_DATE",
        db_index=db_index,
        cache_root=tmp_path,
        schema_profile_root=profile_root,
        lock_timeout_seconds=0.1,
    )

    object_ids = {obj.object_id for obj in cache.objects}
    assert cache.manifest["context_mode"] == "full_metadata"
    assert "table:GITHUB_REPOS_DATE.DAY._20240103" in object_ids
    assert "column:GITHUB_REPOS_DATE.DAY._20240103#PUBLIC" in object_ids
    assert "column:GITHUB_REPOS_DATE.DAY.REPOSITORIES#NAME" in object_ids


def test_compact_table_keys_come_from_profile_catalog():
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
                naming_rules=["Use exact table name DB.PUBLIC.ORDERS."],
                compact_semantic_summary="Order fact table.",
                aliases=["orders"],
                theme_terms=["orders"],
                confidence=0.9,
                provenance_inputs=["raw/orders.json"],
                source_column_count=3,
                source_sample_row_count=0,
            )
        ],
    )

    assert compact_table_keys_for_profiles(catalog) == {"DB.PUBLIC.ORDERS"}
