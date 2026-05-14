"""Build and publish reusable schema metadata contexts."""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sol01.infra.config import SchemaContextConfig
from sol01.infra.fs_cache import (
    acquire_build_lock_or_wait,
    atomic_write_json,
    publish_version_directory,
    quarantine_invalid_directory,
    read_json,
    read_jsonl,
    release_build_lock,
    safe_path_segment,
    stable_hash,
    write_json,
    write_jsonl,
)
from sol01.infra.logging import get_logger
from sol01.infra.paths import REPO_ROOT
from sol01.models import SchemaContextChunk, SchemaObject, TableSchema
from sol01.schema.chunks import render_schema_chunks
from sol01.schema.db_index import load_db_index
from sol01.schema.large_schema_summaries import (
    DEFAULT_LARGE_SCHEMA_SUMMARY_PATH,
    LARGE_SCHEMA_SUMMARY_REGISTRY_VERSION,
    LargeSchemaSummaryRegistry,
    large_schema_summary_registry_hash,
    load_large_schema_summary_registry,
)
from sol01.schema.objects import build_schema_objects

OBJECT_BUILDER_VERSION = "schema-objects-v3"
CHUNK_RENDER_VERSION = "schema-chunks-v2"
MANIFEST_VERSION = 2
REQUIRED_CACHE_ARTIFACTS = frozenset({"objects.jsonl", "chunks.jsonl", "manifest.json"})
REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "manifest_version",
        "db",
        "cache_key",
        "source_schema_hash",
        "object_builder_version",
        "chunk_render_version",
        "curated_summary_registry_hash",
        "curated_summary_registry_version",
        "family_similarity_threshold",
        "object_count",
        "chunk_count",
    }
)
DEFAULT_SCHEMA_CONTEXT_CACHE_ROOT = (
    REPO_ROOT / "methods" / "sol01" / ".cache" / "schema_context_cache"
).resolve()
DEFAULT_LOCK_TIMEOUT_SECONDS = 60.0
DEFAULT_LOCK_POLL_SECONDS = 0.1
logger = get_logger(__name__)


class SchemaContextCacheError(RuntimeError):
    """Raised when a schema context cache cannot be built or loaded."""


class SchemaContextCacheLockTimeout(SchemaContextCacheError):
    """Raised when another worker holds the build lock for too long."""


@dataclass(frozen=True)
class SchemaContextCache:
    """A loaded schema metadata context and its on-disk artifacts."""

    db: str
    cache_key: str
    cache_dir: Path
    manifest: dict[str, Any]
    objects: list[SchemaObject]
    chunks: list[SchemaContextChunk]


def build_schema_context_cache(
    db: str,
    *,
    db_index: Mapping[str, TableSchema] | None = None,
    config: SchemaContextConfig | None = None,
    cache_root: Path = DEFAULT_SCHEMA_CONTEXT_CACHE_ROOT,
    curated_summary_registry_path: Path = DEFAULT_LARGE_SCHEMA_SUMMARY_PATH,
    lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    lock_poll_seconds: float = DEFAULT_LOCK_POLL_SECONDS,
) -> SchemaContextCache:
    """Build or load the versioned schema metadata context for one database."""

    started_at = time.perf_counter()
    config = config or SchemaContextConfig()
    db_index = dict(db_index) if db_index is not None else _load_db_index(db)
    source_hash = schema_source_hash(db_index)
    curated_summary_registry_hash = large_schema_summary_registry_hash(
        curated_summary_registry_path
    )
    cache_key = schema_context_cache_key(
        db=db,
        source_schema_hash=source_hash,
        family_similarity_threshold=config.family_similarity_threshold,
        curated_summary_registry_hash=curated_summary_registry_hash,
    )
    version_dir = _version_dir(cache_root, db, cache_key)
    current_path = _current_pointer_path(cache_root, db)
    lock_path = _build_lock_path(cache_root, db)
    logger.info(
        "schema context cache start",
        db=db,
        table_count=len(db_index),
        cache_key=cache_key,
    )

    existing = _load_valid_cache(
        db=db,
        cache_key=cache_key,
        cache_dir=version_dir,
        expected_source_hash=source_hash,
    )
    if existing is not None:
        _write_current_pointer(current_path, db=db, cache_key=cache_key, cache_dir=version_dir)
        logger.info(
            "schema context cache hit",
            db=db,
            cache_key=cache_key,
            cache_dir=str(version_dir),
            elapsed_seconds=round(time.perf_counter() - started_at, 3),
        )
        return existing

    logger.info("schema context cache lock wait", db=db, cache_key=cache_key)

    def _is_done() -> bool:
        return (
            _load_valid_cache(
                db=db,
                cache_key=cache_key,
                cache_dir=version_dir,
                expected_source_hash=source_hash,
            )
            is not None
            or _load_current_if_matching(
                db=db,
                current_path=current_path,
                cache_key=cache_key,
                expected_source_hash=source_hash,
            )
            is not None
        )

    lock_token = acquire_build_lock_or_wait(
        lock_path,
        is_done=_is_done,
        timeout_seconds=lock_timeout_seconds,
        poll_seconds=lock_poll_seconds,
    )
    if lock_token is None:
        loaded = _load_valid_cache(
            db=db,
            cache_key=cache_key,
            cache_dir=version_dir,
            expected_source_hash=source_hash,
        )
        if loaded is not None:
            logger.info(
                "schema context cache loaded after wait",
                db=db,
                cache_key=cache_key,
                cache_dir=str(version_dir),
                elapsed_seconds=round(time.perf_counter() - started_at, 3),
            )
            return loaded
        loaded = _load_current_if_matching(
            db=db,
            current_path=current_path,
            cache_key=cache_key,
            expected_source_hash=source_hash,
        )
        if loaded is not None:
            logger.info(
                "schema context cache current loaded after wait",
                db=db,
                cache_key=cache_key,
                cache_dir=str(loaded.cache_dir),
                elapsed_seconds=round(time.perf_counter() - started_at, 3),
            )
            return loaded
        raise SchemaContextCacheLockTimeout(
            f"timed out waiting for schema context cache build lock for {db}"
        )

    try:
        logger.info("schema context cache build start", db=db, cache_key=cache_key)
        existing = _load_valid_cache(
            db=db,
            cache_key=cache_key,
            cache_dir=version_dir,
            expected_source_hash=source_hash,
        )
        if existing is not None:
            _write_current_pointer(current_path, db=db, cache_key=cache_key, cache_dir=version_dir)
            logger.info(
                "schema context cache hit after lock",
                db=db,
                cache_key=cache_key,
                cache_dir=str(version_dir),
                elapsed_seconds=round(time.perf_counter() - started_at, 3),
            )
            return existing

        if version_dir.exists():
            quarantine_invalid_directory(version_dir)

        registry = load_large_schema_summary_registry(curated_summary_registry_path)
        covered_table_keys = _covered_table_keys(db_index, registry)
        objects = build_schema_objects(
            db_index,
            family_similarity_threshold=config.family_similarity_threshold,
            covered_table_keys=covered_table_keys,
        )
        chunks = render_schema_chunks(objects, large_schema_summary_registry=registry)
        logger.info(
            "schema context cache objects rendered",
            db=db,
            cache_key=cache_key,
            object_count=len(objects),
            chunk_count=len(chunks),
            covered_table_count=len(covered_table_keys),
        )

        temp_dir = _new_temp_version_dir(cache_root, db, cache_key)
        try:
            _write_cache_artifacts(
                temp_dir,
                db=db,
                cache_key=cache_key,
                source_hash=source_hash,
                family_similarity_threshold=config.family_similarity_threshold,
                curated_summary_registry_hash=curated_summary_registry_hash,
                objects=objects,
                chunks=chunks,
            )
            loaded_temp = _load_valid_cache(
                db=db,
                cache_key=cache_key,
                cache_dir=temp_dir,
                expected_source_hash=source_hash,
            )
            if loaded_temp is None:
                raise SchemaContextCacheError(
                    "new schema context cache failed validation before publish"
                )
            published = publish_version_directory(temp_dir, version_dir)
            if not published:
                loaded = _load_valid_cache(
                    db=db,
                    cache_key=cache_key,
                    cache_dir=version_dir,
                    expected_source_hash=source_hash,
                )
                if loaded is None:
                    raise SchemaContextCacheError(
                        f"existing schema context cache directory is invalid: {version_dir}"
                    )
                _write_current_pointer(
                    current_path,
                    db=db,
                    cache_key=cache_key,
                    cache_dir=version_dir,
                )
                logger.info(
                    "schema context cache loaded existing published version",
                    db=db,
                    cache_key=cache_key,
                    cache_dir=str(version_dir),
                    elapsed_seconds=round(time.perf_counter() - started_at, 3),
                )
                return loaded
        except BaseException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        _write_current_pointer(current_path, db=db, cache_key=cache_key, cache_dir=version_dir)
        loaded = _load_valid_cache(
            db=db,
            cache_key=cache_key,
            cache_dir=version_dir,
            expected_source_hash=source_hash,
        )
        if loaded is None:
            raise SchemaContextCacheError("published schema context cache failed validation")
        logger.info(
            "schema context cache published",
            db=db,
            cache_key=cache_key,
            cache_dir=str(version_dir),
            object_count=len(objects),
            chunk_count=len(chunks),
            elapsed_seconds=round(time.perf_counter() - started_at, 3),
        )
        return loaded
    finally:
        release_build_lock(lock_path, lock_token)


def load_current_schema_context_cache(
    db: str,
    *,
    cache_root: Path = DEFAULT_SCHEMA_CONTEXT_CACHE_ROOT,
) -> SchemaContextCache:
    """Load the current schema context cache pointer for one database."""

    pointer = _read_current_pointer(_current_pointer_path(cache_root, db))
    cache_key = str(pointer.get("cache_key") or "")
    cache_dir = Path(str(pointer.get("cache_dir") or _version_dir(cache_root, db, cache_key)))
    if not cache_key:
        raise SchemaContextCacheError(
            f"current schema context cache pointer for {db} has no cache_key"
        )
    cache = _load_valid_cache(db=db, cache_key=cache_key, cache_dir=cache_dir)
    if cache is None:
        raise SchemaContextCacheError(
            f"current schema context cache for {db} is missing or invalid"
        )
    return cache


def prewarm_schema_context_caches(
    dbs: Iterable[str],
    *,
    config: SchemaContextConfig | None = None,
    cache_root: Path = DEFAULT_SCHEMA_CONTEXT_CACHE_ROOT,
) -> list[SchemaContextCache]:
    """Build schema metadata contexts for unique databases before worker threads start."""

    unique_dbs = sorted({db.strip() for db in dbs if db.strip()})
    logger.info("schema context prewarm start", database_count=len(unique_dbs), dbs=unique_dbs)
    started_at = time.perf_counter()
    caches = [
        build_schema_context_cache(
            db,
            config=config,
            cache_root=cache_root,
        )
        for db in unique_dbs
    ]
    logger.info(
        "schema context prewarm complete",
        database_count=len(caches),
        elapsed_seconds=round(time.perf_counter() - started_at, 3),
    )
    return caches


def schema_source_hash(db_index: Mapping[str, TableSchema]) -> str:
    """Hash the canonical table schema payload used to build schema objects."""

    payload = {
        table_name: db_index[table_name].model_dump(mode="json") for table_name in sorted(db_index)
    }
    return stable_hash(payload)


def schema_context_cache_key(
    *,
    db: str,
    source_schema_hash: str,
    family_similarity_threshold: float,
    curated_summary_registry_hash: str,
) -> str:
    """Return a deterministic cache key for all inputs that affect schema context artifacts."""

    return stable_hash(
        {
            "cache_schema": MANIFEST_VERSION,
            "chunk_render_version": CHUNK_RENDER_VERSION,
            "curated_summary_registry_hash": curated_summary_registry_hash,
            "curated_summary_registry_version": LARGE_SCHEMA_SUMMARY_REGISTRY_VERSION,
            "db": db,
            "family_similarity_threshold": family_similarity_threshold,
            "object_builder_version": OBJECT_BUILDER_VERSION,
            "source_schema_hash": source_schema_hash,
        }
    )


def _write_cache_artifacts(
    cache_dir: Path,
    *,
    db: str,
    cache_key: str,
    source_hash: str,
    family_similarity_threshold: float,
    curated_summary_registry_hash: str,
    objects: Sequence[SchemaObject],
    chunks: Sequence[SchemaContextChunk],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "db": db,
        "cache_key": cache_key,
        "source_schema_hash": source_hash,
        "object_builder_version": OBJECT_BUILDER_VERSION,
        "chunk_render_version": CHUNK_RENDER_VERSION,
        "curated_summary_registry_hash": curated_summary_registry_hash,
        "curated_summary_registry_version": LARGE_SCHEMA_SUMMARY_REGISTRY_VERSION,
        "family_similarity_threshold": family_similarity_threshold,
        "object_count": len(objects),
        "chunk_count": len(chunks),
    }

    write_jsonl(cache_dir / "objects.jsonl", [obj.model_dump(mode="json") for obj in objects])
    write_jsonl(cache_dir / "chunks.jsonl", [chunk.model_dump(mode="json") for chunk in chunks])
    write_json(cache_dir / "manifest.json", manifest)
    logger.info(
        "schema context artifacts written",
        db=db,
        cache_key=cache_key,
        cache_dir=str(cache_dir),
        object_count=len(objects),
        chunk_count=len(chunks),
    )


def _load_valid_cache(
    *,
    db: str,
    cache_key: str,
    cache_dir: Path,
    expected_source_hash: str | None = None,
) -> SchemaContextCache | None:
    try:
        if _artifact_names(cache_dir) != REQUIRED_CACHE_ARTIFACTS:
            return None
        manifest = read_json(cache_dir / "manifest.json")
        if not REQUIRED_MANIFEST_FIELDS.issubset(manifest):
            return None
        if manifest.get("manifest_version") != MANIFEST_VERSION:
            return None
        if manifest.get("db") != db or manifest.get("cache_key") != cache_key:
            return None
        if (
            expected_source_hash is not None
            and manifest.get("source_schema_hash") != expected_source_hash
        ):
            return None
        objects = [
            SchemaObject.model_validate(row) for row in read_jsonl(cache_dir / "objects.jsonl")
        ]
        chunks = [
            SchemaContextChunk.model_validate(row) for row in read_jsonl(cache_dir / "chunks.jsonl")
        ]
    except (FileNotFoundError, ImportError, OSError, ValueError):
        return None

    if manifest.get("object_count") != len(objects) or manifest.get("chunk_count") != len(chunks):
        return None
    object_ids = {obj.object_id for obj in objects}
    if any(chunk.object_id not in object_ids for chunk in chunks):
        return None

    return SchemaContextCache(
        db=db,
        cache_key=cache_key,
        cache_dir=cache_dir,
        manifest=manifest,
        objects=objects,
        chunks=chunks,
    )


def _load_current_if_matching(
    *,
    db: str,
    current_path: Path,
    cache_key: str,
    expected_source_hash: str,
) -> SchemaContextCache | None:
    try:
        pointer = _read_current_pointer(current_path)
    except SchemaContextCacheError:
        return None
    if pointer.get("cache_key") != cache_key:
        return None
    cache_dir = Path(str(pointer.get("cache_dir") or ""))
    if not cache_dir:
        return None
    return _load_valid_cache(
        db=db,
        cache_key=cache_key,
        cache_dir=cache_dir,
        expected_source_hash=expected_source_hash,
    )


def _write_current_pointer(current_path: Path, *, db: str, cache_key: str, cache_dir: Path) -> None:
    atomic_write_json(
        current_path,
        {"db": db, "cache_key": cache_key, "cache_dir": str(cache_dir)},
    )


def _read_current_pointer(current_path: Path) -> dict[str, Any]:
    try:
        return read_json(current_path)
    except FileNotFoundError as exc:
        raise SchemaContextCacheError(
            f"missing current schema context cache pointer: {current_path}"
        ) from exc
    except (ValueError, OSError) as exc:
        raise SchemaContextCacheError(
            f"invalid current schema context cache pointer: {current_path}"
        ) from exc


def _artifact_names(cache_dir: Path) -> frozenset[str]:
    return frozenset(path.name for path in cache_dir.iterdir())


def _covered_table_keys(
    db_index: Mapping[str, TableSchema],
    registry: LargeSchemaSummaryRegistry,
) -> set[str]:
    """Return db-index keys whose tables match at least one curated summary."""

    covered: set[str] = set()
    for table_key, table in db_index.items():
        database = table.database_name or ""
        schema_name = table.schema_name or ""
        table_name = table.name or table_key
        if registry.match_table(database=database, schema_name=schema_name, table_name=table_name):
            covered.add(table_key)
    return covered


def _load_db_index(db: str) -> dict[str, TableSchema]:
    return load_db_index(db)


def _version_dir(cache_root: Path, db: str, cache_key: str) -> Path:
    return cache_root / safe_path_segment(db) / "versions" / cache_key


def _current_pointer_path(cache_root: Path, db: str) -> Path:
    return cache_root / safe_path_segment(db) / "current.json"


def _build_lock_path(cache_root: Path, db: str) -> Path:
    return cache_root / safe_path_segment(db) / "build.lock"


def _new_temp_version_dir(cache_root: Path, db: str, cache_key: str) -> Path:
    versions_dir = cache_root / safe_path_segment(db) / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{cache_key}.",
            suffix=".tmp",
            dir=versions_dir,
        )
    )
