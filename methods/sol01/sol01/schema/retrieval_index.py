"""Build and publish reusable schema retrieval indexes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from sol01.infra.config import SchemaRetrievalConfig
from sol01.infra.paths import REPO_ROOT
from sol01.models import RetrievalChunk, SchemaObject, TableSchema
from sol01.schema.chunks import render_schema_chunks
from sol01.schema.embedding import EmbeddingProvider, SchemaProviderError
from sol01.schema.objects import build_schema_objects
from sol01.schema.ollama_provider import OllamaEmbeddingProvider
from sol01.schema.retrieval import load_db_index

OBJECT_BUILDER_VERSION = "schema-objects-v1"
CHUNK_RENDER_VERSION = "schema-chunks-v1"
SPARSE_INDEX_VERSION = "bm25-v1"
MANIFEST_VERSION = 1
DEFAULT_RETRIEVAL_INDEX_CACHE_ROOT = (
    REPO_ROOT / "methods" / "sol01" / ".cache" / "schema_retrieval_index"
).resolve()
DEFAULT_LOCK_TIMEOUT_SECONDS = 60.0
DEFAULT_LOCK_POLL_SECONDS = 0.1
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class RetrievalIndexError(RuntimeError):
    """Raised when a retrieval index cannot be built or loaded."""


class RetrievalIndexLockTimeout(RetrievalIndexError):
    """Raised when another worker holds the build lock for too long."""


@dataclass(frozen=True)
class SchemaRetrievalIndex:
    """A loaded schema retrieval index and its on-disk artifacts."""

    db: str
    cache_key: str
    cache_dir: Path
    manifest: dict[str, Any]
    objects: list[SchemaObject]
    chunks: list[RetrievalChunk]
    sparse: dict[str, Any]
    embeddings: np.ndarray


def build_retrieval_index(
    db: str,
    *,
    db_index: Mapping[str, TableSchema] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    config: SchemaRetrievalConfig | None = None,
    cache_root: Path = DEFAULT_RETRIEVAL_INDEX_CACHE_ROOT,
    model_metadata: Mapping[str, object] | None = None,
    object_builder_version: str = OBJECT_BUILDER_VERSION,
    chunk_render_version: str = CHUNK_RENDER_VERSION,
    lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    lock_poll_seconds: float = DEFAULT_LOCK_POLL_SECONDS,
) -> SchemaRetrievalIndex:
    """Build or load the versioned retrieval index for one database."""

    config = config or SchemaRetrievalConfig()
    db_index = dict(db_index) if db_index is not None else _load_db_index(db)
    source_hash = schema_source_hash(db_index)
    objects = build_schema_objects(
        db_index,
        family_similarity_threshold=config.family_similarity_threshold,
    )
    chunks = render_schema_chunks(objects)
    provider = embedding_provider or _default_embedding_provider(config)
    model_fingerprint = _embedding_model_fingerprint(
        provider,
        configured_model=config.embedding_model,
        explicit_metadata=model_metadata,
    )
    cache_key = retrieval_index_cache_key(
        db=db,
        source_schema_hash=source_hash,
        object_builder_version=object_builder_version,
        chunk_render_version=chunk_render_version,
        embedding_model=config.embedding_model,
        embedding_model_metadata=model_fingerprint,
        family_similarity_threshold=config.family_similarity_threshold,
    )
    version_dir = _version_dir(cache_root, db, cache_key)
    current_path = _current_pointer_path(cache_root, db)
    lock_path = _build_lock_path(cache_root, db)

    existing = _load_valid_index(
        db=db,
        cache_key=cache_key,
        cache_dir=version_dir,
        expected_source_hash=source_hash,
    )
    if existing is not None:
        _write_current_pointer(current_path, db=db, cache_key=cache_key, cache_dir=version_dir)
        return existing

    lock_token = _acquire_build_lock_or_wait(
        lock_path,
        db=db,
        cache_key=cache_key,
        current_path=current_path,
        version_dir=version_dir,
        expected_source_hash=source_hash,
        timeout_seconds=lock_timeout_seconds,
        poll_seconds=lock_poll_seconds,
    )
    if lock_token is None:
        loaded = _load_valid_index(
            db=db,
            cache_key=cache_key,
            cache_dir=version_dir,
            expected_source_hash=source_hash,
        )
        if loaded is not None:
            return loaded
        loaded = _load_current_if_matching(
            db=db,
            current_path=current_path,
            cache_key=cache_key,
            expected_source_hash=source_hash,
        )
        if loaded is not None:
            return loaded
        raise RetrievalIndexLockTimeout(
            f"timed out waiting for retrieval index build lock for {db}"
        )

    try:
        existing = _load_valid_index(
            db=db,
            cache_key=cache_key,
            cache_dir=version_dir,
            expected_source_hash=source_hash,
        )
        if existing is not None:
            _write_current_pointer(current_path, db=db, cache_key=cache_key, cache_dir=version_dir)
            return existing

        if version_dir.exists():
            _quarantine_invalid_version_directory(version_dir)

        temp_dir = _new_temp_version_dir(cache_root, db, cache_key)
        try:
            _write_index_artifacts(
                temp_dir,
                db=db,
                cache_key=cache_key,
                source_hash=source_hash,
                object_builder_version=object_builder_version,
                chunk_render_version=chunk_render_version,
                embedding_model=config.embedding_model,
                embedding_model_metadata=model_fingerprint,
                family_similarity_threshold=config.family_similarity_threshold,
                objects=objects,
                chunks=chunks,
                embedding_provider=provider,
            )
            loaded_temp = _load_valid_index(
                db=db,
                cache_key=cache_key,
                cache_dir=temp_dir,
                expected_source_hash=source_hash,
            )
            if loaded_temp is None:
                raise RetrievalIndexError("new retrieval index failed validation before publish")
            published = _publish_version_directory(temp_dir, version_dir)
            if not published:
                loaded = _load_valid_index(
                    db=db,
                    cache_key=cache_key,
                    cache_dir=version_dir,
                    expected_source_hash=source_hash,
                )
                if loaded is None:
                    raise RetrievalIndexError(
                        f"existing retrieval index directory is invalid: {version_dir}"
                    )
                _write_current_pointer(
                    current_path,
                    db=db,
                    cache_key=cache_key,
                    cache_dir=version_dir,
                )
                return loaded
        except BaseException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        _write_current_pointer(current_path, db=db, cache_key=cache_key, cache_dir=version_dir)
        loaded = _load_valid_index(
            db=db,
            cache_key=cache_key,
            cache_dir=version_dir,
            expected_source_hash=source_hash,
        )
        if loaded is None:
            raise RetrievalIndexError("published retrieval index failed validation")
        return loaded
    finally:
        _release_build_lock(lock_path, lock_token)


def load_current_retrieval_index(
    db: str,
    *,
    cache_root: Path = DEFAULT_RETRIEVAL_INDEX_CACHE_ROOT,
) -> SchemaRetrievalIndex:
    """Load the current retrieval index pointer for one database."""

    pointer = _read_current_pointer(_current_pointer_path(cache_root, db))
    cache_key = str(pointer.get("cache_key") or "")
    cache_dir = Path(str(pointer.get("cache_dir") or _version_dir(cache_root, db, cache_key)))
    if not cache_key:
        raise RetrievalIndexError(f"current retrieval index pointer for {db} has no cache_key")
    index = _load_valid_index(db=db, cache_key=cache_key, cache_dir=cache_dir)
    if index is None:
        raise RetrievalIndexError(f"current retrieval index for {db} is missing or invalid")
    return index


def prewarm_retrieval_indexes(
    dbs: Iterable[str],
    *,
    embedding_provider: EmbeddingProvider | None = None,
    config: SchemaRetrievalConfig | None = None,
    cache_root: Path = DEFAULT_RETRIEVAL_INDEX_CACHE_ROOT,
) -> list[SchemaRetrievalIndex]:
    """Build retrieval indexes for unique databases before worker threads start."""

    unique_dbs = sorted({db.strip() for db in dbs if db.strip()})
    return [
        build_retrieval_index(
            db,
            embedding_provider=embedding_provider,
            config=config,
            cache_root=cache_root,
        )
        for db in unique_dbs
    ]


def schema_source_hash(db_index: Mapping[str, TableSchema]) -> str:
    """Hash the canonical table schema payload used to build retrieval objects."""

    payload = {
        table_name: db_index[table_name].model_dump(mode="json") for table_name in sorted(db_index)
    }
    return _stable_hash(payload)


def retrieval_index_cache_key(
    *,
    db: str,
    source_schema_hash: str,
    object_builder_version: str,
    chunk_render_version: str,
    embedding_model: str,
    embedding_model_metadata: Mapping[str, object],
    family_similarity_threshold: float,
) -> str:
    """Return a deterministic cache key for all inputs that affect retrieval artifacts."""

    return _stable_hash(
        {
            "cache_schema": MANIFEST_VERSION,
            "chunk_render_version": chunk_render_version,
            "db": db,
            "embedding_model": embedding_model,
            "embedding_model_metadata": dict(embedding_model_metadata),
            "family_similarity_threshold": family_similarity_threshold,
            "object_builder_version": object_builder_version,
            "source_schema_hash": source_schema_hash,
            "sparse_index_version": SPARSE_INDEX_VERSION,
        }
    )


def _write_index_artifacts(
    cache_dir: Path,
    *,
    db: str,
    cache_key: str,
    source_hash: str,
    object_builder_version: str,
    chunk_render_version: str,
    embedding_model: str,
    embedding_model_metadata: Mapping[str, object],
    family_similarity_threshold: float,
    objects: Sequence[SchemaObject],
    chunks: Sequence[RetrievalChunk],
    embedding_provider: EmbeddingProvider,
) -> None:
    """Write one complete version directory before it is published."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    sparse = _build_sparse_index(chunks)
    embeddings = _build_embedding_matrix(chunks, embedding_provider)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "db": db,
        "cache_key": cache_key,
        "source_schema_hash": source_hash,
        "object_builder_version": object_builder_version,
        "chunk_render_version": chunk_render_version,
        "sparse_index_version": SPARSE_INDEX_VERSION,
        "embedding_model": embedding_model,
        "embedding_model_metadata": dict(embedding_model_metadata),
        "family_similarity_threshold": family_similarity_threshold,
        "object_count": len(objects),
        "chunk_count": len(chunks),
        "embedding_shape": list(embeddings.shape),
    }

    _write_jsonl(cache_dir / "objects.jsonl", [obj.model_dump(mode="json") for obj in objects])
    _write_jsonl(cache_dir / "chunks.jsonl", [chunk.model_dump(mode="json") for chunk in chunks])
    _write_json(cache_dir / "sparse.json", sparse)
    np.save(cache_dir / "embeddings.npy", embeddings)
    _write_json(cache_dir / "manifest.json", manifest)


def _build_sparse_index(chunks: Sequence[RetrievalChunk]) -> dict[str, Any]:
    """Build a deterministic local BM25-style sparse index payload."""

    documents: list[dict[str, Any]] = []
    document_frequency: Counter[str] = Counter()
    lengths: list[int] = []

    for chunk in chunks:
        terms = _tokenize(chunk.bm25_text or chunk.text)
        counts = Counter(terms)
        lengths.append(len(terms))
        document_frequency.update(counts.keys())
        documents.append(
            {
                "chunk_id": chunk.chunk_id,
                "terms": dict(sorted(counts.items())),
            }
        )

    average_length = sum(lengths) / len(lengths) if lengths else 0.0
    return {
        "version": SPARSE_INDEX_VERSION,
        "algorithm": "bm25",
        "parameters": {"k1": 1.2, "b": 0.75},
        "chunk_ids": [chunk.chunk_id for chunk in chunks],
        "document_count": len(chunks),
        "document_lengths": lengths,
        "average_document_length": average_length,
        "document_frequency": dict(sorted(document_frequency.items())),
        "documents": documents,
    }


def _build_embedding_matrix(
    chunks: Sequence[RetrievalChunk],
    embedding_provider: EmbeddingProvider,
) -> np.ndarray:
    """Return one embedding row per chunk, with zero rows for sparse-only chunks."""

    dense_positions: list[int] = []
    dense_texts: list[str] = []
    for index, chunk in enumerate(chunks):
        if not chunk.include_dense_embedding or not chunk.embedding_text.strip():
            continue
        dense_positions.append(index)
        dense_texts.append(chunk.embedding_text)

    dense_vectors = embedding_provider.embed_texts(dense_texts) if dense_texts else []
    if len(dense_vectors) != len(dense_texts):
        raise SchemaProviderError(
            "embedding provider returned a different number of vectors than input texts"
        )
    if not dense_vectors:
        return np.zeros((len(chunks), 0), dtype=np.float32)

    dimensions = len(dense_vectors[0])
    if dimensions < 1:
        raise SchemaProviderError("embedding provider returned an empty vector")
    matrix = np.zeros((len(chunks), dimensions), dtype=np.float32)
    for position, vector in zip(dense_positions, dense_vectors, strict=True):
        if len(vector) != dimensions:
            raise SchemaProviderError("embedding provider returned inconsistent vector dimensions")
        matrix[position] = np.asarray(vector, dtype=np.float32)
    return matrix


def _load_valid_index(
    *,
    db: str,
    cache_key: str,
    cache_dir: Path,
    expected_source_hash: str | None = None,
) -> SchemaRetrievalIndex | None:
    """Load one version directory, returning None when validation fails."""

    try:
        manifest = _read_json(cache_dir / "manifest.json")
        if manifest.get("db") != db or manifest.get("cache_key") != cache_key:
            return None
        if (
            expected_source_hash is not None
            and manifest.get("source_schema_hash") != expected_source_hash
        ):
            return None
        objects = [
            SchemaObject.model_validate(row) for row in _read_jsonl(cache_dir / "objects.jsonl")
        ]
        chunks = [
            RetrievalChunk.model_validate(row) for row in _read_jsonl(cache_dir / "chunks.jsonl")
        ]
        sparse = _read_json(cache_dir / "sparse.json")
        embeddings = np.load(cache_dir / "embeddings.npy")
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return None

    if manifest.get("object_count") != len(objects) or manifest.get("chunk_count") != len(chunks):
        return None
    if embeddings.shape[0] != len(chunks):
        return None
    if sparse.get("chunk_ids") != [chunk.chunk_id for chunk in chunks]:
        return None
    object_ids = {obj.object_id for obj in objects}
    if any(chunk.object_id not in object_ids for chunk in chunks):
        return None

    return SchemaRetrievalIndex(
        db=db,
        cache_key=cache_key,
        cache_dir=cache_dir,
        manifest=manifest,
        objects=objects,
        chunks=chunks,
        sparse=sparse,
        embeddings=embeddings,
    )


def _acquire_build_lock_or_wait(
    lock_path: Path,
    *,
    db: str,
    cache_key: str,
    current_path: Path,
    version_dir: Path,
    expected_source_hash: str,
    timeout_seconds: float,
    poll_seconds: float,
) -> str | None:
    """Acquire the build lock or wait for another worker's published cache."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    deadline = time.monotonic() + timeout_seconds
    payload = json.dumps(
        {
            "cache_key": cache_key,
            "created_at": time.time(),
            "pid": os.getpid(),
            "token": token,
        },
        sort_keys=True,
    )

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            loaded = _load_valid_index(
                db=db,
                cache_key=cache_key,
                cache_dir=version_dir,
                expected_source_hash=expected_source_hash,
            )
            if loaded is not None:
                return None
            current = _load_current_if_matching(
                db=db,
                current_path=current_path,
                cache_key=cache_key,
                expected_source_hash=expected_source_hash,
            )
            if current is not None:
                return None
            if time.monotonic() >= deadline:
                return None
            time.sleep(min(poll_seconds, max(deadline - time.monotonic(), 0.0)))
            continue

        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
        return token


def _load_current_if_matching(
    *,
    db: str,
    current_path: Path,
    cache_key: str,
    expected_source_hash: str,
) -> SchemaRetrievalIndex | None:
    try:
        pointer = _read_current_pointer(current_path)
    except RetrievalIndexError:
        return None
    if pointer.get("cache_key") != cache_key:
        return None
    cache_dir = Path(str(pointer.get("cache_dir") or ""))
    if not cache_dir:
        return None
    return _load_valid_index(
        db=db,
        cache_key=cache_key,
        cache_dir=cache_dir,
        expected_source_hash=expected_source_hash,
    )


def _release_build_lock(lock_path: Path, token: str) -> None:
    """Remove only the lock file created by this process."""

    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return
    if payload.get("token") == token:
        lock_path.unlink(missing_ok=True)


def _publish_version_directory(temp_dir: Path, final_dir: Path) -> bool:
    """Publish a built version directory without overwriting an existing one."""

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False
    try:
        temp_dir.rename(final_dir)
    except FileExistsError:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False
    return True


def _quarantine_invalid_version_directory(version_dir: Path) -> None:
    """Move an invalid version directory aside without overwriting it."""

    quarantine = version_dir.with_name(f".{version_dir.name}.invalid.{uuid4().hex}")
    try:
        version_dir.rename(quarantine)
    except FileNotFoundError:
        return


def _write_current_pointer(current_path: Path, *, db: str, cache_key: str, cache_dir: Path) -> None:
    """Atomically point a database cache root at the current version."""

    current_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = current_path.with_name(f".current.{os.getpid()}.{uuid4().hex}.json.tmp")
    _write_json(
        temp_path,
        {
            "db": db,
            "cache_key": cache_key,
            "cache_dir": str(cache_dir),
        },
    )
    os.replace(temp_path, current_path)


def _read_current_pointer(current_path: Path) -> dict[str, Any]:
    try:
        pointer = _read_json(current_path)
    except FileNotFoundError as exc:
        raise RetrievalIndexError(
            f"missing current retrieval index pointer: {current_path}"
        ) from exc
    if not isinstance(pointer, dict):
        raise RetrievalIndexError(f"invalid current retrieval index pointer: {current_path}")
    return pointer


def _embedding_model_fingerprint(
    provider: EmbeddingProvider,
    *,
    configured_model: str,
    explicit_metadata: Mapping[str, object] | None,
) -> dict[str, object]:
    """Collect model metadata that should affect cache invalidation when available."""

    fingerprint: dict[str, object] = {
        "configured_model": configured_model,
        "provider_class": f"{provider.__class__.__module__}.{provider.__class__.__qualname__}",
    }
    for attribute in ("model", "model_digest", "digest"):
        value = getattr(provider, attribute, None)
        if value is not None:
            fingerprint[attribute] = _json_safe(value)
    provider_metadata = getattr(provider, "model_metadata", None)
    if callable(provider_metadata):
        provider_metadata = provider_metadata()
    if provider_metadata is not None:
        fingerprint["provider_metadata"] = _json_safe(provider_metadata)
    if explicit_metadata:
        fingerprint["explicit_metadata"] = _json_safe(dict(explicit_metadata))
    return fingerprint


def _default_embedding_provider(config: SchemaRetrievalConfig) -> EmbeddingProvider:
    return OllamaEmbeddingProvider(
        model=config.embedding_model,
        base_url=config.ollama_base_url,
    )


def _load_db_index(db: str) -> dict[str, TableSchema]:
    return load_db_index(db)


def _version_dir(cache_root: Path, db: str, cache_key: str) -> Path:
    return cache_root / _safe_path_segment(db) / "versions" / cache_key


def _current_pointer_path(cache_root: Path, db: str) -> Path:
    return cache_root / _safe_path_segment(db) / "current.json"


def _build_lock_path(cache_root: Path, db: str) -> Path:
    return cache_root / _safe_path_segment(db) / "build.lock"


def _new_temp_version_dir(cache_root: Path, db: str, cache_key: str) -> Path:
    versions_dir = cache_root / _safe_path_segment(db) / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{cache_key}.",
            suffix=".tmp",
            dir=versions_dir,
        )
    )


def _safe_path_segment(value: str) -> str:
    segment = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return segment or "default"


def _tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(text)]


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")
    return payload


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("JSONL row must be an object")
            rows.append(payload)
    return rows
