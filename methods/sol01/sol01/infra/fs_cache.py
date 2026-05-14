"""Filesystem cache helpers shared across modules."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4


def path_signature(path: Path) -> tuple[int, int] | None:
    """Return a cheap cache key for one file path, or None when it is missing."""

    try:
        stat_result = path.stat()
    except FileNotFoundError:
        return None
    return stat_result.st_mtime_ns, stat_result.st_size


def safe_path_segment(value: str) -> str:
    """Sanitize an arbitrary string for use as a filesystem path component."""

    segment = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return segment or "default"


def stable_hash(payload: object) -> str:
    """Return a deterministic SHA-256 hex digest of a JSON-serialisable payload."""

    encoded = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from *path*. Raises FileNotFoundError or ValueError on bad input."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")
    return payload


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def atomic_write_json(path: Path, payload: object) -> None:
    """Atomically replace *path* with a JSON file by writing to a temp file first."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    write_json(temp_path, payload)
    os.replace(temp_path, path)


def quarantine_invalid_directory(path: Path) -> None:
    """Rename an invalid directory aside without overwriting anything."""

    quarantine = path.with_name(f".{path.name}.invalid.{uuid4().hex}")
    try:
        path.rename(quarantine)
    except FileNotFoundError:
        return


def publish_version_directory(temp_dir: Path, final_dir: Path) -> bool:
    """Rename *temp_dir* to *final_dir* atomically; return False when *final_dir* already exists."""

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


def acquire_build_lock_or_wait(
    lock_path: Path,
    *,
    is_done: Callable[[], bool] | None = None,
    timeout_seconds: float,
    poll_seconds: float,
) -> str | None:
    """Acquire a build lock file and return a token, or return None on timeout.

    While waiting for the lock, *is_done* is polled on each iteration — if it
    returns True the function returns None immediately (another worker finished).
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    deadline = time.monotonic() + timeout_seconds
    payload = json.dumps(
        {
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
            if is_done is not None and is_done():
                return None
            if time.monotonic() >= deadline:
                return None
            time.sleep(min(poll_seconds, max(deadline - time.monotonic(), 0.0)))
            continue

        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
        return token


def release_build_lock(lock_path: Path, token: str) -> None:
    """Remove the lock file only if its token matches the one we wrote."""

    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return
    if payload.get("token") == token:
        lock_path.unlink(missing_ok=True)


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
