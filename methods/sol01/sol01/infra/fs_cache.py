"""Filesystem cache helpers shared across modules."""

from __future__ import annotations

from pathlib import Path


def path_signature(path: Path) -> tuple[int, int] | None:
    """Return a cheap cache key for one file path, or None when it is missing."""

    try:
        stat_result = path.stat()
    except FileNotFoundError:
        return None
    return stat_result.st_mtime_ns, stat_result.st_size
