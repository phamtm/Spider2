"""Load and render compact Snowflake schema indexes."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from sol01.models import TableSchema
from sol01.schema.index import CACHE_PATH, build_db_index, build_index_cache


def load_db_index(db: str, *, cache_path: Path = CACHE_PATH) -> dict[str, TableSchema]:
    """Load one database index from the cache, building it when needed."""

    payload = load_index_cache(cache_path=cache_path) if cache_path.exists() else {}
    if db not in payload:
        payload = dict(payload)
        payload[db] = build_db_index(db)
        _write_index_cache(payload, cache_path)

    return dict(payload[db])


def load_index_cache(
    *,
    cache_path: Path = CACHE_PATH,
) -> dict[str, dict[str, TableSchema]]:
    """Load the cached schema index, or build it if the cache is missing."""

    signature = _path_signature(cache_path)
    if signature is None:
        return build_index_cache(cache_path=cache_path)

    return _load_index_cache_snapshot(str(cache_path.resolve()), signature)


@cache
def _load_index_cache_snapshot(
    cache_path: str,
    signature: tuple[int, int],
) -> dict[str, dict[str, TableSchema]]:
    """Load one cached schema snapshot and keep it in memory."""

    raw_payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    return {
        db_name: {
            table_name: TableSchema.model_validate(table_schema)
            for table_name, table_schema in tables.items()
        }
        for db_name, tables in raw_payload.items()
    }


def _write_index_cache(
    payload: dict[str, dict[str, TableSchema]],
    cache_path: Path,
) -> None:
    """Persist a schema index cache payload."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                db: {table: schema.model_dump(mode="json") for table, schema in tables.items()}
                for db, tables in payload.items()
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _path_signature(path: Path) -> tuple[int, int] | None:
    """Return a cheap cache key for one file path, or None when it is missing."""

    try:
        stat_result = path.stat()
    except FileNotFoundError:
        return None
    return stat_result.st_mtime_ns, stat_result.st_size
