"""Load and render compact Snowflake schema indexes."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

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


def _db_schema_summary(db_index: dict[str, TableSchema]) -> str:
    """Render one compact all-table schema summary for the selector."""

    parts: list[str] = []
    for table_name in sorted(db_index):
        table = db_index[table_name]
        columns = ", ".join(_column_summary(column) for column in table.columns)
        parts.append(f"Table {table_name}: {columns}")
    return "\n".join(parts)


def _column_summary(column: Any) -> str:
    """Keep one column summary short enough for selector prompts."""

    summary = column.name
    if column.type:
        summary += f" [{column.type}]"
    if column.description:
        summary += f" - {column.description}"
    elif column.sample_values:
        preview = ", ".join(column.sample_values[:2])
        summary += f" - sample values: {preview}"
    return summary


def _sanitize_llm_tables(
    requested_tables: list[str],
    db_index: dict[str, TableSchema],
) -> list[str]:
    """Keep valid unique table names and surface an empty selection when none survive."""

    valid_tables = {table_name.lower(): table_name for table_name in db_index}
    suffix_lookup: dict[str, list[str]] = {}
    for table_identity in db_index:
        parts = table_identity.lower().split(".")
        for start in range(len(parts)):
            suffix = ".".join(parts[start:])
            suffix_lookup.setdefault(suffix, []).append(table_identity)

    selected_tables: list[str] = []
    for table_name in requested_tables:
        normalized = table_name.strip().lower()
        canonical = valid_tables.get(normalized)
        if canonical is None:
            matches = suffix_lookup.get(normalized, [])
            if len(matches) == 1:
                canonical = matches[0]
        if canonical is None or canonical in selected_tables:
            continue
        selected_tables.append(canonical)
    return selected_tables
