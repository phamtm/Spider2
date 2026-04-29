"""Build a compact schema index from the Spider2-snow metadata files."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

from sol01.logging import get_logger
from sol01.models import ColumnSchema, TableSchema

REPO_ROOT = Path(__file__).resolve().parents[3]
SNOW_METADATA_ROOT = REPO_ROOT / "spider2-snow" / "resource" / "databases"
CACHE_PATH = (REPO_ROOT / "methods" / "sol01" / ".cache" / "snow_index.json").resolve()
logger = get_logger(__name__)


def build_db_index(db: str, *, metadata_root: Path = SNOW_METADATA_ROOT) -> dict[str, TableSchema]:
    """Build table metadata for one Snowflake database."""

    db_dir = metadata_root / db
    logger.info("db index start", db=db, metadata_root=str(metadata_root))
    index: dict[str, TableSchema] = {}

    for schema_dir in _schema_dirs(db_dir):
        schema_name = None if schema_dir == db_dir else schema_dir.name
        ddl_by_table = _read_ddl_map(schema_dir / "DDL.csv")

        for metadata_path in sorted(schema_dir.glob("*.json")):
            metadata = _read_table_metadata(metadata_path)
            table_name = _short_table_name(metadata, metadata_path)
            table_identity = _table_identity(
                metadata,
                db=db,
                schema=schema_name,
                table_name=table_name,
            )
            index[table_identity] = TableSchema(
                name=table_name,
                database_name=db,
                schema_name=schema_name,
                full_name=table_identity,
                ddl=_table_ddl(
                    ddl_by_table,
                    metadata=metadata,
                    schema=schema_name,
                    table_name=table_name,
                    table_identity=table_identity,
                ),
                columns=_build_columns(metadata),
                sample_rows=_metadata_rows(metadata),
                searchable_text=_build_searchable_text(table_identity, table_name, metadata),
            )

    logger.info("db index complete", db=db, table_count=len(index))
    return index


def build_index_cache(
    *,
    metadata_root: Path = SNOW_METADATA_ROOT,
    cache_path: Path = CACHE_PATH,
) -> dict[str, dict[str, TableSchema]]:
    """Build and persist the full Snowflake metadata cache used by later retrieval."""

    logger.info("index cache start", metadata_root=str(metadata_root), cache_path=str(cache_path))
    payload = {
        db_dir.name: build_db_index(db_dir.name, metadata_root=metadata_root)
        for db_dir in sorted(path for path in metadata_root.iterdir() if path.is_dir())
    }

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
    logger.info("index cache complete", database_count=len(payload), cache_path=str(cache_path))
    return payload


def _schema_dirs(db_dir: Path) -> list[Path]:
    """Return schema metadata directories for one database."""

    if (db_dir / "DDL.csv").exists():
        return [db_dir]
    return sorted(
        path for path in db_dir.iterdir() if path.is_dir() and (path / "DDL.csv").exists()
    )


def _read_ddl_map(path: Path) -> dict[str, str]:
    """Read the table DDL strings from the Spider2 metadata CSV."""

    with path.open(encoding="utf-8", newline="") as handle:
        return {row["table_name"]: row["DDL"] for row in csv.DictReader(handle)}


def _read_table_metadata(path: Path) -> dict[str, object]:
    """Read the per-table JSON metadata file."""

    return json.loads(path.read_text(encoding="utf-8"))


def _build_columns(metadata: dict[str, object]) -> list[ColumnSchema]:
    """Pair column names, types, descriptions, and example values by position."""

    column_names = _metadata_list(metadata, "column_names")
    column_types = _metadata_list(metadata, "column_types")
    descriptions = _metadata_list(metadata, "description")
    sample_rows = _metadata_rows(metadata)
    columns: list[ColumnSchema] = []

    for index, name in enumerate(column_names):
        columns.append(
            ColumnSchema(
                name=name,
                type=_safe_list_get(column_types, index),
                description=_safe_list_get(descriptions, index),
                sample_values=_sample_values(sample_rows, name),
            )
        )

    return columns


def _short_table_name(metadata: dict[str, object], metadata_path: Path) -> str:
    """Return the unqualified table name from metadata."""

    raw_name = str(metadata.get("table_name") or metadata_path.stem).strip()
    return raw_name.split(".")[-1] if raw_name else metadata_path.stem


def _table_identity(
    metadata: dict[str, object],
    *,
    db: str,
    schema: str | None,
    table_name: str,
) -> str:
    """Return the canonical table identity used by retrieval."""

    full_name = str(metadata.get("table_fullname") or "").strip()
    if full_name:
        return full_name

    raw_name = str(metadata.get("table_name") or "").strip()
    if raw_name and schema is None:
        return raw_name
    if raw_name.count(".") >= 2:
        return raw_name
    if schema is not None:
        return f"{db}.{schema}.{table_name}"
    return table_name


def _table_ddl(
    ddl_by_table: dict[str, str],
    *,
    metadata: dict[str, object],
    schema: str | None,
    table_name: str,
    table_identity: str,
) -> str:
    """Find the best DDL entry for a table across short and qualified names."""

    raw_name = str(metadata.get("table_name") or "").strip()
    candidates = [table_name, raw_name, table_identity]
    if schema is not None:
        candidates.append(f"{schema}.{table_name}")

    for candidate in candidates:
        if candidate in ddl_by_table:
            return ddl_by_table[candidate]
    return ""


def _build_searchable_text(
    table_identity: str, table_name: str, metadata: dict[str, object]
) -> str:
    """Flatten useful names, descriptions, and examples into one search string."""

    parts: list[str] = [table_identity, table_name, str(metadata.get("table_name", ""))]
    parts.extend(str(name) for name in _metadata_list(metadata, "column_names"))
    parts.extend(str(value) for value in _metadata_list(metadata, "column_types"))
    parts.extend(str(value) for value in _metadata_list(metadata, "description"))

    for row in _metadata_rows(metadata):
        if isinstance(row, dict):
            parts.extend(str(value) for value in row.values())

    return " ".join(part for part in parts if part).strip()


def _sample_values(sample_rows: list[object], column_name: str) -> list[str]:
    """Extract up to five stringified example values for one column."""

    values: list[str] = []
    for row in sample_rows:
        if not isinstance(row, dict):
            continue
        value = row.get(column_name)
        if value is None:
            continue
        values.append(str(value))
        if len(values) == 5:
            break
    return values


def _safe_list_get(values: list[object], index: int) -> str | None:
    """Return a cleaned string value when the metadata list has that position."""

    if index >= len(values):
        return None

    value = values[index]
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def _metadata_list(metadata: dict[str, object], key: str) -> list[object]:
    """Return one metadata list field, or an empty list when the value is missing."""

    value = metadata.get(key, [])
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _metadata_rows(metadata: dict[str, object]) -> list[dict[str, object]]:
    """Return sample rows as dictionaries, dropping malformed entries."""

    rows: list[dict[str, object]] = []
    for row in _metadata_list(metadata, "sample_rows"):
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows
