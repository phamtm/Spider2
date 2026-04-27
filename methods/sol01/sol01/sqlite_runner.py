"""Execute read-only SQL against an in-memory copy of a Spider2 SQLite database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from sol01.models import ExecutionResult
from sol01.tasks import REPO_ROOT

SQLITE_DB_ROOT = REPO_ROOT / "spider2-lite" / "resource" / "databases" / "spider2-localdb"


def resolve_sqlite_path(db: str, *, db_root: Path = SQLITE_DB_ROOT) -> Path:
    """Resolve a task DB name to the shipped SQLite file, allowing small name mismatches."""

    exact_path = db_root / f"{db}.sqlite"
    if exact_path.exists():
        return exact_path

    normalized_name = _normalize_db_name(db)
    matches = [
        path
        for path in db_root.glob("*.sqlite")
        if not path.name.startswith("._") and _normalize_db_name(path.stem) == normalized_name
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Could not find a SQLite file for database '{db}'.")
    raise ValueError(f"Found multiple SQLite files for database '{db}'.")


def open_snapshot_connection(
    *,
    db: str | None = None,
    db_path: Path | None = None,
) -> sqlite3.Connection:
    """Open an isolated in-memory copy of the requested SQLite database."""

    resolved_path = _resolve_db_path(db=db, db_path=db_path)
    source = sqlite3.connect(f"file:{resolved_path}?mode=ro", uri=True)
    destination = sqlite3.connect(":memory:")
    try:
        source.backup(destination)
    finally:
        source.close()
    return destination


def fetch_query_dataframe(
    sql: str,
    *,
    db: str | None = None,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Run one query against the in-memory snapshot and return the full DataFrame."""

    connection = open_snapshot_connection(db=db, db_path=db_path)
    try:
        return pd.read_sql_query(sql, connection)
    finally:
        connection.close()


def execute_sql(
    sql: str,
    *,
    db: str | None = None,
    db_path: Path | None = None,
    csv_path: Path | None = None,
    sample_limit: int = 3,
) -> ExecutionResult:
    """Execute one query, optionally write its CSV output, and return a compact summary."""

    try:
        dataframe = fetch_query_dataframe(sql, db=db, db_path=db_path)
    except Exception as exc:
        return ExecutionResult(
            ok=False,
            row_count=0,
            columns=[],
            sample_rows=[],
            csv_path=None,
            error=str(exc),
        )

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(csv_path, index=False)

    return ExecutionResult(
        ok=True,
        row_count=len(dataframe),
        columns=[str(column) for column in dataframe.columns],
        sample_rows=_dataframe_records(dataframe.head(sample_limit)),
        csv_path=str(csv_path) if csv_path is not None else None,
        error=None,
    )


def _resolve_db_path(*, db: str | None, db_path: Path | None) -> Path:
    """Resolve the database path from either an explicit path or a task DB name."""

    if db_path is not None:
        return db_path
    if db is None:
        raise ValueError("Either db or db_path must be provided.")
    return resolve_sqlite_path(db)


def _normalize_db_name(name: str) -> str:
    """Normalize DB names so underscores, dashes, and case do not matter."""

    return "".join(character.lower() for character in name if character.isalnum())


def _dataframe_records(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a DataFrame slice into JSON-friendly row dictionaries.

    Result sets can repeat column names after joins, so we suffix duplicates in
    the summary rows instead of silently dropping values.
    """

    record_keys = _record_keys(dataframe.columns)
    records: list[dict[str, object]] = []
    for row in dataframe.itertuples(index=False, name=None):
        records.append(
            {
                record_key: _clean_value(value)
                for record_key, value in zip(record_keys, row, strict=True)
            }
        )
    return records


def _clean_value(value: object) -> object:
    """Convert pandas and NumPy scalars into plain Python values."""

    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _record_keys(columns: pd.Index) -> list[str]:
    """Build stable summary keys even when SQL returns duplicate column names."""

    seen_counts: dict[str, int] = {}
    keys: list[str] = []
    for column in columns:
        base_name = str(column)
        seen_counts[base_name] = seen_counts.get(base_name, 0) + 1
        count = seen_counts[base_name]
        if count == 1:
            keys.append(base_name)
        else:
            keys.append(f"{base_name}__{count}")
    return keys
