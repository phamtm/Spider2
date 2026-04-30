"""Execute read-only SQL through Snowflake."""

from __future__ import annotations

import importlib
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from sol01.logging import get_logger
from sol01.models import ExecutionResult

METHOD_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CREDENTIAL_PATH = METHOD_ROOT / "snowflake_credential.json"
logger = get_logger(__name__)


def load_snowflake_credentials(credential_path: Path | None = None) -> dict[str, Any]:
    """Load Snowflake connector kwargs from the local credential JSON file."""

    credential_path = _credential_path(credential_path)
    payload = json.loads(credential_path.read_text(encoding="utf-8"))
    missing = [
        name
        for name in ("username", "password", "account", "role", "warehouse")
        if not str(payload.get(name) or "").strip()
    ]
    if missing:
        raise ValueError(f"Snowflake credential file is missing: {', '.join(missing)}")

    credentials = {
        "user": str(payload["username"]).strip(),
        "password": str(payload["password"]).strip(),
        "account": str(payload["account"]).strip(),
        "role": str(payload["role"]).strip(),
        "warehouse": str(payload["warehouse"]).strip(),
    }
    session_parameters = payload.get("session_parameters")
    if isinstance(session_parameters, dict):
        credentials["session_parameters"] = session_parameters
    return credentials


def fetch_query_dataframe(
    sql: str,
    *,
    db: str,
    credential_path: Path | None = None,
    connector: Any | None = None,
) -> pd.DataFrame:
    """Run one query against Snowflake and return the full DataFrame."""

    connection = open_connection(db=db, credential_path=credential_path, connector=connector)
    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        return cursor.fetch_pandas_all()
    finally:
        cursor.close()
        connection.close()


def execute_sql(
    sql: str,
    *,
    db: str,
    credential_path: Path | None = None,
    csv_path: Path | None = None,
    sample_limit: int = 3,
    connector: Any | None = None,
) -> ExecutionResult:
    """Execute one query, optionally write its CSV output, and return a compact summary."""

    try:
        dataframe = fetch_query_dataframe(
            sql,
            db=db,
            credential_path=credential_path,
            connector=connector,
        )
    except Exception as exc:
        logger.warning("sql execution failed", db=db, error=str(exc))
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

    logger.info(
        "sql executed",
        db=db,
        row_count=len(dataframe),
        columns=[str(column) for column in dataframe.columns],
        csv_path=str(csv_path) if csv_path is not None else None,
    )
    return ExecutionResult(
        ok=True,
        row_count=len(dataframe),
        columns=[str(column) for column in dataframe.columns],
        sample_rows=_dataframe_records(dataframe.head(sample_limit)),
        csv_path=str(csv_path) if csv_path is not None else None,
        error=None,
    )


def open_connection(
    *,
    db: str,
    credential_path: Path | None = None,
    connector: Any | None = None,
) -> Any:
    """Open a Snowflake connection for one task database."""

    connector_module = connector or importlib.import_module("snowflake.connector")
    credentials = load_snowflake_credentials(credential_path)
    return connector_module.connect(**credentials, database=db)


def _credential_path(credential_path: Path | None) -> Path:
    """Resolve the local credential file path."""

    if credential_path is not None:
        return credential_path

    env_path = os.environ.get("SOL01_SNOWFLAKE_CREDENTIAL_PATH")
    if env_path and env_path.strip():
        path = Path(env_path.strip())
        return path if path.is_absolute() else METHOD_ROOT / path

    return DEFAULT_CREDENTIAL_PATH


def _dataframe_records(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a DataFrame slice into JSON-friendly row dictionaries.

    Result sets can repeat column names after joins, so duplicate columns are
    suffixed in summaries instead of silently dropping values.
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
    if isinstance(value, Decimal):
        return str(value)
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
