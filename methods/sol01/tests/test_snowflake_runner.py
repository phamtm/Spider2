"""Tests for Snowflake execution and result profiling helpers."""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from sol01.execution.profiling import profile_dataframe
from sol01.execution.snowflake_runner import (
    DEFAULT_CREDENTIAL_PATH,
    execute_sql,
    load_snowflake_credentials,
)


def test_default_credential_path_points_to_method_root():
    assert (
        DEFAULT_CREDENTIAL_PATH == Path(__file__).resolve().parents[1] / "snowflake_credential.json"
    )


def test_load_snowflake_credentials_normalizes_username_and_keeps_session_parameters(
    tmp_path: Path,
):
    credential_path = _write_credentials(
        tmp_path / "snowflake_credential.json",
        session_parameters={"QUERY_TAG": "sol01-test"},
    )

    credentials = load_snowflake_credentials(credential_path)

    assert credentials == {
        "user": "alice",
        "password": "pat-token",
        "account": "RSRSBDK-YDB67606",
        "role": "PARTICIPANT",
        "warehouse": "COMPUTE_WH_PARTICIPANT",
        "session_parameters": {"QUERY_TAG": "sol01-test"},
    }


def test_load_snowflake_credentials_rejects_missing_required_fields(tmp_path: Path):
    credential_path = tmp_path / "snowflake_credential.json"
    credential_path.write_text('{"username": "alice"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="password, account, role, warehouse"):
        load_snowflake_credentials(credential_path)


def test_execute_sql_writes_csv_and_passes_database_to_connector(tmp_path: Path):
    credential_path = _write_credentials(tmp_path / "snowflake_credential.json")
    csv_path = tmp_path / "result.csv"
    connector = FakeConnector(
        pd.DataFrame(
            [
                {"customer": "alice", "amount": 10.5},
                {"customer": "bob", "amount": 12.0},
                {"customer": "alice", "amount": None},
                {"customer": "carol", "amount": 7.5},
            ]
        )
    )

    result = execute_sql(
        "SELECT customer, amount FROM TEST_DB.PUBLIC.SALES ORDER BY customer",
        db="TEST_DB",
        credential_path=credential_path,
        csv_path=csv_path,
        connector=connector,
    )

    assert connector.connect_kwargs["database"] == "TEST_DB"
    assert connector.executed_sql == [
        "SELECT customer, amount FROM TEST_DB.PUBLIC.SALES ORDER BY customer"
    ]
    assert result.ok is True
    assert result.row_count == 4
    assert result.columns == ["customer", "amount"]
    assert result.sample_rows == [
        {"customer": "alice", "amount": 10.5},
        {"customer": "bob", "amount": 12.0},
        {"customer": "alice", "amount": None},
    ]
    assert result.csv_path == str(csv_path)
    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "customer,amount",
        "alice,10.5",
        "bob,12.0",
        "alice,",
        "carol,7.5",
    ]


def test_execute_sql_does_not_report_csv_path_on_failure(tmp_path: Path):
    credential_path = _write_credentials(tmp_path / "snowflake_credential.json")
    csv_path = tmp_path / "missing.csv"

    result = execute_sql(
        "SELECT * FROM TEST_DB.PUBLIC.MISSING_TABLE",
        db="TEST_DB",
        credential_path=credential_path,
        csv_path=csv_path,
        connector=FailingConnector("object does not exist"),
    )

    assert result.ok is False
    assert result.csv_path is None
    assert "object does not exist" in result.error
    assert csv_path.exists() is False


def test_profile_dataframe_handles_duplicate_column_names():
    dataframe = pd.DataFrame(
        [
            ["alice", "Sydney"],
            ["bob", "Melbourne"],
            ["alice", "Sydney"],
            ["carol", None],
        ],
        columns=["name", "name"],
    )

    profile = profile_dataframe(dataframe, top_k=2)

    assert profile["columns"] == ["name", "name"]
    assert profile["sample_rows"][0] == {"name": "alice", "name__2": "Sydney"}
    assert profile["null_counts"] == {"name": 0, "name__2": 1}
    assert profile["distinct_counts"] == {"name": 3, "name__2": 3}
    assert profile["top_values"]["name"][0] == {"value": "alice", "count": 2}
    assert profile["top_values"]["name__2"][0] == {"value": "Sydney", "count": 2}


@pytest.mark.parametrize(
    "dataframe, expected_sample, expected_min, expected_max",
    [
        pytest.param(
            pd.DataFrame(
                [
                    {"percentage": Decimal("78.15")},
                    {"percentage": Decimal("80.00")},
                    {"percentage": Decimal("78.15")},
                ]
            ),
            [{"percentage": "78.15"}, {"percentage": "80.00"}, {"percentage": "78.15"}],
            {"percentage": "78.15"},
            {"percentage": "80.00"},
            id="decimal-serialization",
        ),
        pytest.param(
            pd.DataFrame(
                [
                    {"day": date(2023, 6, 1), "seen_at": datetime(2023, 6, 1, 12, 30, 5)},
                    {"day": date(2023, 6, 2), "seen_at": datetime(2023, 6, 2, 8, 15, 0)},
                ]
            ),
            [
                {"day": "2023-06-01", "seen_at": "2023-06-01T12:30:05"},
                {"day": "2023-06-02", "seen_at": "2023-06-02T08:15:00"},
            ],
            {"day": "2023-06-01", "seen_at": "2023-06-01T12:30:05"},
            {"day": "2023-06-02", "seen_at": "2023-06-02T08:15:00"},
            id="date-serialization",
        ),
    ],
)
def test_profile_dataframe_serializes_special_types(dataframe, expected_sample, expected_min, expected_max):
    profile = profile_dataframe(dataframe, top_k=2)

    assert profile["sample_rows"] == expected_sample
    assert profile["min_values"] == expected_min
    assert profile["max_values"] == expected_max
    json.dumps(profile, sort_keys=True)


def _write_credentials(path: Path, **extra: Any) -> Path:
    payload = {
        "username": "alice",
        "password": "pat-token",
        "account": "RSRSBDK-YDB67606",
        "role": "PARTICIPANT",
        "warehouse": "COMPUTE_WH_PARTICIPANT",
        **extra,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class FakeConnector:

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe
        self.connect_kwargs: dict[str, Any] = {}
        self.executed_sql: list[str] = []

    def connect(self, **kwargs: Any) -> FakeConnection:
        self.connect_kwargs = kwargs
        return FakeConnection(self)


class FailingConnector(FakeConnector):

    def __init__(self, error: str) -> None:
        super().__init__(pd.DataFrame())
        self.error = error


class FakeConnection:

    def __init__(self, connector: FakeConnector) -> None:
        self.connector = connector
        self.closed = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.connector)

    def close(self) -> None:
        self.closed = True


class FakeCursor:

    def __init__(self, connector: FakeConnector) -> None:
        self.connector = connector
        self.closed = False

    def execute(self, sql: str) -> None:
        if isinstance(self.connector, FailingConnector):
            raise RuntimeError(self.connector.error)
        self.connector.executed_sql.append(sql)

    def fetch_pandas_all(self) -> pd.DataFrame:
        return self.connector.dataframe

    def close(self) -> None:
        self.closed = True
