"""Tests for SQLite execution and result profiling helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sol01.profiling import profile_sql
from sol01.sqlite_runner import execute_sql, resolve_sqlite_path


def test_resolve_sqlite_path_handles_normalized_db_names():
    assert resolve_sqlite_path("E_commerce").name == "E_commerce.sqlite"
    assert resolve_sqlite_path("DB_IMDB").name == "Db-IMDB.sqlite"
    assert resolve_sqlite_path("SQLITE_SAKILA").name == "sqlite-sakila.sqlite"


def test_execute_sql_writes_csv_for_simple_select(tmp_path):
    db_path = _build_test_db(tmp_path / "sample.sqlite")
    csv_path = tmp_path / "result.csv"

    result = execute_sql(
        "SELECT customer, amount FROM sales ORDER BY id",
        db_path=db_path,
        csv_path=csv_path,
    )

    assert result.ok is True
    assert result.row_count == 4
    assert result.columns == ["customer", "amount"]
    assert result.sample_rows == [
        {"customer": "alice", "amount": 10.5},
        {"customer": "bob", "amount": 12.0},
        {"customer": "alice", "amount": None},
    ]
    assert result.csv_path == str(csv_path)
    assert csv_path.exists()
    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "customer,amount",
        "alice,10.5",
        "bob,12.0",
        "alice,",
        "carol,7.5",
    ]


def test_profile_sql_returns_bounded_stats(tmp_path):
    db_path = _build_test_db(tmp_path / "sample.sqlite")

    profile = profile_sql(
        "SELECT customer, amount, city FROM sales ORDER BY id",
        db_path=db_path,
        top_k=2,
    )

    assert profile["row_count"] == 4
    assert profile["profile_row_count"] == 4
    assert profile["columns"] == ["customer", "amount", "city"]
    assert profile["sample_rows"][:2] == [
        {"customer": "alice", "amount": 10.5, "city": "Sydney"},
        {"customer": "bob", "amount": 12.0, "city": "Melbourne"},
    ]
    assert profile["null_counts"] == {"customer": 0, "amount": 1, "city": 1}
    assert profile["distinct_counts"]["customer"] == 3
    assert profile["min_values"]["amount"] == 7.5
    assert profile["max_values"]["amount"] == 12.0
    assert profile["top_values"]["customer"][0] == {"value": "alice", "count": 2}
    assert profile["top_values"]["city"] == [
        {"value": "Sydney", "count": 2},
        {"value": "Melbourne", "count": 1},
    ]


def test_execute_sql_does_not_report_csv_path_on_failure(tmp_path):
    db_path = _build_test_db(tmp_path / "sample.sqlite")
    csv_path = tmp_path / "missing.csv"

    result = execute_sql(
        "SELECT * FROM unknown_table",
        db_path=db_path,
        csv_path=csv_path,
    )

    assert result.ok is False
    assert result.csv_path is None
    assert csv_path.exists() is False


def test_execute_sql_keeps_duplicate_column_values_in_sample_rows(tmp_path):
    db_path = _build_test_db(tmp_path / "sample.sqlite")

    result = execute_sql(
        "SELECT customer AS name, city AS name FROM sales ORDER BY id",
        db_path=db_path,
    )

    assert result.ok is True
    assert result.columns == ["name", "name"]
    assert result.sample_rows[0] == {"name": "alice", "name__2": "Sydney"}
    assert result.sample_rows[1] == {"name": "bob", "name__2": "Melbourne"}


def test_profile_sql_handles_duplicate_column_names(tmp_path):
    db_path = _build_test_db(tmp_path / "sample.sqlite")

    profile = profile_sql(
        "SELECT customer AS name, city AS name FROM sales ORDER BY id",
        db_path=db_path,
        top_k=2,
    )

    assert profile["columns"] == ["name", "name"]
    assert profile["sample_rows"][0] == {"name": "alice", "name__2": "Sydney"}
    assert profile["null_counts"] == {"name": 0, "name__2": 1}
    assert profile["distinct_counts"] == {"name": 3, "name__2": 3}
    assert profile["top_values"]["name"][0] == {"value": "alice", "count": 2}
    assert profile["top_values"]["name__2"][0] == {"value": "Sydney", "count": 2}


def _build_test_db(path: Path) -> Path:
    """Create a tiny SQLite database for execution tests."""

    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE sales (
                id INTEGER PRIMARY KEY,
                customer TEXT,
                amount REAL,
                city TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO sales (customer, amount, city) VALUES (?, ?, ?)",
            [
                ("alice", 10.5, "Sydney"),
                ("bob", 12.0, "Melbourne"),
                ("alice", None, "Sydney"),
                ("carol", 7.5, None),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return path
