import json
from pathlib import Path

from sol01.index import (
    CACHE_PATH,
    REPO_ROOT,
    SQLITE_METADATA_ROOT,
    build_db_index,
    build_index_cache,
)


def test_sqlite_metadata_root_exists():
    assert SQLITE_METADATA_ROOT.exists()


def test_build_db_index_for_e_commerce():
    index = build_db_index("E_commerce")

    assert len(index) == 11
    assert "customers" in index
    assert "orders" in index

    customers = index["customers"]
    assert customers.name == "customers"
    assert customers.columns[0].name == "customer_id"
    assert customers.columns[0].type == "TEXT"
    assert customers.sample_rows
    assert "customer_unique_id" in customers.searchable_text
    assert "praia grande" in customers.searchable_text


def test_build_index_cache_writes_cache_file(tmp_path):
    cache_path = tmp_path / "index.json"

    payload = build_index_cache(cache_path=cache_path)

    assert cache_path.exists()
    assert "E_commerce" in payload
    assert "customers" in payload["E_commerce"]


def test_default_cache_path_points_inside_method_directory():
    assert CACHE_PATH == (REPO_ROOT / "methods" / "sol01" / ".cache" / "index.json").resolve()


def test_build_db_index_ignores_malformed_metadata_rows(tmp_path):
    metadata_root = tmp_path / "sqlite"
    db_dir = metadata_root / "broken_db"
    db_dir.mkdir(parents=True)

    (db_dir / "DDL.csv").write_text(
        "table_name,DDL\nbroken,\"CREATE TABLE broken (id TEXT, name TEXT);\"\n",
        encoding="utf-8",
    )
    (db_dir / "broken.json").write_text(
        json.dumps(
            {
                "table_name": "broken",
                "table_fullname": "broken",
                "column_names": ["id", "name"],
                "column_types": ["TEXT"],
                "description": None,
                "sample_rows": [{"id": "1", "name": "alpha"}, "ignored row"],
            }
        ),
        encoding="utf-8",
    )

    index = build_db_index("broken_db", metadata_root=metadata_root)

    broken = index["broken"]
    assert [column.name for column in broken.columns] == ["id", "name"]
    assert broken.columns[0].type == "TEXT"
    assert broken.columns[1].type is None
    assert broken.columns[0].sample_values == ["1"]
    assert broken.columns[1].sample_values == ["alpha"]
    assert broken.sample_rows == [{"id": "1", "name": "alpha"}]
    assert "ignored row" not in broken.searchable_text
