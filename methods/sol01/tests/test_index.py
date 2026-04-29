import json
from pathlib import Path

from sol01.index import (
    CACHE_PATH,
    REPO_ROOT,
    SNOW_METADATA_ROOT,
    build_db_index,
    build_index_cache,
)
from sol01.retrieval import load_db_index, retrieve_schema

CUSTOMERS = "E_COMMERCE.E_COMMERCE.CUSTOMERS"
ORDER_ITEMS = "E_COMMERCE.E_COMMERCE.ORDER_ITEMS"
ORDER_PAYMENTS = "E_COMMERCE.E_COMMERCE.ORDER_PAYMENTS"
ORDERS = "E_COMMERCE.E_COMMERCE.ORDERS"
PRODUCTS = "E_COMMERCE.E_COMMERCE.PRODUCTS"
SELLERS = "E_COMMERCE.E_COMMERCE.SELLERS"


def _write_table_metadata(
    schema_dir: Path,
    *,
    table_name: str,
    ddl: str,
    column_names: list[str],
    sample_rows: list[dict[str, object]],
    column_types: list[str] | None = None,
    descriptions: list[str] | None = None,
) -> None:
    """Write one synthetic table metadata file for retrieval tests."""

    payload = {
        "table_name": table_name,
        "table_fullname": table_name,
        "column_names": column_names,
        "column_types": column_types or ["TEXT"] * len(column_names),
        "description": descriptions or [""] * len(column_names),
        "sample_rows": sample_rows,
    }
    (schema_dir / f"{table_name.rsplit('.', 1)[-1]}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_snow_metadata_root_exists():
    assert SNOW_METADATA_ROOT.exists()


def test_build_db_index_for_e_commerce():
    index = build_db_index("E_COMMERCE")

    assert len(index) == 11
    assert CUSTOMERS in index
    assert ORDERS in index

    customers = index[CUSTOMERS]
    assert customers.name == "CUSTOMERS"
    assert customers.database_name == "E_COMMERCE"
    assert customers.schema_name == "E_COMMERCE"
    assert customers.full_name == CUSTOMERS
    customer_id = next(column for column in customers.columns if column.name == "customer_id")
    assert customer_id.type == "TEXT"
    assert customers.sample_rows
    assert "customer_unique_id" in customers.searchable_text
    assert "mogi das cruzes" in customers.searchable_text


def test_build_index_cache_writes_cache_file(tmp_path):
    metadata_root = tmp_path / "databases"
    schema_dir = metadata_root / "E_COMMERCE" / "E_COMMERCE"
    schema_dir.mkdir(parents=True)
    (schema_dir / "DDL.csv").write_text(
        'table_name,DDL\nCUSTOMERS,"CREATE TABLE CUSTOMERS (customer_id TEXT);"\n',
        encoding="utf-8",
    )
    _write_table_metadata(
        schema_dir,
        table_name=CUSTOMERS,
        ddl="CREATE TABLE CUSTOMERS (customer_id TEXT);",
        column_names=["customer_id"],
        sample_rows=[{"customer_id": "c1"}],
    )
    cache_path = tmp_path / "index.json"

    payload = build_index_cache(metadata_root=metadata_root, cache_path=cache_path)

    assert cache_path.exists()
    assert "E_COMMERCE" in payload
    assert CUSTOMERS in payload["E_COMMERCE"]


def test_default_cache_path_points_inside_method_directory():
    assert CACHE_PATH == (REPO_ROOT / "methods" / "sol01" / ".cache" / "snow_index.json").resolve()


def test_build_db_index_ignores_malformed_metadata_rows(tmp_path):
    metadata_root = tmp_path / "databases"
    schema_dir = metadata_root / "BROKEN_DB" / "PUBLIC"
    schema_dir.mkdir(parents=True)

    (schema_dir / "DDL.csv").write_text(
        'table_name,DDL\nBROKEN,"CREATE TABLE BROKEN (id TEXT, name TEXT);"\n',
        encoding="utf-8",
    )
    (schema_dir / "BROKEN.json").write_text(
        json.dumps(
            {
                "table_name": "PUBLIC.BROKEN",
                "table_fullname": "BROKEN_DB.PUBLIC.BROKEN",
                "column_names": ["id", "name"],
                "column_types": ["TEXT"],
                "description": None,
                "sample_rows": [{"id": "1", "name": "alpha"}, "ignored row"],
            }
        ),
        encoding="utf-8",
    )

    index = build_db_index("BROKEN_DB", metadata_root=metadata_root)

    broken = index["BROKEN_DB.PUBLIC.BROKEN"]
    assert [column.name for column in broken.columns] == ["id", "name"]
    assert broken.columns[0].type == "TEXT"
    assert broken.columns[1].type is None
    assert broken.columns[0].sample_values == ["1"]
    assert broken.columns[1].sample_values == ["alpha"]
    assert broken.sample_rows == [{"id": "1", "name": "alpha"}]
    assert "ignored row" not in broken.searchable_text


def test_load_db_index_reads_cached_table_models(tmp_path):
    cache_path = tmp_path / "index.json"
    cache_payload = {
        "E_COMMERCE": {
            ORDERS: build_db_index("E_COMMERCE")[ORDERS],
        }
    }
    cache_path.write_text(
        json.dumps(
            {
                db: {table: schema.model_dump(mode="json") for table, schema in tables.items()}
                for db, tables in cache_payload.items()
            }
        ),
        encoding="utf-8",
    )

    index = load_db_index("E_COMMERCE", cache_path=cache_path)

    assert ORDERS in index
    assert index[ORDERS].name == "ORDERS"
    assert any(column.name == "order_id" for column in index[ORDERS].columns)


def test_retrieve_schema_ranks_customer_order_tables():
    selection = retrieve_schema(
        "Which customers placed the highest value orders?",
        "E_COMMERCE",
        retrieval_mode="lexical",
    )

    assert selection.db == "E_COMMERCE"
    assert CUSTOMERS in selection.selected_tables
    assert ORDERS in selection.selected_tables
    assert selection.expanded_tables == selection.selected_tables
    assert selection.confidence >= 0.5
    assert CUSTOMERS in selection.rationale
    assert ORDERS in selection.rationale


def test_retrieve_schema_does_not_expand_tables():
    selection = retrieve_schema(
        "Show product categories and seller names for the highest priced order items.",
        "E_COMMERCE",
        retrieval_mode="lexical",
        max_tables=2,
    )

    assert len(selection.selected_tables) == 2
    assert selection.expanded_tables == selection.selected_tables
    assert ORDER_ITEMS in selection.selected_tables
    assert PRODUCTS not in selection.expanded_tables
    assert SELLERS not in selection.expanded_tables


def test_retrieve_schema_stays_inside_the_requested_database():
    db_index = build_db_index("E_COMMERCE")
    selection = retrieve_schema(
        "Which zip code areas have the most customers and sellers?",
        "E_COMMERCE",
        retrieval_mode="lexical",
    )

    assert selection.selected_tables
    assert set(selection.expanded_tables).issubset(db_index)


def test_retrieve_schema_does_not_expand_tables_just_because_they_share_id(tmp_path):
    metadata_root = tmp_path / "databases"
    db_dir = metadata_root / "id_only_db"
    db_dir.mkdir(parents=True)

    (db_dir / "DDL.csv").write_text(
        "\n".join(
            [
                "table_name,DDL",
                'teams,"CREATE TABLE teams (id TEXT, team_name TEXT);"',
                'players,"CREATE TABLE players (id TEXT, player_name TEXT);"',
                'venues,"CREATE TABLE venues (id TEXT, venue_name TEXT);"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_table_metadata(
        db_dir,
        table_name="teams",
        ddl="CREATE TABLE teams (id TEXT, team_name TEXT);",
        column_names=["id", "team_name"],
        sample_rows=[{"id": "t1", "team_name": "Wolves"}],
    )
    _write_table_metadata(
        db_dir,
        table_name="players",
        ddl="CREATE TABLE players (id TEXT, player_name TEXT);",
        column_names=["id", "player_name"],
        sample_rows=[{"id": "p1", "player_name": "Jordan"}],
    )
    _write_table_metadata(
        db_dir,
        table_name="venues",
        ddl="CREATE TABLE venues (id TEXT, venue_name TEXT);",
        column_names=["id", "venue_name"],
        sample_rows=[{"id": "v1", "venue_name": "Arena"}],
    )

    cache_path = tmp_path / "index.json"
    build_index_cache(metadata_root=metadata_root, cache_path=cache_path)

    selection = retrieve_schema(
        "List team names.",
        "id_only_db",
        cache_path=cache_path,
        retrieval_mode="lexical",
        max_tables=1,
    )

    assert selection.selected_tables == ["teams"]
    assert selection.expanded_tables == ["teams"]


def test_retrieve_schema_ignores_tables_that_only_match_sample_text(tmp_path):
    metadata_root = tmp_path / "databases"
    db_dir = metadata_root / "sample_noise_db"
    db_dir.mkdir(parents=True)

    (db_dir / "DDL.csv").write_text(
        "\n".join(
            [
                "table_name,DDL",
                'ratings,"CREATE TABLE ratings (user_id TEXT, rating_score REAL);"',
                'erd,"CREATE TABLE erd (note TEXT, label TEXT);"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_table_metadata(
        db_dir,
        table_name="ratings",
        ddl="CREATE TABLE ratings (user_id TEXT, rating_score REAL);",
        column_names=["user_id", "rating_score"],
        sample_rows=[{"user_id": "u1", "rating_score": 4.5}],
    )
    _write_table_metadata(
        db_dir,
        table_name="erd",
        ddl="CREATE TABLE erd (note TEXT, label TEXT);",
        column_names=["note", "label"],
        sample_rows=[{"note": "rating", "label": "example"}],
    )

    cache_path = tmp_path / "index.json"
    build_index_cache(metadata_root=metadata_root, cache_path=cache_path)

    selection = retrieve_schema(
        "Show average rating by user.",
        "sample_noise_db",
        cache_path=cache_path,
        retrieval_mode="lexical",
        max_tables=2,
    )

    assert selection.selected_tables == ["ratings"]
    assert selection.expanded_tables == selection.selected_tables
