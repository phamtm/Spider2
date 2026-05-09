"""Tests for canonical schema object construction."""

from __future__ import annotations

from sol01.models import ColumnSchema, TableSchema
from sol01.schema.objects import build_schema_objects

ORDERS = "DB.PUBLIC.ORDERS"
CUSTOMERS = "DB.PUBLIC.CUSTOMERS"
EVENTS = "DB.PUBLIC.EVENTS"


def _table(
    name: str,
    columns: list[ColumnSchema],
    *,
    schema_name: str = "PUBLIC",
    searchable_text: str | None = None,
) -> TableSchema:
    """Create one compact synthetic table schema."""

    full_name = f"DB.{schema_name}.{name}"
    return TableSchema(
        name=name,
        database_name="DB",
        schema_name=schema_name,
        full_name=full_name,
        ddl="",
        columns=columns,
        searchable_text=searchable_text or name.lower(),
    )


def _objects_by_type(index: dict[str, TableSchema], object_type: str):
    """Return schema objects of one type from a synthetic index."""

    return [obj for obj in build_schema_objects(index) if obj.object_type == object_type]


def test_builds_table_and_column_objects_with_stable_ids_and_metadata():
    index = {
        ORDERS: TableSchema(
            name="ORDERS",
            database_name="DB",
            schema_name="PUBLIC",
            full_name=ORDERS,
            ddl="CREATE TABLE ORDERS (ORDER_ID TEXT, STATUS TEXT);",
            columns=[
                ColumnSchema(name="ORDER_ID", type="TEXT"),
                ColumnSchema(name="STATUS", type="TEXT", description="Order state"),
            ],
            sample_rows=[{"ORDER_ID": "o1", "STATUS": "shipped"}],
            searchable_text="orders status",
        )
    }

    first = build_schema_objects(index)
    second = build_schema_objects(index)
    object_ids = [obj.object_id for obj in first]

    assert object_ids == [obj.object_id for obj in second]
    assert "table:DB.PUBLIC.ORDERS" in object_ids
    assert "column:DB.PUBLIC.ORDERS#STATUS" in object_ids

    table = next(obj for obj in first if obj.object_id == "table:DB.PUBLIC.ORDERS")
    assert table.metadata["source"] == "load_db_index"
    assert table.metadata["full_name"] == ORDERS
    assert table.metadata["family_hooks"]["normalized_table_name"] == "orders"

    status = next(obj for obj in first if obj.object_id == "column:DB.PUBLIC.ORDERS#STATUS")
    assert status.description == "Order state"
    assert status.metadata["column_type"] == "TEXT"
    assert status.metadata["primitive_type"] == "string"


def test_column_groups_cover_key_time_numeric_measure_and_repeated_prefixes():
    index = {
        EVENTS: TableSchema(
            name="EVENTS",
            database_name="DB",
            schema_name="PUBLIC",
            full_name=EVENTS,
            ddl="",
            columns=[
                ColumnSchema(name="EVENT_ID", type="TEXT"),
                ColumnSchema(name="USER_ID", type="TEXT"),
                ColumnSchema(name="CREATED_AT", type="TIMESTAMP"),
                ColumnSchema(name="TOTAL_AMOUNT", type="NUMBER"),
                ColumnSchema(name="DISCOUNT_RATE", type="FLOAT"),
                ColumnSchema(name="BILLING_CITY", type="TEXT"),
                ColumnSchema(name="BILLING_STATE", type="TEXT"),
                ColumnSchema(name="BILLING_COUNTRY", type="TEXT"),
            ],
            searchable_text="events",
        )
    }

    groups = _objects_by_type(index, "column_group")
    group_by_type = {obj.metadata["group_type"]: obj for obj in groups}

    assert group_by_type["key_like"].metadata["columns"] == ["EVENT_ID", "USER_ID"]
    assert group_by_type["time_like"].metadata["columns"] == ["CREATED_AT"]
    assert group_by_type["numeric_measures"].metadata["columns"] == [
        "DISCOUNT_RATE",
        "TOTAL_AMOUNT",
    ]
    assert group_by_type["prefix_billing"].metadata["columns"] == [
        "BILLING_CITY",
        "BILLING_COUNTRY",
        "BILLING_STATE",
    ]


def test_join_candidates_are_inferred_evidence_not_foreign_keys():
    index = {
        ORDERS: TableSchema(
            name="ORDERS",
            database_name="DB",
            schema_name="PUBLIC",
            full_name=ORDERS,
            ddl="",
            columns=[ColumnSchema(name="CUSTOMER_ID", type="TEXT")],
            searchable_text="orders",
        ),
        CUSTOMERS: TableSchema(
            name="CUSTOMERS",
            database_name="DB",
            schema_name="PUBLIC",
            full_name=CUSTOMERS,
            ddl="",
            columns=[ColumnSchema(name="CUSTOMER_ID", type="VARCHAR")],
            searchable_text="customers",
        ),
    }

    join_candidates = _objects_by_type(index, "join_candidate")

    assert len(join_candidates) == 1
    candidate = join_candidates[0]
    assert candidate.object_id.startswith(
        "join_candidate:DB.PUBLIC.CUSTOMERS#CUSTOMER_ID->DB.PUBLIC.ORDERS#CUSTOMER_ID:"
    )
    assert candidate.metadata["is_inferred_evidence"] is True
    assert candidate.metadata["is_foreign_key"] is False
    assert candidate.metadata["evidence_type"] == "same_normalized_column_name_and_compatible_type"


def test_sample_values_include_only_bounded_categorical_values():
    index = {
        ORDERS: TableSchema(
            name="ORDERS",
            database_name="DB",
            schema_name="PUBLIC",
            full_name=ORDERS,
            ddl="",
            columns=[
                ColumnSchema(name="STATUS", type="TEXT"),
                ColumnSchema(name="ORDER_ID", type="TEXT"),
                ColumnSchema(name="BODY_TEXT", type="TEXT"),
                ColumnSchema(name="CREATED_AT", type="TIMESTAMP"),
                ColumnSchema(name="PAYLOAD", type="VARIANT"),
                ColumnSchema(name="AMOUNT", type="NUMBER"),
                ColumnSchema(name="CUSTOMER_EMAIL", type="TEXT"),
                ColumnSchema(name="SESSION_HASH", type="TEXT"),
                ColumnSchema(name="DISPLAY_NAME", type="TEXT"),
                ColumnSchema(name="DESCRIPTION", type="TEXT"),
                ColumnSchema(name="UNBOUNDED_LABEL", type="TEXT"),
            ],
            sample_rows=[
                {
                    "STATUS": "open",
                    "ORDER_ID": "550e8400-e29b-41d4-a716-446655440000",
                    "BODY_TEXT": "A long free-form support note that should not be indexed.",
                    "CREATED_AT": "2026-05-09 12:00:00",
                    "PAYLOAD": {"kind": "event"},
                    "AMOUNT": 19.99,
                    "CUSTOMER_EMAIL": "buyer@example.com",
                    "SESSION_HASH": "0123456789abcdef0123456789abcdef",
                    "DISPLAY_NAME": "Alice Smith",
                    "DESCRIPTION": "Useful free-form text",
                    "UNBOUNDED_LABEL": "alpha",
                },
                {
                    "STATUS": "closed",
                    "ORDER_ID": "550e8400-e29b-41d4-a716-446655440001",
                    "BODY_TEXT": "Another long free-form support note that should not be indexed.",
                    "CREATED_AT": "2026-05-09 13:00:00",
                    "PAYLOAD": {"kind": "event"},
                    "AMOUNT": 20.99,
                    "CUSTOMER_EMAIL": "seller@example.com",
                    "SESSION_HASH": "abcdef0123456789abcdef0123456789",
                    "DISPLAY_NAME": "Bob Smith",
                    "DESCRIPTION": "More free-form text",
                    "UNBOUNDED_LABEL": "beta",
                },
                {
                    "STATUS": "open",
                    "ORDER_ID": "550e8400-e29b-41d4-a716-446655440002",
                    "BODY_TEXT": "More text",
                    "CREATED_AT": "2026-05-09 14:00:00",
                    "PAYLOAD": {"kind": "event"},
                    "AMOUNT": 21.99,
                    "CUSTOMER_EMAIL": "support@example.com",
                    "SESSION_HASH": "fedcba9876543210fedcba9876543210",
                    "DISPLAY_NAME": "Carol Smith",
                    "DESCRIPTION": "Even more free-form text",
                    "UNBOUNDED_LABEL": "gamma",
                },
            ],
            searchable_text="orders",
        )
    }

    sample_values = _objects_by_type(index, "sample_value")
    values_by_column = {(obj.column_name, obj.metadata["value"]) for obj in sample_values}

    assert values_by_column == {("STATUS", "closed"), ("STATUS", "open")}
    for sample_value in sample_values:
        assert sample_value.metadata["sample_size"] == 3
        assert sample_value.metadata["distinct_count"] == 2
        assert sample_value.metadata["dense_embedding_default"] is False
        assert sample_value.metadata["inclusion_reason"] == "repeated_bounded_sample"


def test_exact_table_families_use_ordered_column_signature_and_stable_canonical_member():
    columns = [
        ColumnSchema(name="ORDER_ID", type="TEXT"),
        ColumnSchema(name="AMOUNT", type="NUMBER"),
        ColumnSchema(name="ORDER_DATE", type="DATE"),
    ]
    index = {
        "DB.PUBLIC.SALES_2024": _table("SALES_2024", columns),
        "DB.PUBLIC.SALES_2023": _table("SALES_2023", columns),
        "DB.PUBLIC.SALES_2022": _table("SALES_2022", columns),
    }

    first = _objects_by_type(index, "family")
    second = _objects_by_type(index, "family")

    assert [obj.object_id for obj in first] == [obj.object_id for obj in second]
    assert len(first) == 1
    family = first[0]
    assert family.metadata["schema_object_subtype"] == "table_family"
    assert family.metadata["family_kind"] == "exact"
    assert family.metadata["canonical_member"] == "DB.PUBLIC.SALES_2022"
    assert family.metadata["member_table_refs"] == [
        "DB.PUBLIC.SALES_2022",
        "DB.PUBLIC.SALES_2023",
        "DB.PUBLIC.SALES_2024",
    ]
    assert family.metadata["common_columns"] == ["ORDER_ID", "AMOUNT", "ORDER_DATE"]
    assert family.metadata["variant_columns"] == {
        "columns": [],
        "total_count": 0,
        "truncated": False,
    }
    assert family.metadata["suffix_dimensions"] == [
        {
            "kind": "YYYY",
            "raw_values": ["2022", "2023", "2024"],
            "values": ["2022", "2023", "2024"],
        }
    ]


def test_near_table_families_require_same_stem_and_column_jaccard_threshold():
    index = {
        "DB.PUBLIC.EVENTS_1": _table(
            "EVENTS_1",
            [
                ColumnSchema(name="EVENT_ID", type="TEXT"),
                ColumnSchema(name="USER_ID", type="TEXT"),
                ColumnSchema(name="CREATED_AT", type="TIMESTAMP"),
                ColumnSchema(name="EVENT_TYPE", type="TEXT"),
            ],
        ),
        "DB.PUBLIC.EVENTS_2": _table(
            "EVENTS_2",
            [
                ColumnSchema(name="EVENT_ID", type="TEXT"),
                ColumnSchema(name="USER_ID", type="TEXT"),
                ColumnSchema(name="CREATED_AT", type="TIMESTAMP"),
                ColumnSchema(name="EVENT_SOURCE", type="TEXT"),
            ],
        ),
        "DB.PUBLIC.EVENTS_3": _table(
            "EVENTS_3",
            [
                ColumnSchema(name="EVENT_ID", type="TEXT"),
                ColumnSchema(name="USER_ID", type="TEXT"),
                ColumnSchema(name="CREATED_AT", type="TIMESTAMP"),
                ColumnSchema(name="SESSION_ID", type="TEXT"),
            ],
        ),
    }

    families = [
        obj
        for obj in build_schema_objects(index, family_similarity_threshold=0.6)
        if obj.object_type == "family"
    ]

    assert len(families) == 1
    family = families[0]
    assert family.metadata["family_kind"] == "near"
    assert family.metadata["common_columns"] == ["EVENT_ID", "USER_ID", "CREATED_AT"]
    assert family.metadata["variant_columns"]["total_count"] == 3
    assert family.metadata["suffix_dimensions"] == [
        {"kind": "integer", "raw_values": ["1", "2", "3"], "values": ["1", "2", "3"]}
    ]
    assert family.metadata["caveats"]


def test_partition_suffix_dimensions_cover_supported_generic_suffix_shapes():
    columns = [ColumnSchema(name="ID", type="TEXT")]
    index = {
        "DB.PUBLIC.DAILY_20240101": _table("DAILY_20240101", columns),
        "DB.PUBLIC.DAILY_20240102": _table("DAILY_20240102", columns),
        "DB.PUBLIC.SNAPSHOT_2024_01_01": _table("SNAPSHOT_2024_01_01", columns),
        "DB.PUBLIC.SNAPSHOT_2024_01_02": _table("SNAPSHOT_2024_01_02", columns),
        "DB.PUBLIC.MODEL_v1": _table("MODEL_v1", columns),
        "DB.PUBLIC.MODEL_v2": _table("MODEL_v2", columns),
    }

    families = {
        family.metadata["normalized_stem"]: family.metadata["suffix_dimensions"]
        for family in _objects_by_type(index, "family")
    }

    assert families["daily"] == [
        {
            "kind": "YYYYMMDD",
            "raw_values": ["20240101", "20240102"],
            "values": ["2024-01-01", "2024-01-02"],
        }
    ]
    assert families["snapshot"] == [
        {
            "kind": "YYYY_MM_DD",
            "raw_values": ["2024_01_01", "2024_01_02"],
            "values": ["2024-01-01", "2024-01-02"],
        }
    ]
    assert families["model"] == [
        {"kind": "version", "raw_values": ["_v1", "_v2"], "values": ["1", "2"]}
    ]


def test_table_family_detection_rejects_unrelated_tables_and_wide_single_tables():
    index = {
        "DB.PUBLIC.CUSTOMERS": _table(
            "CUSTOMERS",
            [ColumnSchema(name="ID", type="TEXT"), ColumnSchema(name="NAME", type="TEXT")],
        ),
        "DB.PUBLIC.VENDORS": _table(
            "VENDORS",
            [ColumnSchema(name="ID", type="TEXT"), ColumnSchema(name="NAME", type="TEXT")],
        ),
        "DB.PUBLIC.WIDE_METRICS": _table(
            "WIDE_METRICS",
            [
                ColumnSchema(name="METRIC_1", type="NUMBER"),
                ColumnSchema(name="METRIC_2", type="NUMBER"),
                ColumnSchema(name="METRIC_3", type="NUMBER"),
                ColumnSchema(name="METRIC_4", type="NUMBER"),
            ],
        ),
    }

    assert _objects_by_type(index, "family") == []
