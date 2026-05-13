"""Tests for the read-only SQL validator."""

import pytest

from sol01.execution.validation import validate_sql
from sol01.models import ColumnSchema, TableSchema

ALLOWED_TABLES = {"customers", "orders", "order_items"}
SNOW_ALLOWED_TABLES = {
    "E_COMMERCE.E_COMMERCE.CUSTOMERS",
    "E_COMMERCE.E_COMMERCE.ORDERS",
    "E_COMMERCE.E_COMMERCE.ORDER_ITEMS",
}
DICOM_PIVOT = "IDC.IDC_V17.DICOM_PIVOT"
DICOM_PIVOT_SCHEMA = TableSchema(
    name="DICOM_PIVOT",
    database_name="IDC",
    schema_name="IDC_V17",
    full_name=DICOM_PIVOT,
    ddl='create or replace TABLE DICOM_PIVOT ("StudyInstanceUID" VARCHAR, '
    '"SegmentedPropertyTypeCodeSequence" VARCHAR, "collection_id" VARCHAR);',
    columns=[
        ColumnSchema(name="StudyInstanceUID", type="TEXT"),
        ColumnSchema(name="SegmentedPropertyTypeCodeSequence", type="TEXT"),
        ColumnSchema(name="collection_id", type="TEXT"),
    ],
    searchable_text="DICOM_PIVOT StudyInstanceUID SegmentedPropertyTypeCodeSequence collection_id",
)


def test_validate_sql_allows_valid_cte_query():
    report = validate_sql(
        """
        WITH recent_orders AS (
            SELECT order_id, customer_id
            FROM orders
        )
        SELECT customers.customer_id
        FROM recent_orders
        JOIN customers ON customers.customer_id = recent_orders.customer_id
        """,
        allowed_tables=ALLOWED_TABLES,
    )

    assert report.ok is True
    assert report.errors == []
    assert report.warnings == []
    assert set(report.referenced_tables) == {"orders", "customers"}


def test_validate_sql_allows_snowflake_fully_qualified_tables():
    report = validate_sql(
        """
        SELECT c.customer_id, o.order_id
        FROM "E_COMMERCE"."E_COMMERCE"."CUSTOMERS" AS c
        JOIN E_COMMERCE.E_COMMERCE.ORDERS AS o
            ON c.customer_id = o.customer_id
        """,
        allowed_tables=SNOW_ALLOWED_TABLES,
    )

    assert report.ok is True
    assert report.errors == []
    assert set(report.referenced_tables) == {
        "E_COMMERCE.E_COMMERCE.CUSTOMERS",
        "E_COMMERCE.E_COMMERCE.ORDERS",
    }


def test_validate_sql_allows_unique_short_reference_to_selected_snowflake_table():
    report = validate_sql(
        "SELECT customer_id FROM CUSTOMERS",
        allowed_tables=SNOW_ALLOWED_TABLES,
    )

    assert report.ok is True
    assert report.referenced_tables == ["E_COMMERCE.E_COMMERCE.CUSTOMERS"]


@pytest.mark.parametrize(
    "sql, allowed_tables, expected_errors",
    [
        pytest.param(
            "SELECT * FROM payments",
            ALLOWED_TABLES,
            ["Unknown table referenced: payments."],
            id="unknown-table",
        ),
        pytest.param(
            """
            SELECT *
            FROM customers
            WHERE EXISTS (
                WITH customers AS (SELECT 1)
                SELECT 1
            )
            """,
            set(),
            ["Unknown table referenced: customers."],
            id="nested-cte-shadows-outer-table",
        ),
        pytest.param(
            """
            WITH customers AS (
                SELECT customer_id
                FROM customers
            )
            SELECT *
            FROM E_COMMERCE.E_COMMERCE.CUSTOMERS
            """,
            set(),
            [
                "Unknown table referenced: customers.",
                "Unknown table referenced: E_COMMERCE.E_COMMERCE.CUSTOMERS.",
            ],
            id="qualified-base-table-with-same-cte-name",
        ),
        pytest.param(
            "SELECT * FROM OTHER_DB.PUBLIC.CUSTOMERS",
            SNOW_ALLOWED_TABLES,
            ["Unknown table referenced: OTHER_DB.PUBLIC.CUSTOMERS."],
            id="unselected-qualified-table",
        ),
    ],
)
def test_validate_sql_rejects_unknown_or_unselected_tables(sql, allowed_tables, expected_errors):
    report = validate_sql(sql, allowed_tables=allowed_tables)
    assert report.ok is False
    assert report.errors == expected_errors


@pytest.mark.parametrize(
    "sql, expected_errors",
    [
        pytest.param(
            "DROP TABLE customers",
            ["Disallowed statement type: DROP."],
            id="drop-table",
        ),
        pytest.param(
            "USE DATABASE E_COMMERCE",
            ["Disallowed statement type: USE."],
            id="session-command",
        ),
        pytest.param(
            """
            WITH created AS (
                INSERT INTO customers VALUES ('c1')
                RETURNING customer_id
            )
            SELECT *
            FROM created
            """,
            ["Disallowed statement type: INSERT."],
            id="mutation-in-cte",
        ),
        pytest.param(
            "SELECT load_extension('unsafe')",
            ["Extension loading is not allowed."],
            id="extension-loading",
        ),
        pytest.param(
            "SELECT 1; SELECT 2",
            ["SQL must contain exactly one statement."],
            id="chained-statements",
        ),
    ],
)
def test_validate_sql_rejects_disallowed_patterns(sql, expected_errors):
    report = validate_sql(sql, allowed_tables=ALLOWED_TABLES)
    assert report.ok is False
    assert report.errors == expected_errors


def test_validate_sql_rejects_bare_snowflake_columns_and_accepts_quoted():
    bare_report = validate_sql(
        f"SELECT COUNT(DISTINCT StudyInstanceUID) FROM {DICOM_PIVOT}",
        allowed_tables={DICOM_PIVOT},
        table_schemas={DICOM_PIVOT: DICOM_PIVOT_SCHEMA},
    )
    assert bare_report.ok is False
    assert bare_report.errors == [
        'Use "StudyInstanceUID" instead of StudyInstanceUID; '
        "Snowflake uppercases unquoted identifiers to STUDYINSTANCEUID."
    ]

    quoted_report = validate_sql(
        f"""
        SELECT COUNT(DISTINCT "StudyInstanceUID") AS unique_count
        FROM {DICOM_PIVOT}
        WHERE LOWER("SegmentedPropertyTypeCodeSequence") = '15825003'
          AND "collection_id" IN ('Community', 'nsclc_radiomics')
        """,
        allowed_tables={DICOM_PIVOT},
        table_schemas={DICOM_PIVOT: DICOM_PIVOT_SCHEMA},
    )
    assert quoted_report.ok is True
    assert quoted_report.errors == []
    assert quoted_report.warnings == []


def test_validate_sql_rejects_bare_lower_case_snowflake_columns():
    report = validate_sql(
        f"""
        SELECT COUNT(DISTINCT "StudyInstanceUID")
        FROM {DICOM_PIVOT}
        WHERE LOWER(SegmentedPropertyTypeCodeSequence) = '15825003'
          AND collection_id IN ('Community', 'nsclc_radiomics')
        """,
        allowed_tables={DICOM_PIVOT},
        table_schemas={DICOM_PIVOT: DICOM_PIVOT_SCHEMA},
    )

    assert report.ok is False
    assert report.errors == [
        'Use "SegmentedPropertyTypeCodeSequence" instead of '
        "SegmentedPropertyTypeCodeSequence; Snowflake uppercases unquoted identifiers to "
        "SEGMENTEDPROPERTYTYPECODESEQUENCE.",
        'Use "collection_id" instead of collection_id; '
        "Snowflake uppercases unquoted identifiers to COLLECTION_ID.",
    ]


def test_validate_sql_reports_scope_resolution_errors():
    report = validate_sql(
        """
        SELECT bt.board_type, AVG(sc.score) AS avg_score
        FROM FIREBASE.ANALYTICS_153293282.EVENTS_20180915 e,
        LATERAL (
            SELECT value:string_value::string AS board_type
            FROM FLATTEN(input => e.event_params)
            WHERE key = 'board_type'
        ) bt,
        LATERAL (
            SELECT COALESCE(value:int_value, value:double_value, value:float_value)::float AS score
            FROM FLATTEN(input => e.event_params)
            WHERE key = 'score'
        ) sc
        WHERE e.event_name = 'level_complete_quickplay'
        GROUP BY bt.board_type
        ORDER BY avg_score DESC
        """,
        allowed_tables={"FIREBASE.ANALYTICS_153293282.EVENTS_20180915"},
    )

    assert report.ok is False
    assert report.errors == ["SQL scope could not be resolved: Alias already used."]
