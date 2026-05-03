"""Tests for the read-only SQL validator."""

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


def test_validate_sql_rejects_unknown_table():
    report = validate_sql(
        "SELECT * FROM payments",
        allowed_tables=ALLOWED_TABLES,
    )

    assert report.ok is False
    assert report.errors == ["Unknown table referenced: payments."]


def test_validate_sql_does_not_skip_outer_table_shadowed_by_nested_cte():
    report = validate_sql(
        """
        SELECT *
        FROM customers
        WHERE EXISTS (
            WITH customers AS (SELECT 1)
            SELECT 1
        )
        """,
        allowed_tables=set(),
    )

    assert report.ok is False
    assert report.errors == ["Unknown table referenced: customers."]


def test_validate_sql_does_not_skip_qualified_base_table_with_same_cte_name():
    report = validate_sql(
        """
        WITH customers AS (
            SELECT customer_id
            FROM customers
        )
        SELECT *
        FROM E_COMMERCE.E_COMMERCE.CUSTOMERS
        """,
        allowed_tables=set(),
    )

    assert report.ok is False
    assert report.errors == [
        "Unknown table referenced: customers.",
        "Unknown table referenced: E_COMMERCE.E_COMMERCE.CUSTOMERS.",
    ]


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


def test_validate_sql_rejects_drop_table():
    report = validate_sql(
        "DROP TABLE customers",
        allowed_tables=ALLOWED_TABLES,
    )

    assert report.ok is False
    assert report.errors == ["Disallowed statement type: DROP."]


def test_validate_sql_rejects_session_commands():
    report = validate_sql(
        "USE DATABASE E_COMMERCE",
        allowed_tables=ALLOWED_TABLES,
    )

    assert report.ok is False
    assert report.errors == ["Disallowed statement type: USE."]


def test_validate_sql_rejects_unselected_qualified_table():
    report = validate_sql(
        "SELECT * FROM OTHER_DB.PUBLIC.CUSTOMERS",
        allowed_tables=SNOW_ALLOWED_TABLES,
    )

    assert report.ok is False
    assert report.errors == ["Unknown table referenced: OTHER_DB.PUBLIC.CUSTOMERS."]


def test_validate_sql_rejects_mutation_hidden_inside_cte():
    report = validate_sql(
        """
        WITH created AS (
            INSERT INTO customers VALUES ('c1')
            RETURNING customer_id
        )
        SELECT *
        FROM created
        """,
        allowed_tables=ALLOWED_TABLES,
    )

    assert report.ok is False
    assert report.errors == ["Disallowed statement type: INSERT."]


def test_validate_sql_rejects_extension_loading():
    report = validate_sql(
        "SELECT load_extension('unsafe')",
        allowed_tables=ALLOWED_TABLES,
    )

    assert report.ok is False
    assert report.errors == ["Extension loading is not allowed."]


def test_validate_sql_rejects_chained_statements():
    report = validate_sql(
        "SELECT 1; SELECT 2",
        allowed_tables=ALLOWED_TABLES,
    )

    assert report.ok is False
    assert report.errors == ["SQL must contain exactly one statement."]


def test_validate_sql_rejects_bare_mixed_case_snowflake_columns():
    report = validate_sql(
        f"SELECT COUNT(DISTINCT StudyInstanceUID) FROM {DICOM_PIVOT}",
        allowed_tables={DICOM_PIVOT},
        table_schemas={DICOM_PIVOT: DICOM_PIVOT_SCHEMA},
    )

    assert report.ok is False
    assert report.errors == [
        'Use "StudyInstanceUID" instead of StudyInstanceUID; '
        "Snowflake uppercases unquoted identifiers to STUDYINSTANCEUID."
    ]


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


def test_validate_sql_accepts_quoted_snowflake_columns():
    report = validate_sql(
        f"""
        SELECT COUNT(DISTINCT "StudyInstanceUID") AS unique_count
        FROM {DICOM_PIVOT}
        WHERE LOWER("SegmentedPropertyTypeCodeSequence") = '15825003'
          AND "collection_id" IN ('Community', 'nsclc_radiomics')
        """,
        allowed_tables={DICOM_PIVOT},
        table_schemas={DICOM_PIVOT: DICOM_PIVOT_SCHEMA},
    )

    assert report.ok is True
    assert report.errors == []
    assert report.warnings == []
