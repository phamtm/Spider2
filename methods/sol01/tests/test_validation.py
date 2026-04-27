"""Tests for the read-only SQL validator."""

from sol01.validation import validate_sql

ALLOWED_TABLES = {"customers", "orders", "order_items"}


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
        FROM main.customers
        """,
        allowed_tables=set(),
    )

    assert report.ok is False
    assert report.errors == ["Unknown table referenced: customers."]


def test_validate_sql_rejects_drop_table():
    report = validate_sql(
        "DROP TABLE customers",
        allowed_tables=ALLOWED_TABLES,
    )

    assert report.ok is False
    assert report.errors == ["Disallowed statement type: DROP."]


def test_validate_sql_rejects_pragma():
    report = validate_sql(
        "PRAGMA table_info(customers)",
        allowed_tables=ALLOWED_TABLES,
    )

    assert report.ok is False
    assert report.errors == ["Disallowed statement type: PRAGMA."]


def test_validate_sql_rejects_unsafe_database_qualifier():
    report = validate_sql(
        "SELECT * FROM other.customers",
        allowed_tables=ALLOWED_TABLES,
    )

    assert report.ok is False
    assert report.errors == ["Unsupported database qualifier on table 'other.customers'."]


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
