"""Tests for exact selected-table grounding and sanitization."""

from __future__ import annotations

from sol01.models import (
    ColumnSchema,
    Intent,
    SchemaGrounding,
    SchemaGroundingBinding,
    TableSchema,
    UnresolvedSchemaTerm,
)
from sol01.schema.schema_grounding import grounding_targets, sanitize_schema_grounding


def test_grounding_targets_collect_distinct_schema_facing_terms():
    intent = Intent(
        summary="Compute revenue by customer.",
        entities=["customers"],
        metrics=["revenue", "revenue"],
        filters=["closed orders"],
        native_value_terms=["orders.status=closed"],
        requested_ordering=["highest revenue first"],
        output_expectation="customer and revenue columns",
    )

    targets = grounding_targets(intent)

    assert targets == [
        {"requested_term": "revenue", "binding_kind": "metric", "essential": True},
        {"requested_term": "closed orders", "binding_kind": "filter", "essential": True},
        {
            "requested_term": "orders.status=closed",
            "binding_kind": "native_term",
            "essential": True,
        },
        {"requested_term": "customers", "binding_kind": "entity", "essential": False},
        {
            "requested_term": "highest revenue first",
            "binding_kind": "order",
            "essential": False,
        },
    ]


def test_sanitize_schema_grounding_drops_invented_bindings_and_marks_term_unresolved():
    table_name = "DB.PUBLIC.ORDERS"
    table_schemas = {
        table_name: TableSchema(
            name="ORDERS",
            database_name="DB",
            schema_name="PUBLIC",
            full_name=table_name,
            ddl="",
            columns=[ColumnSchema(name="TOTAL", type="NUMBER")],
            searchable_text="orders total",
        )
    }
    requested_terms = [
        {"requested_term": "revenue", "binding_kind": "metric", "essential": True},
        {"requested_term": "status", "binding_kind": "filter", "essential": True},
    ]

    grounding, diagnostics = sanitize_schema_grounding(
        SchemaGrounding(
            bindings=[
                SchemaGroundingBinding(
                    requested_term="revenue",
                    binding_kind="metric",
                    table_name=table_name,
                    column_name="TOTAL",
                ),
                SchemaGroundingBinding(
                    requested_term="status",
                    binding_kind="filter",
                    table_name=table_name,
                    column_name="STATUS",
                ),
            ],
            unresolved_terms=[
                UnresolvedSchemaTerm(
                    requested_term="status",
                    binding_kind="filter",
                    reason="no exact selected-table column matched",
                )
            ],
        ),
        available_tables=[table_name],
        table_schemas=table_schemas,
        requested_terms=requested_terms,
    )

    assert [(binding.requested_term, binding.column_name) for binding in grounding.bindings] == [
        ("revenue", "TOTAL"),
    ]
    assert grounding.unresolved_terms == [
        UnresolvedSchemaTerm(
            requested_term="status",
            binding_kind="filter",
            reason="no exact selected-table column matched",
            essential=True,
        )
    ]
    assert diagnostics["invalid_bindings"] == [f"{table_name}.STATUS"]
    assert grounding.warnings == [f"Dropped invented schema bindings: {table_name}.STATUS"]


def test_sanitize_schema_grounding_marks_missing_requested_terms_unresolved():
    table_name = "DB.PUBLIC.ORDERS"
    table_schemas = {
        table_name: TableSchema(
            name="ORDERS",
            database_name="DB",
            schema_name="PUBLIC",
            full_name=table_name,
            ddl="",
            columns=[ColumnSchema(name="TOTAL", type="NUMBER")],
            searchable_text="orders total",
        )
    }

    grounding, diagnostics = sanitize_schema_grounding(
        SchemaGrounding(),
        available_tables=[table_name],
        table_schemas=table_schemas,
        requested_terms=[
            {"requested_term": "revenue", "binding_kind": "metric", "essential": True}
        ],
    )

    assert grounding.bindings == []
    assert grounding.unresolved_terms == [
        UnresolvedSchemaTerm(
            requested_term="revenue",
            binding_kind="metric",
            reason="model did not account for requested term",
            essential=True,
        )
    ]
    assert diagnostics["unresolved_count"] == 1
