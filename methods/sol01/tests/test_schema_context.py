"""Tests for deterministic schema context availability."""

from __future__ import annotations

from pathlib import Path

from sol01.infra.config import SchemaContextConfig
from sol01.models import ColumnSchema, SchemaContextChunk, SchemaObject, TableSchema
from sol01.schema.chunks import render_schema_chunks
from sol01.schema.objects import build_schema_objects
from sol01.schema.schema_context import (
    build_available_schema_context,
    build_schema_context_inputs,
    clip_linked_docs,
)
from sol01.schema.schema_context_cache import SchemaContextCache


def test_query_construction_extracts_signals_and_clips_linked_docs_by_overlap():
    query = build_schema_context_inputs(
        "Show DB.PUBLIC.ORDERS where STATUS = 'closed' in FY_2024 and 2024-01-03",
        linked_docs=[
            "Irrelevant billing policy paragraph.\n\n"
            "Orders status values include closed and open for fulfilled orders."
        ],
        exact_literals=["VIP_CUSTOMER"],
        max_doc_chars=80,
    )

    assert query.exact_literals == ("closed", "VIP_CUSTOMER")
    assert query.dates == ("2024-01-03",)
    assert "2024" in query.years
    assert "DB.PUBLIC.ORDERS" in query.identifiers
    assert "FY_2024" in query.uppercase_codes
    assert "vip_customer" in query.normalized_tokens
    assert "vip" in query.normalized_tokens
    assert query.linked_doc_context.startswith("Orders status values")
    assert len(query.linked_doc_context) <= 80

    clipped = clip_linked_docs(
        ["alpha beta.\n\ntarget revenue status paragraph.\n\ntrailing text."],
        query_terms=["status"],
        max_doc_chars=40,
    )

    assert clipped == "target revenue status paragraph."


def test_full_database_metadata_context_returns_all_schema_objects_without_ranking():
    index = _fake_index()

    objects, diagnostics = build_available_schema_context(
        index,
        "Find ORDERS where status is 'closed'",
        config=SchemaContextConfig(object_cutoff=2),
    )

    selected_ids = [obj.schema_object.object_id for obj in objects]
    assert selected_ids == [
        "family:DB.PUBLIC:orders_family:deadbeef",
        "table:DB.PUBLIC.CUSTOMERS",
        "table:DB.PUBLIC.ORDERS",
        "column:DB.PUBLIC.ORDERS#AMOUNT",
        "column:DB.PUBLIC.ORDERS#STATUS",
        "join_candidate:DB.PUBLIC.ORDERS#CUSTOMER_ID->DB.PUBLIC.CUSTOMERS#CUSTOMER_ID:abcdef12",
        "sample_value:DB.PUBLIC.ORDERS#STATUS:11111111",
    ]
    assert diagnostics["context_mode"] == "full_database_metadata"
    assert diagnostics["context_counts"] == {
        "objects_total": 7,
        "chunks_total": 7,
        "available_objects": 7,
    }
    assert all(obj.score is None for obj in objects)


def test_object_cutoff_does_not_limit_available_schema_context():
    index = _fake_index()

    objects, _ = build_available_schema_context(
        index,
        "orders status customer amount",
        config=SchemaContextConfig(object_cutoff=1),
    )

    assert len(objects) == len(index.objects)


def test_summary_backed_context_uses_only_curated_large_schema_objects():
    table = TableSchema(
        name="_20240103",
        database_name="GITHUB_REPOS_DATE",
        schema_name="DAY",
        full_name="GITHUB_REPOS_DATE.DAY._20240103",
        ddl="CREATE TABLE _20240103 (SECRET_DDL_MARKER TEXT);",
        columns=[
            ColumnSchema(name="public", type="BOOLEAN"),
            ColumnSchema(name="actor", type="VARIANT"),
            ColumnSchema(name="created_at", type="TIMESTAMP"),
            ColumnSchema(name="type", type="TEXT"),
            ColumnSchema(name="repo", type="VARIANT"),
            ColumnSchema(name="payload", type="VARIANT"),
            ColumnSchema(name="id", type="TEXT"),
            ColumnSchema(name="other", type="VARIANT"),
            ColumnSchema(name="org", type="VARIANT"),
        ],
        sample_rows=[],
        searchable_text="github events",
    )
    uncovered_table = TableSchema(
        name="REPOSITORIES",
        database_name="GITHUB_REPOS_DATE",
        schema_name="DAY",
        full_name="GITHUB_REPOS_DATE.DAY.REPOSITORIES",
        ddl="CREATE TABLE REPOSITORIES (ID TEXT, NAME TEXT);",
        columns=[ColumnSchema(name="ID", type="TEXT"), ColumnSchema(name="NAME", type="TEXT")],
        sample_rows=[],
        searchable_text="repository metadata",
    )
    objects = build_schema_objects(
        {
            "GITHUB_REPOS_DATE.DAY._20240103": table,
            "GITHUB_REPOS_DATE.DAY.REPOSITORIES": uncovered_table,
        }
    )
    chunks = render_schema_chunks(objects)
    index = SchemaContextCache(
        db="GITHUB_REPOS_DATE",
        cache_key="summary-rendered",
        cache_dir=Path("/tmp/test-schema-context-summary-rendered"),
        manifest={},
        objects=objects,
        chunks=chunks,
    )

    schema_context_objects, diagnostics = build_available_schema_context(
        index,
        "Count daily github archive repository events",
        config=SchemaContextConfig(object_cutoff=1),
    )

    assert diagnostics["context_mode"] == "large_schema_summary"
    schema_context_ids = [item.schema_object.object_id for item in schema_context_objects]
    assert "table:GITHUB_REPOS_DATE.DAY._20240103" in schema_context_ids
    assert "column:GITHUB_REPOS_DATE.DAY._20240103#payload" not in schema_context_ids
    assert "table:GITHUB_REPOS_DATE.DAY.REPOSITORIES" in schema_context_ids
    assert "column:GITHUB_REPOS_DATE.DAY.REPOSITORIES#NAME" in schema_context_ids
    summary_object = next(
        item
        for item in schema_context_objects
        if item.schema_object.object_id == "table:GITHUB_REPOS_DATE.DAY._20240103"
    )
    assert "Large-schema summary: github_repos_day_events." in summary_object.planning_text
    assert "SECRET_DDL_MARKER" not in summary_object.planning_text


def _fake_index() -> SchemaContextCache:
    objects = [
        SchemaObject(
            object_id="table:DB.PUBLIC.ORDERS",
            object_type="table",
            name="ORDERS",
            db="DB",
            table_name="DB.PUBLIC.ORDERS",
            searchable_text="orders status customer",
        ),
        SchemaObject(
            object_id="table:DB.PUBLIC.CUSTOMERS",
            object_type="table",
            name="CUSTOMERS",
            db="DB",
            table_name="DB.PUBLIC.CUSTOMERS",
            searchable_text="customers",
        ),
        SchemaObject(
            object_id="column:DB.PUBLIC.ORDERS#STATUS",
            object_type="column",
            name="STATUS",
            db="DB",
            table_name="DB.PUBLIC.ORDERS",
            column_name="STATUS",
            searchable_text="status closed open",
        ),
        SchemaObject(
            object_id="column:DB.PUBLIC.ORDERS#AMOUNT",
            object_type="column",
            name="AMOUNT",
            db="DB",
            table_name="DB.PUBLIC.ORDERS",
            column_name="AMOUNT",
            searchable_text="amount revenue",
        ),
        SchemaObject(
            object_id="sample_value:DB.PUBLIC.ORDERS#STATUS:11111111",
            object_type="sample_value",
            name="closed",
            db="DB",
            table_name="DB.PUBLIC.ORDERS",
            column_name="STATUS",
            searchable_text="closed",
        ),
        SchemaObject(
            object_id="join_candidate:DB.PUBLIC.ORDERS#CUSTOMER_ID->DB.PUBLIC.CUSTOMERS#CUSTOMER_ID:abcdef12",
            object_type="join_candidate",
            name="ORDERS_CUSTOMERS",
            db="DB",
            searchable_text="orders customers customer_id",
        ),
        SchemaObject(
            object_id="family:DB.PUBLIC:orders_family:deadbeef",
            object_type="family",
            name="orders_family",
            db="DB",
            searchable_text="orders table family",
            metadata={"member_table_refs": ["DB.PUBLIC.ORDERS"]},
        ),
    ]
    chunks = [
        SchemaContextChunk(
            chunk_id="table:DB.PUBLIC.ORDERS::table",
            object_id="table:DB.PUBLIC.ORDERS",
            chunk_type="table",
            prompt_text="Orders table with status and customer fields.",
        ),
        SchemaContextChunk(
            chunk_id="table:DB.PUBLIC.CUSTOMERS::table",
            object_id="table:DB.PUBLIC.CUSTOMERS",
            chunk_type="table",
            prompt_text="Customers table.",
        ),
        SchemaContextChunk(
            chunk_id="column:DB.PUBLIC.ORDERS#STATUS::column",
            object_id="column:DB.PUBLIC.ORDERS#STATUS",
            chunk_type="column",
            parent_object_ids=["table:DB.PUBLIC.ORDERS"],
            prompt_text="Order status column.",
        ),
        SchemaContextChunk(
            chunk_id="column:DB.PUBLIC.ORDERS#AMOUNT::column",
            object_id="column:DB.PUBLIC.ORDERS#AMOUNT",
            chunk_type="column",
            parent_object_ids=["table:DB.PUBLIC.ORDERS"],
            prompt_text="Order amount column.",
        ),
        SchemaContextChunk(
            chunk_id="sample_value:DB.PUBLIC.ORDERS#STATUS:11111111::sample_value",
            object_id="sample_value:DB.PUBLIC.ORDERS#STATUS:11111111",
            chunk_type="sample_value",
            parent_object_ids=[
                "table:DB.PUBLIC.ORDERS",
                "column:DB.PUBLIC.ORDERS#STATUS",
            ],
            prompt_text="Sample value closed for order status.",
        ),
        SchemaContextChunk(
            chunk_id="join_candidate:DB.PUBLIC.ORDERS#CUSTOMER_ID->DB.PUBLIC.CUSTOMERS#CUSTOMER_ID:abcdef12::join_candidate",
            object_id="join_candidate:DB.PUBLIC.ORDERS#CUSTOMER_ID->DB.PUBLIC.CUSTOMERS#CUSTOMER_ID:abcdef12",
            chunk_type="join_candidate",
            parent_object_ids=[
                "table:DB.PUBLIC.ORDERS",
                "column:DB.PUBLIC.ORDERS#CUSTOMER_ID",
                "table:DB.PUBLIC.CUSTOMERS",
                "column:DB.PUBLIC.CUSTOMERS#CUSTOMER_ID",
            ],
            prompt_text="Join orders to customers on customer id.",
        ),
        SchemaContextChunk(
            chunk_id="family:DB.PUBLIC:orders_family:deadbeef::table_family",
            object_id="family:DB.PUBLIC:orders_family:deadbeef",
            chunk_type="table_family",
            parent_object_ids=["table:DB.PUBLIC.ORDERS"],
            prompt_text="Orders table family.",
        ),
    ]
    return SchemaContextCache(
        db="DB",
        cache_key="test",
        cache_dir=Path("/tmp/test-schema-index"),
        manifest={},
        objects=objects,
        chunks=chunks,
    )
