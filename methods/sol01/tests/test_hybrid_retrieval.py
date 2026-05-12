"""Tests for hybrid schema retrieval and object aggregation."""

from __future__ import annotations

from pathlib import Path

from sol01.infra.config import SchemaRetrievalConfig
from sol01.models import ColumnSchema, RetrievalChunk, SchemaObject, TableSchema
from sol01.schema.chunks import render_schema_chunks
from sol01.schema.hybrid_retrieval import (
    build_retrieval_query,
    clip_linked_docs,
    retrieve_schema_objects,
)
from sol01.schema.objects import build_schema_objects
from sol01.schema.retrieval_index import SchemaRetrievalIndex, _build_sparse_index


def test_query_construction_extracts_signals_and_clips_linked_docs_by_overlap():
    query = build_retrieval_query(
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


def test_hybrid_retrieval_merges_sparse_and_exact_hits():
    index = _fake_index()

    objects, diagnostics = retrieve_schema_objects(
        index,
        "Find ORDERS where status is 'closed'",
        config=SchemaRetrievalConfig(object_top_k=6),
    )

    selected_ids = [obj.schema_object.object_id for obj in objects]
    assert "table:DB.PUBLIC.ORDERS" in selected_ids
    assert "sample_value:DB.PUBLIC.ORDERS#STATUS:11111111" in selected_ids

    evidence = {item["chunk_id"]: item["sources"] for item in diagnostics["candidate_evidence"]}
    assert {"sparse", "exact"}.issubset(evidence["table:DB.PUBLIC.ORDERS::table"])
    assert {"sparse", "exact"}.issubset(
        evidence["sample_value:DB.PUBLIC.ORDERS#STATUS:11111111::sample_value"]
    )
    assert set(diagnostics["hit_counts"]) == {"sparse", "exact"}
    assert set(diagnostics["candidate_counts"]) == {"merged", "by_type"}


def test_type_quotas_deduplicate_and_limit_candidates():
    index = _fake_index()

    _, diagnostics = retrieve_schema_objects(
        index,
        "orders status customer amount",
        config=SchemaRetrievalConfig(object_top_k=4),
        type_quotas={
            "table": 20,
            "column": 1,
            "column_group": 20,
            "join_candidate": 20,
            "sample_value": 20,
            "table_family": 20,
        },
    )

    assert diagnostics["candidate_counts"]["by_type"]["column"] == 1


def test_object_aggregation_lifts_parent_tables_and_families_from_child_hits():
    index = _fake_index()

    objects, _ = retrieve_schema_objects(
        index,
        "Which customers have closed order status?",
        config=SchemaRetrievalConfig(object_top_k=8),
    )

    scores = {obj.schema_object.object_id: obj.score for obj in objects}
    assert scores["table:DB.PUBLIC.ORDERS"] > scores["column:DB.PUBLIC.ORDERS#STATUS"]
    assert scores["table:DB.PUBLIC.CUSTOMERS"] > 0
    assert scores["family:DB.PUBLIC:orders_family:deadbeef"] > 0


def test_exact_date_table_and_column_terms_lift_objects_without_sparse_hits():
    index = _exact_only_index()

    objects, diagnostics = retrieve_schema_objects(
        index,
        "Count DB.PUBLIC._20240103 rows by CREATED_AT on 2024-01-03",
        config=SchemaRetrievalConfig(object_top_k=5),
    )

    assert diagnostics["hit_counts"]["sparse"] == 0
    assert diagnostics["hit_counts"]["exact"] >= 2
    selected = [obj.schema_object.object_id for obj in objects]
    assert "table:DB.PUBLIC._20240103" in selected
    assert "column:DB.PUBLIC._20240103#CREATED_AT" in selected
    assert all(hasattr(obj, "schema_object") and hasattr(obj, "chunks") for obj in objects)


def test_curated_summary_aliases_are_exact_retrieval_terms():
    index = _exact_only_index()

    objects, diagnostics = retrieve_schema_objects(
        index,
        "Use the daily github archive for event counts",
        config=SchemaRetrievalConfig(object_top_k=5),
    )

    assert diagnostics["hit_counts"]["sparse"] == 0
    assert diagnostics["hit_counts"]["exact"] >= 1
    assert objects[0].schema_object.object_id == "family:DB.PUBLIC:github_events:12345678"


def test_curated_summary_aliases_are_searchable_from_rendered_chunks():
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
    objects = build_schema_objects({"GITHUB_REPOS_DATE.DAY._20240103": table})
    chunks = render_schema_chunks(objects)
    index = SchemaRetrievalIndex(
        db="GITHUB_REPOS_DATE",
        cache_key="summary-rendered",
        cache_dir=Path("/tmp/test-schema-index-summary-rendered"),
        manifest={},
        objects=objects,
        chunks=chunks,
        sparse=_build_sparse_index(chunks),
    )

    retrieved, diagnostics = retrieve_schema_objects(
        index,
        "Count daily github archive repository events",
        config=SchemaRetrievalConfig(object_top_k=5),
    )

    assert diagnostics["hit_counts"]["sparse"] > 0
    assert diagnostics["hit_counts"]["exact"] > 0
    assert retrieved[0].schema_object.object_id == "table:GITHUB_REPOS_DATE.DAY._20240103"
    assert (
        "Large-schema summary: github_repos_day_events." in retrieved[0].chunks[0].chunk.prompt_text
    )


def _fake_index() -> SchemaRetrievalIndex:
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
        RetrievalChunk(
            chunk_id="table:DB.PUBLIC.ORDERS::table",
            object_id="table:DB.PUBLIC.ORDERS",
            chunk_type="table",
            bm25_text="ORDERS status customer DB.PUBLIC.ORDERS",
            prompt_text="Orders table with status and customer fields.",
        ),
        RetrievalChunk(
            chunk_id="table:DB.PUBLIC.CUSTOMERS::table",
            object_id="table:DB.PUBLIC.CUSTOMERS",
            chunk_type="table",
            bm25_text="CUSTOMERS customer DB.PUBLIC.CUSTOMERS",
            prompt_text="Customers table.",
        ),
        RetrievalChunk(
            chunk_id="column:DB.PUBLIC.ORDERS#STATUS::column",
            object_id="column:DB.PUBLIC.ORDERS#STATUS",
            chunk_type="column",
            parent_object_ids=["table:DB.PUBLIC.ORDERS"],
            bm25_text="STATUS closed open order status",
            prompt_text="Order status column.",
        ),
        RetrievalChunk(
            chunk_id="column:DB.PUBLIC.ORDERS#AMOUNT::column",
            object_id="column:DB.PUBLIC.ORDERS#AMOUNT",
            chunk_type="column",
            parent_object_ids=["table:DB.PUBLIC.ORDERS"],
            bm25_text="AMOUNT amount revenue",
            prompt_text="Order amount column.",
        ),
        RetrievalChunk(
            chunk_id="sample_value:DB.PUBLIC.ORDERS#STATUS:11111111::sample_value",
            object_id="sample_value:DB.PUBLIC.ORDERS#STATUS:11111111",
            chunk_type="sample_value",
            parent_object_ids=[
                "table:DB.PUBLIC.ORDERS",
                "column:DB.PUBLIC.ORDERS#STATUS",
            ],
            bm25_text="closed STATUS DB.PUBLIC.ORDERS",
            prompt_text="Sample value closed for order status.",
        ),
        RetrievalChunk(
            chunk_id="join_candidate:DB.PUBLIC.ORDERS#CUSTOMER_ID->DB.PUBLIC.CUSTOMERS#CUSTOMER_ID:abcdef12::join_candidate",
            object_id="join_candidate:DB.PUBLIC.ORDERS#CUSTOMER_ID->DB.PUBLIC.CUSTOMERS#CUSTOMER_ID:abcdef12",
            chunk_type="join_candidate",
            parent_object_ids=[
                "table:DB.PUBLIC.ORDERS",
                "column:DB.PUBLIC.ORDERS#CUSTOMER_ID",
                "table:DB.PUBLIC.CUSTOMERS",
                "column:DB.PUBLIC.CUSTOMERS#CUSTOMER_ID",
            ],
            bm25_text="orders customers customer_id join",
            prompt_text="Join orders to customers on customer id.",
        ),
        RetrievalChunk(
            chunk_id="family:DB.PUBLIC:orders_family:deadbeef::table_family",
            object_id="family:DB.PUBLIC:orders_family:deadbeef",
            chunk_type="table_family",
            parent_object_ids=["table:DB.PUBLIC.ORDERS"],
            bm25_text="orders family DB.PUBLIC.ORDERS",
            prompt_text="Orders table family.",
        ),
    ]
    return SchemaRetrievalIndex(
        db="DB",
        cache_key="test",
        cache_dir=Path("/tmp/test-schema-index"),
        manifest={},
        objects=objects,
        chunks=chunks,
        sparse=_build_sparse_index(chunks),
    )


def _exact_only_index() -> SchemaRetrievalIndex:
    objects = [
        SchemaObject(
            object_id="table:DB.PUBLIC._20240103",
            object_type="table",
            name="_20240103",
            db="DB",
            table_name="DB.PUBLIC._20240103",
            searchable_text="partition table",
        ),
        SchemaObject(
            object_id="column:DB.PUBLIC._20240103#CREATED_AT",
            object_type="column",
            name="CREATED_AT",
            db="DB",
            table_name="DB.PUBLIC._20240103",
            column_name="CREATED_AT",
            searchable_text="timestamp column",
        ),
        SchemaObject(
            object_id="family:DB.PUBLIC:github_events:12345678",
            object_type="family",
            name="github_events",
            db="DB",
            searchable_text="github events family",
            metadata={"member_table_refs": ["DB.PUBLIC._20240103"]},
        ),
    ]
    chunks = [
        RetrievalChunk(
            chunk_id="table:DB.PUBLIC._20240103::table",
            object_id="table:DB.PUBLIC._20240103",
            chunk_type="table",
            bm25_text="partition shard",
            prompt_text="Table DB.PUBLIC._20240103 with CREATED_AT.",
            metadata={
                "table_name": "DB.PUBLIC._20240103",
                "short_name": "_20240103",
            },
        ),
        RetrievalChunk(
            chunk_id="column:DB.PUBLIC._20240103#CREATED_AT::column",
            object_id="column:DB.PUBLIC._20240103#CREATED_AT",
            chunk_type="column",
            parent_object_ids=["table:DB.PUBLIC._20240103"],
            bm25_text="timestamp field",
            prompt_text="Column CREATED_AT on DB.PUBLIC._20240103.",
            metadata={
                "table_name": "DB.PUBLIC._20240103",
                "column_name": "CREATED_AT",
            },
        ),
        RetrievalChunk(
            chunk_id="family:DB.PUBLIC:github_events:12345678::table_family",
            object_id="family:DB.PUBLIC:github_events:12345678",
            chunk_type="table_family",
            parent_object_ids=["table:DB.PUBLIC._20240103"],
            bm25_text="partition shard",
            prompt_text="Curated table family.",
            metadata={
                "member_table_refs": ["DB.PUBLIC._20240103"],
                "summary_aliases": ["daily github archive", "repository event"],
            },
        ),
    ]
    return SchemaRetrievalIndex(
        db="DB",
        cache_key="test-exact",
        cache_dir=Path("/tmp/test-schema-index-exact"),
        manifest={},
        objects=objects,
        chunks=chunks,
        sparse=_build_sparse_index(chunks),
    )
