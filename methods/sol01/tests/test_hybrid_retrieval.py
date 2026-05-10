"""Tests for hybrid schema retrieval and object aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from sol01.infra.config import SchemaRetrievalConfig
from sol01.models import RetrievalChunk, SchemaObject
from sol01.schema.hybrid_retrieval import (
    build_retrieval_query,
    clip_linked_docs,
    retrieve_schema_objects,
)
from sol01.schema.retrieval_index import SchemaRetrievalIndex, _build_sparse_index


class StaticEmbeddingProvider:
    """Return deterministic vectors for dense retrieval tests."""

    def __init__(self, vector: Sequence[float]) -> None:
        self._vector = list(vector)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector for _ in texts]


class RecordingReranker:
    """Reranker test double that records invocation size."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        return [1.0 - (index * 0.01) for index, _ in enumerate(texts)]


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
    assert "2024-01-03" in query.dates
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


def test_hybrid_retrieval_merges_sparse_dense_exact_and_excludes_sample_dense():
    index = _fake_index()
    reranker = RecordingReranker()

    objects, diagnostics = retrieve_schema_objects(
        index,
        "Find ORDERS where status is 'closed'",
        embedding_provider=StaticEmbeddingProvider([1.0, 0.0, 0.0]),
        reranker_provider=reranker,
        config=SchemaRetrievalConfig(rerank_top_k=3, object_top_k=6),
    )

    selected_ids = [obj.schema_object.object_id for obj in objects]
    assert "table:DB.PUBLIC.ORDERS" in selected_ids
    assert "sample_value:DB.PUBLIC.ORDERS#STATUS:11111111" in selected_ids
    assert diagnostics["reranked_count"] == 3
    assert len(reranker.calls[0][1]) == 3

    evidence = {item["chunk_id"]: item["sources"] for item in diagnostics["candidate_evidence"]}
    assert "dense" not in evidence["sample_value:DB.PUBLIC.ORDERS#STATUS:11111111::sample_value"]
    assert {"sparse", "dense", "exact", "rerank"}.issubset(
        evidence["table:DB.PUBLIC.ORDERS::table"]
    )


def test_type_quotas_deduplicate_before_reranking_and_limit_invocation():
    index = _fake_index()
    reranker = RecordingReranker()

    _, diagnostics = retrieve_schema_objects(
        index,
        "orders status customer amount",
        embedding_provider=StaticEmbeddingProvider([1.0, 0.0, 0.0]),
        reranker_provider=reranker,
        config=SchemaRetrievalConfig(rerank_top_k=2, object_top_k=4),
        type_quotas={
            "table": 20,
            "column": 1,
            "column_group": 20,
            "join_candidate": 20,
            "sample_value": 20,
            "table_family": 20,
        },
    )

    assert diagnostics["hit_counts"]["by_type"]["column"] == 1
    assert diagnostics["reranked_count"] == 2
    assert len(reranker.calls[0][1]) == 2


def test_object_aggregation_lifts_parent_tables_and_families_from_child_hits():
    index = _fake_index()

    objects, _ = retrieve_schema_objects(
        index,
        "Which customers have closed order status?",
        embedding_provider=StaticEmbeddingProvider([0.0, 0.0, 1.0]),
        reranker_provider=RecordingReranker(),
        config=SchemaRetrievalConfig(rerank_top_k=5, object_top_k=8),
    )

    scores = {obj.schema_object.object_id: obj.score for obj in objects}
    assert scores["table:DB.PUBLIC.ORDERS"] > scores["column:DB.PUBLIC.ORDERS#STATUS"]
    assert scores["table:DB.PUBLIC.CUSTOMERS"] > 0
    assert scores["family:DB.PUBLIC:orders_family:deadbeef"] > 0


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
            embedding_text="orders table",
            bm25_text="ORDERS status customer DB.PUBLIC.ORDERS",
            rerank_text="Orders table with status and customer fields.",
        ),
        RetrievalChunk(
            chunk_id="table:DB.PUBLIC.CUSTOMERS::table",
            object_id="table:DB.PUBLIC.CUSTOMERS",
            chunk_type="table",
            embedding_text="customers table",
            bm25_text="CUSTOMERS customer DB.PUBLIC.CUSTOMERS",
            rerank_text="Customers table.",
        ),
        RetrievalChunk(
            chunk_id="column:DB.PUBLIC.ORDERS#STATUS::column",
            object_id="column:DB.PUBLIC.ORDERS#STATUS",
            chunk_type="column",
            parent_object_ids=["table:DB.PUBLIC.ORDERS"],
            embedding_text="status column",
            bm25_text="STATUS closed open order status",
            rerank_text="Order status column.",
        ),
        RetrievalChunk(
            chunk_id="column:DB.PUBLIC.ORDERS#AMOUNT::column",
            object_id="column:DB.PUBLIC.ORDERS#AMOUNT",
            chunk_type="column",
            parent_object_ids=["table:DB.PUBLIC.ORDERS"],
            embedding_text="amount revenue column",
            bm25_text="AMOUNT amount revenue",
            rerank_text="Order amount column.",
        ),
        RetrievalChunk(
            chunk_id="sample_value:DB.PUBLIC.ORDERS#STATUS:11111111::sample_value",
            object_id="sample_value:DB.PUBLIC.ORDERS#STATUS:11111111",
            chunk_type="sample_value",
            parent_object_ids=[
                "table:DB.PUBLIC.ORDERS",
                "column:DB.PUBLIC.ORDERS#STATUS",
            ],
            embedding_text="closed dense trap",
            bm25_text="closed STATUS DB.PUBLIC.ORDERS",
            rerank_text="Sample value closed for order status.",
            include_dense_embedding=False,
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
            embedding_text="orders customers join",
            bm25_text="orders customers customer_id join",
            rerank_text="Join orders to customers on customer id.",
        ),
        RetrievalChunk(
            chunk_id="family:DB.PUBLIC:orders_family:deadbeef::table_family",
            object_id="family:DB.PUBLIC:orders_family:deadbeef",
            chunk_type="table_family",
            parent_object_ids=["table:DB.PUBLIC.ORDERS"],
            embedding_text="orders family",
            bm25_text="orders family DB.PUBLIC.ORDERS",
            rerank_text="Orders table family.",
        ),
    ]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.2, 0.8, 0.0],
            [0.8, 0.1, 0.0],
            [0.7, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.3, 0.0, 0.7],
        ],
        dtype=np.float32,
    )
    return SchemaRetrievalIndex(
        db="DB",
        cache_key="test",
        cache_dir=Path("/tmp/test-schema-index"),
        manifest={},
        objects=objects,
        chunks=chunks,
        sparse=_build_sparse_index(chunks),
        embeddings=embeddings,
    )
