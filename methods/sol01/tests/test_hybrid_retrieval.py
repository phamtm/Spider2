"""Tests for BM25 sparse retrieval and embedding provider protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sol01.infra.config import SchemaContextConfig
from sol01.models import SchemaContextChunk, SchemaObject
from sol01.schema.embedding import BM25Index, FakeEmbeddingProvider, normalize_vector
from sol01.schema.schema_context import build_available_schema_context
from sol01.schema.schema_context_cache import SchemaContextCache

# ---------------------------------------------------------------------------
# BM25Index tests
# ---------------------------------------------------------------------------


def test_bm25_ranks_matching_document_higher():
    docs = [
        "orders table with customer id and status",
        "revenue and financial metrics by region",
        "user accounts and login history",
    ]
    bm25 = BM25Index(docs)
    ranked = bm25.scores("orders status")
    assert len(ranked) > 0
    top_id, top_score = ranked[0]
    assert top_id == 0
    assert top_score > 0


def test_bm25_returns_empty_for_missing_query_terms():
    bm25 = BM25Index(["alpha beta", "gamma delta"])
    assert bm25.scores("zzznotfound") == []


def test_bm25_top_k_limits_results():
    docs = [f"doc_{i} orders revenue status" for i in range(20)]
    bm25 = BM25Index(docs)
    ranked = bm25.scores("orders", top_k=5)
    assert len(ranked) == 5


def test_bm25_scores_all_without_top_k():
    docs = ["alpha orders", "beta revenue", "gamma status", "delta other"]
    bm25 = BM25Index(docs)
    ranked = bm25.scores("orders status")
    assert len(ranked) == 2  # only docs 0 ("orders") and 2 ("status") match
    doc_ids = {doc_id for doc_id, _ in ranked}
    assert doc_ids == {0, 2}


def test_bm25_exact_match_boosts_relevant_doc():
    docs = [
        "DB.PUBLIC.ORDERS with columns id status amount",
        "DB.PUBLIC.USERS with columns id name email",
    ]
    bm25 = BM25Index(docs)
    boosts = bm25.exact_match_boosts(["DB.PUBLIC.ORDERS"])
    assert 0 in boosts
    assert boosts[0] > 0


def test_bm25_exact_match_returns_empty_for_no_match():
    bm25 = BM25Index(["alpha beta", "gamma delta"])
    assert bm25.exact_match_boosts(["zzz_not_found"]) == {}


def test_bm25_idf_penalizes_ubiquitous_terms():
    docs = ["orders status closed", "users status active", "payments status pending"]
    bm25 = BM25Index(docs)
    # "status" appears in all 3 docs — idf should be low
    idf_status = bm25._idf.get("status", 0.0)
    idf_orders = bm25._idf.get("orders", 0.0)
    assert idf_status < idf_orders


def test_bm25_handles_empty_corpus():
    bm25 = BM25Index([])
    assert bm25.scores("anything") == []
    assert bm25.exact_match_boosts(["ORDERS"]) == {}


def test_bm25_handles_single_document():
    bm25 = BM25Index(["only one document here"])
    ranked = bm25.scores("document")
    assert len(ranked) == 1
    assert ranked[0][0] == 0


# ---------------------------------------------------------------------------
# normalize_vector / cosine_scores
# ---------------------------------------------------------------------------


def test_normalize_vector_produces_unit_length():
    v = np.array([3.0, 4.0], dtype=np.float32)
    nv = normalize_vector(v)
    assert abs(float(np.linalg.norm(nv)) - 1.0) < 1e-6


def test_normalize_zero_vector_returns_zeros():
    v = np.zeros(4, dtype=np.float32)
    nv = normalize_vector(v)
    assert np.all(nv == 0.0)


# ---------------------------------------------------------------------------
# FakeEmbeddingProvider tests
# ---------------------------------------------------------------------------


def test_fake_provider_returns_correct_shape():
    provider = FakeEmbeddingProvider(dim=32)
    vecs = provider.embed(["hello", "world", "orders"])
    assert vecs.shape == (3, 32)
    assert vecs.dtype == np.float32


def test_fake_provider_returns_unit_vectors():
    provider = FakeEmbeddingProvider(dim=64)
    vecs = provider.embed(["test text here"])
    norms = np.linalg.norm(vecs, axis=1)
    assert abs(float(norms[0]) - 1.0) < 1e-5


def test_fake_provider_is_deterministic():
    provider = FakeEmbeddingProvider(dim=16)
    v1 = provider.embed(["hello world"])
    v2 = provider.embed(["hello world"])
    np.testing.assert_array_equal(v1, v2)


def test_fake_provider_different_texts_differ():
    provider = FakeEmbeddingProvider(dim=32)
    vecs = provider.embed(["orders status", "revenue metrics"])
    assert not np.allclose(vecs[0], vecs[1])


def test_fake_provider_handles_empty_batch():
    provider = FakeEmbeddingProvider(dim=8)
    vecs = provider.embed([])
    assert vecs.shape == (0, 8)


def test_fake_provider_dim_property():
    provider = FakeEmbeddingProvider(dim=128)
    assert provider.dim == 128


# ---------------------------------------------------------------------------
# BM25 retrieval integration via build_available_schema_context
# ---------------------------------------------------------------------------


def test_bm25_retrieval_prefers_question_relevant_objects():
    """Objects matching the question vocabulary should rank above unrelated ones."""
    objects = [
        SchemaObject(
            object_id="table:DB.PUBLIC.ORDERS",
            object_type="table",
            name="ORDERS",
            db="DB",
            table_name="DB.PUBLIC.ORDERS",
            searchable_text="orders status customer amount",
        ),
        SchemaObject(
            object_id="table:DB.PUBLIC.WEATHER",
            object_type="table",
            name="WEATHER",
            db="DB",
            table_name="DB.PUBLIC.WEATHER",
            searchable_text="weather temperature precipitation forecast",
        ),
        SchemaObject(
            object_id="column:DB.PUBLIC.ORDERS#STATUS",
            object_type="column",
            name="STATUS",
            db="DB",
            table_name="DB.PUBLIC.ORDERS",
            column_name="STATUS",
            searchable_text="status closed open pending",
        ),
    ]
    chunks = [
        SchemaContextChunk(
            chunk_id="table:DB.PUBLIC.ORDERS::table",
            object_id="table:DB.PUBLIC.ORDERS",
            chunk_type="table",
            evidence_text="DB.PUBLIC.ORDERS orders status customer amount table",
        ),
        SchemaContextChunk(
            chunk_id="table:DB.PUBLIC.WEATHER::table",
            object_id="table:DB.PUBLIC.WEATHER",
            chunk_type="table",
            evidence_text="DB.PUBLIC.WEATHER weather temperature precipitation forecast table",
        ),
        SchemaContextChunk(
            chunk_id="column:DB.PUBLIC.ORDERS#STATUS::column",
            object_id="column:DB.PUBLIC.ORDERS#STATUS",
            chunk_type="column",
            parent_object_ids=["table:DB.PUBLIC.ORDERS"],
            evidence_text="DB.PUBLIC.ORDERS STATUS status closed open pending column",
        ),
    ]
    cache = SchemaContextCache(
        db="DB",
        cache_key="test",
        cache_dir=Path("/tmp/test"),
        manifest={},
        objects=objects,
        chunks=chunks,
    )

    result_objects, diagnostics = build_available_schema_context(
        cache, "Find all orders where status is closed"
    )

    assert diagnostics["context_mode"] == "hybrid_retrieval"
    ids = [obj.schema_object.object_id for obj in result_objects]
    weather_pos = ids.index("table:DB.PUBLIC.WEATHER")
    orders_pos = ids.index("table:DB.PUBLIC.ORDERS")
    status_pos = ids.index("column:DB.PUBLIC.ORDERS#STATUS")
    assert orders_pos < weather_pos
    assert status_pos < weather_pos


def test_bm25_retrieval_top_k_limits_returned_objects():
    """When the database has more objects than top_k, only top_k are returned."""
    objects = [
        SchemaObject(
            object_id=f"table:DB.T{i}",
            object_type="table",
            name=f"T{i}",
            db="DB",
            table_name=f"DB.T{i}",
            searchable_text=f"table_{i} col_a col_b" + (" orders status" if i == 3 else ""),
        )
        for i in range(20)
    ]
    chunks = [
        SchemaContextChunk(
            chunk_id=f"table:DB.T{i}::table",
            object_id=f"table:DB.T{i}",
            chunk_type="table",
            evidence_text=f"T{i} col_a col_b" + (" orders status amount" if i == 3 else ""),
        )
        for i in range(20)
    ]
    cache = SchemaContextCache(
        db="DB",
        cache_key="test",
        cache_dir=Path("/tmp/test"),
        manifest={},
        objects=objects,
        chunks=chunks,
    )

    config = SchemaContextConfig(top_k_objects=5)
    result_objects, diagnostics = build_available_schema_context(
        cache, "Find orders status summary", config=config
    )

    assert len(result_objects) == 5
    assert diagnostics["context_counts"]["available_objects"] == 5
    assert diagnostics["context_counts"]["objects_total"] == 20
    ids = [obj.schema_object.object_id for obj in result_objects]
    assert "table:DB.T3" in ids


def test_per_type_quotas_prevent_single_type_domination():
    """Many column objects should not crowd out table objects when quotas are applied."""
    objects = [
        SchemaObject(
            object_id="table:DB.ORDERS",
            object_type="table",
            name="ORDERS",
            db="DB",
            table_name="DB.ORDERS",
            searchable_text="orders table",
        ),
    ] + [
        SchemaObject(
            object_id=f"column:DB.ORDERS#COL{i}",
            object_type="column",
            name=f"COL{i}",
            db="DB",
            table_name="DB.ORDERS",
            column_name=f"COL{i}",
            searchable_text=f"col{i} orders column",
        )
        for i in range(50)
    ]
    chunks = [
        SchemaContextChunk(
            chunk_id="table:DB.ORDERS::table",
            object_id="table:DB.ORDERS",
            chunk_type="table",
            evidence_text="orders DB.ORDERS table status amount",
        ),
    ] + [
        SchemaContextChunk(
            chunk_id=f"column:DB.ORDERS#COL{i}::column",
            object_id=f"column:DB.ORDERS#COL{i}",
            chunk_type="column",
            parent_object_ids=["table:DB.ORDERS"],
            evidence_text=f"col{i} orders column status amount",
        )
        for i in range(50)
    ]
    cache = SchemaContextCache(
        db="DB",
        cache_key="test",
        cache_dir=Path("/tmp/test"),
        manifest={},
        objects=objects,
        chunks=chunks,
    )

    config = SchemaContextConfig(top_k_objects=30)
    result_objects, _ = build_available_schema_context(
        cache, "Find orders with status", config=config
    )

    types = [obj.schema_object.object_type for obj in result_objects]
    column_count = types.count("column")
    assert column_count <= 40  # per-type quota for columns
    assert "table:DB.ORDERS" in {obj.schema_object.object_id for obj in result_objects}
