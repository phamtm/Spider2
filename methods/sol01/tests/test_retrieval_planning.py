"""Tests for retrieval-scoped planning prompts and selection cleanup."""

from sol01.llm.prompt_builders import (
    _retrieval_planning_user_prompt,
    sanitize_hybrid_planning_decision,
)
from sol01.models import (
    HybridPlanningConstraints,
    HybridPlanningDecision,
    Intent,
    RetrievalChunk,
    RetrievedChunk,
    RetrievedSchemaObject,
    SchemaObject,
    SelectedSchemaObject,
    Task,
)


def test_retrieval_planning_prompt_uses_retrieved_objects_without_full_schema_summary():
    prompt = _retrieval_planning_user_prompt(
        Task(instance_id="local001", db="DB", question="Revenue for closed orders in 2024"),
        "DB",
        "Closed means STATUS = 'closed'. " * 200,
        _retrieved_objects(),
        max_docs_chars=120,
    )

    assert "Question: Revenue for closed orders in 2024" in prompt
    assert "Document context:" in prompt
    assert "Retrieved logical schema object evidence:" in prompt
    assert "Available object ids:" in prompt
    assert "table:DB.PUBLIC.ORDERS" in prompt
    assert "column:DB.PUBLIC.ORDERS#AMOUNT" in prompt
    assert "Do not invent object ids" in prompt
    assert "Return a HybridPlanningDecision" in prompt
    assert "Schema summary:" not in prompt
    assert "CREATE TABLE" not in prompt
    assert len(prompt.split("Document context:\n", 1)[1].split("\n\n", 1)[0]) <= 123


def test_hybrid_planning_decision_constraints_have_defaults_and_parse_values():
    default_decision = HybridPlanningDecision(
        selected_objects=[],
        rationale="No relevant object was found.",
        confidence=0.0,
        intent=_intent(),
    )
    constrained = HybridPlanningDecision(
        selected_objects=[SelectedSchemaObject(object_id="table:DB.PUBLIC.ORDERS")],
        constraints=HybridPlanningConstraints(
            date_start="2024-01-01",
            date_end="2024-12-31",
            years=[2024],
            suffixes=["2024"],
            version="v2",
            include_all=True,
            notes=["Use annual partition tables."],
        ),
        rationale="Orders table answers the question.",
        confidence=0.8,
        intent=_intent(),
    )

    assert default_decision.constraints.years == []
    assert default_decision.constraints.include_all is False
    assert constrained.constraints.date_start == "2024-01-01"
    assert constrained.constraints.years == [2024]
    assert constrained.constraints.include_all is True


def test_sanitize_hybrid_planning_rejects_hallucinated_ids_and_normalizes_exact_tables():
    decision = HybridPlanningDecision(
        selected_objects=[
            SelectedSchemaObject(object_id="column:DB.PUBLIC.ORDERS#AMOUNT", role="metric"),
            SelectedSchemaObject(object_id="table:DB.PUBLIC.MISSING", role="primary"),
            SelectedSchemaObject(object_id="column:DB.PUBLIC.ORDERS#AMOUNT", role="metric"),
        ],
        selected_tables=["DB.PUBLIC.ORDERS", "DB.PUBLIC.MISSING"],
        rationale="Use orders and amount.",
        confidence=0.7,
        intent=_intent(),
    )

    sanitized, diagnostics = sanitize_hybrid_planning_decision(decision, _retrieved_objects())

    assert [item.object_id for item in sanitized.selected_objects] == [
        "column:DB.PUBLIC.ORDERS#AMOUNT",
        "table:DB.PUBLIC.ORDERS",
    ]
    assert sanitized.selected_objects[1].role == "unknown"
    assert sanitized.selected_tables == []
    assert diagnostics["rejected_object_ids"] == ["table:DB.PUBLIC.MISSING"]
    assert diagnostics["rejected_table_names"] == ["DB.PUBLIC.MISSING"]
    assert diagnostics["normalized_table_names"] == ["DB.PUBLIC.ORDERS"]
    assert "outside retrieved candidates" in sanitized.rationale


def test_sanitize_hybrid_planning_sets_zero_confidence_when_nothing_valid_remains():
    decision = HybridPlanningDecision(
        selected_objects=[SelectedSchemaObject(object_id="table:DB.PUBLIC.MISSING")],
        selected_tables=["DB.PUBLIC.MISSING"],
        rationale="Use missing table.",
        confidence=0.9,
        intent=_intent(),
    )

    sanitized, diagnostics = sanitize_hybrid_planning_decision(decision, _retrieved_objects())

    assert sanitized.selected_objects == []
    assert sanitized.confidence == 0.0
    assert diagnostics["selected_object_count"] == 0
    assert "No valid retrieved schema objects" in sanitized.rationale


def _intent() -> Intent:
    return Intent(
        summary="Find closed order revenue.",
        entities=["orders"],
        metrics=["revenue"],
        filters=["closed"],
        time_constraints=["2024"],
        output_expectation="revenue value",
    )


def _retrieved_objects() -> list[RetrievedSchemaObject]:
    orders = SchemaObject(
        object_id="table:DB.PUBLIC.ORDERS",
        object_type="table",
        name="ORDERS",
        db="DB",
        table_name="DB.PUBLIC.ORDERS",
        searchable_text="orders status amount",
    )
    amount = SchemaObject(
        object_id="column:DB.PUBLIC.ORDERS#AMOUNT",
        object_type="column",
        name="AMOUNT",
        db="DB",
        table_name="DB.PUBLIC.ORDERS",
        column_name="AMOUNT",
        searchable_text="order amount revenue",
    )
    return [
        RetrievedSchemaObject(
            schema_object=orders,
            chunks=[
                RetrievedChunk(
                    chunk=RetrievalChunk(
                        chunk_id="table:DB.PUBLIC.ORDERS::table",
                        object_id="table:DB.PUBLIC.ORDERS",
                        chunk_type="table",
                        prompt_text="Orders table with status and amount fields.",
                    ),
                    rank=1,
                )
            ],
            rank=1,
            score=0.9,
        ),
        RetrievedSchemaObject(
            schema_object=amount,
            chunks=[
                RetrievedChunk(
                    chunk=RetrievalChunk(
                        chunk_id="column:DB.PUBLIC.ORDERS#AMOUNT::column",
                        object_id="column:DB.PUBLIC.ORDERS#AMOUNT",
                        chunk_type="column",
                        prompt_text="Amount column used as revenue evidence.",
                    ),
                    rank=2,
                )
            ],
            rank=2,
            score=0.7,
        ),
    ]
