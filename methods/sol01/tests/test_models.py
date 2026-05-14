import pytest
from pydantic import ValidationError

from sol01.models import (
    ColumnSchema,
    ConfidenceReport,
    ExecutionResult,
    FinalAnswer,
    Intent,
    ResolvedSchemaContext,
    SchemaContextChunk,
    SchemaContextChunkEvidence,
    SchemaContextObject,
    SchemaObject,
    SchemaSelection,
    SelectedSchemaObject,
    SQLCandidate,
    TableSchema,
    Task,
    ValidationReport,
    is_schema_object_id,
    schema_object_id_kind,
    validate_schema_object_id,
)


def test_task_and_intent_models_construct_from_expected_fields():
    task = Task(
        instance_id="local003",
        db="E_commerce",
        question="Which customers have the highest AOV?",
        external_knowledge=None,
    )
    intent = Intent(
        summary="Find top customers by average order value.",
        entities=["customers"],
        metrics=["average order value"],
        filters=[],
        native_value_terms=["orders.status=active"],
        derived_behavioral_definitions=["active means has recent activity"],
        time_constraints=[],
        answer_grain="one row per customer",
        requested_ordering=["highest AOV first"],
        output_expectation="customer and AOV columns",
        assumptions=["Use all orders."],
        evidence=["Question asks for customers with highest AOV."],
        unsupported_assumptions=[],
        do_not_assume=["Do not limit to active customers unless requested."],
    )

    assert task.instance_id == "local003"
    assert task.external_knowledge is None
    assert intent.metrics == ["average order value"]
    assert intent.native_value_terms == ["orders.status=active"]
    assert intent.derived_behavioral_definitions == ["active means has recent activity"]
    assert intent.answer_grain == "one row per customer"
    assert intent.do_not_assume == ["Do not limit to active customers unless requested."]


def test_schema_models_use_independent_default_lists():
    first = ColumnSchema(name="customer_id")
    second = ColumnSchema(name="order_id")
    first.sample_values.append("1")

    table = TableSchema(
        name="orders",
        ddl="CREATE TABLE orders (customer_id INTEGER)",
        columns=[first, second],
        searchable_text="orders customer_id order_id",
    )

    assert first.sample_values == ["1"]
    assert second.sample_values == []
    assert table.sample_rows == []


def test_schema_selection_model_validates_confidence_range():
    selection = SchemaSelection(
        db="E_commerce",
        selected_object_ids=["table:orders"],
        expanded_tables=["orders", "customers"],
        rationale="Question mentions customers and orders.",
        confidence=0.8,
        diagnostics={"selection_prompt_chars": 100, "candidate_table_count": 2},
    )
    assert selection.confidence == 0.8
    assert selection.selected_object_ids == ["table:orders"]
    assert selection.expanded_tables == ["orders", "customers"]
    assert selection.diagnostics["selection_prompt_chars"] == 100

    with pytest.raises(ValidationError):
        SchemaSelection(
            db="E_commerce",
            expanded_tables=[],
            rationale="bad confidence",
            confidence=1.2,
        )


def test_sql_validation_execution_and_critic_models_construct():
    candidate = SQLCandidate(
        sql="SELECT 1 AS answer",
        explanation="Simple query.",
        assumptions=[],
        constraint_ledger=[],
        unsupported_assumptions=[],
        confidence=0.6,
    )
    validation = ValidationReport(
        ok=True,
        errors=[],
        warnings=[],
        referenced_tables=[],
    )
    execution = ExecutionResult(
        ok=True,
        row_count=1,
        columns=["answer"],
        sample_rows=[{"answer": 1}],
        csv_path="outputs/run/csv/local003.csv",
    )
    confidence = ConfidenceReport(
        confidence=0.9,
        issues=[],
        should_repair=False,
    )

    assert candidate.sql.startswith("SELECT")
    assert candidate.constraint_ledger == []
    assert validation.ok is True
    assert execution.error is None
    assert confidence.repair_focus is None


def test_schema_object_id_helpers_validate_stable_formats():
    valid_ids = {
        "table:DB.PUBLIC.ORDERS": "table",
        "column:DB.PUBLIC.ORDERS#CUSTOMER_ID": "column",
        "column_group:DB.PUBLIC.ORDERS#money_fields:1a2b3c4d": "column_group",
        "sample_value:DB.PUBLIC.ORDERS#STATUS:1a2b3c4d": "sample_value",
        "join_candidate:DB.PUBLIC.ORDERS#CUSTOMER_ID->DB.PUBLIC.CUSTOMERS#ID:1a2b3c4d": (
            "join_candidate"
        ),
        "family:DB.PUBLIC:customer_orders:1a2b3c4d": "family",
    }

    for object_id, object_type in valid_ids.items():
        assert is_schema_object_id(object_id)
        assert schema_object_id_kind(object_id) == object_type
        assert validate_schema_object_id(object_id) == object_id

    assert not is_schema_object_id("table")
    with pytest.raises(ValueError, match="stable format"):
        validate_schema_object_id("column:DB.PUBLIC.ORDERS")


def test_schema_context_core_models_construct_and_validate_object_types():
    schema_object = SchemaObject(
        object_id="table:DB.PUBLIC.ORDERS",
        object_type="table",
        name="DB.PUBLIC.ORDERS",
        db="DB",
        searchable_text="orders customers amounts",
    )
    chunk = SchemaContextChunk(
        chunk_id="chunk-1",
        object_id=schema_object.object_id,
        text="Orders table with amount and customer fields.",
    )
    context_chunk = SchemaContextChunkEvidence(chunk=chunk, rank=1, score=0.82)
    context_object = SchemaContextObject(
        schema_object=schema_object,
        chunks=[context_chunk],
        rank=1,
        score=0.9,
    )

    assert context_object.schema_object.object_type == "table"
    assert context_object.chunks[0].chunk.object_id == "table:DB.PUBLIC.ORDERS"

    with pytest.raises(ValidationError, match="object_type"):
        SchemaObject(
            object_id="column:DB.PUBLIC.ORDERS#CUSTOMER_ID",
            object_type="table",
            name="CUSTOMER_ID",
        )


def test_resolved_schema_context_keeps_compact_selection_context():
    selected = SelectedSchemaObject(
        object_id="column:DB.PUBLIC.ORDERS#AMOUNT",
        role="metric",
        confidence=0.8,
    )
    context = ResolvedSchemaContext(
        db="DB",
        selected_objects=[selected],
        resolved_tables=["DB.PUBLIC.ORDERS"],
        prompt_context="Table DB.PUBLIC.ORDERS: AMOUNT",
        diagnostics={"schema_context_object_count": 1},
    )

    assert context.selected_objects[0].role == "metric"
    assert context.resolved_tables == ["DB.PUBLIC.ORDERS"]
    assert context.prompt_context == "Table DB.PUBLIC.ORDERS: AMOUNT"
    assert context.diagnostics == {"schema_context_object_count": 1}


def test_final_answer_status_is_limited_to_expected_values():
    answer = FinalAnswer(
        instance_id="local003",
        status="success",
        sql="SELECT 1",
        csv_path="outputs/run/csv/local003.csv",
        trace_path="outputs/run/traces/local003.json",
    )

    assert answer.status == "success"

    with pytest.raises(ValidationError):
        FinalAnswer(
            instance_id="local003",
            status="done",
            sql=None,
            csv_path=None,
            trace_path="trace.json",
        )
