import pytest
from pydantic import ValidationError

from sol01.models import (
    AggregateGrainReport,
    CandidateComparisonReport,
    CategoryMetadata,
    ColumnSchema,
    ConfidenceReport,
    ExecutionResult,
    FinalAnswer,
    Intent,
    MetricDefinition,
    SchemaSelection,
    SQLCandidate,
    TableSchema,
    Task,
    ValidationReport,
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
        time_constraints=[],
        output_expectation="customer and AOV columns",
        assumptions=["Use all orders."],
    )

    assert task.instance_id == "local003"
    assert task.external_knowledge is None
    assert intent.metrics == ["average order value"]


def test_category_metadata_model_constructs():
    metadata = CategoryMetadata(
        instance_id="sf_bq011",
        primary_tier=6,
        tags=["aggregation", "temporal"],
        difficulty_notes=None,
    )

    assert metadata.instance_id == "sf_bq011"
    assert metadata.primary_tier == 6


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


def test_retrieval_and_metric_models_validate_confidence_range():
    selection = SchemaSelection(
        db="E_commerce",
        selected_tables=["orders"],
        expanded_tables=["orders", "customers"],
        rationale="Question mentions customers and orders.",
        confidence=0.8,
    )
    metric = MetricDefinition(
        metric_name="retention rate",
        definition="Share of users retained over a period.",
        confidence=0.7,
    )

    assert selection.confidence == 0.8
    assert selection.retrieval_mode == "llm_only"
    assert selection.selection_prompt_chars == 0
    assert metric.source_file is None

    with pytest.raises(ValidationError):
        SchemaSelection(
            db="E_commerce",
            selected_tables=[],
            expanded_tables=[],
            rationale="bad confidence",
            confidence=1.2,
        )


def test_sql_validation_execution_and_critic_models_construct():
    candidate = SQLCandidate(
        sql="SELECT 1 AS answer",
        explanation="Simple query.",
        assumptions=[],
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
    assert validation.ok is True
    assert execution.error is None
    assert confidence.repair_focus is None


def test_aggregate_grain_model_constructs():
    grain = AggregateGrainReport(
        inferred_grain="row_count",
        reason="Single entity table with no joins usually counts rows.",
        distinct_reason="DISTINCT is redundant on a single entity table.",
        uses_distinct=True,
        has_joins=False,
        selected_tables=["TEST_DB.PUBLIC.MST_USERS"],
    )

    assert grain.inferred_grain == "row_count"
    assert grain.uses_distinct is True


def test_candidate_comparison_model_constructs():
    comparison = CandidateComparisonReport(
        baseline_stage="initial_1",
        preferred_stage="initial_2",
        compared_stages=["initial_1", "initial_2"],
        reasons=["The second attempt matches the requested shape better."],
    )

    assert comparison.preferred_stage == "initial_2"
    assert comparison.compared_stages == ["initial_1", "initial_2"]


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
