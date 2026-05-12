"""Tests for retrieval-scoped planning prompts and selection cleanup."""

import pytest

from sol01.infra.config import DEFAULT_MAX_SCHEMA_PROMPT_CHARS
from sol01.llm.prompt_builders import (
    PromptBudgetExceededError,
    _retrieval_planning_user_prompt,
    _sql_reference_context,
    _sql_repair_prompt,
    enforce_prompt_budget,
    sanitize_hybrid_planning_decision,
)
from sol01.models import (
    ColumnSchema,
    HybridPlanningConstraints,
    HybridPlanningDecision,
    Intent,
    RetrievalChunk,
    RetrievedChunk,
    RetrievedSchemaObject,
    SchemaObject,
    SchemaSelection,
    SelectedSchemaObject,
    TableSchema,
    Task,
)
from sol01.schema.chunks import render_schema_chunks
from sol01.schema.objects import build_schema_objects


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


def test_retrieval_planning_prompt_fits_total_budget_without_dropping_object_ids():
    minimal_prompt = _retrieval_planning_user_prompt(
        Task(instance_id="local001", db="DB", question="Revenue for closed orders in 2024"),
        "DB",
        "Closed means STATUS = 'closed'. " * 200,
        _retrieved_objects(),
        max_docs_chars=0,
        max_evidence_chars=0,
    )
    budget = len(minimal_prompt) + 40

    prompt = _retrieval_planning_user_prompt(
        Task(instance_id="local001", db="DB", question="Revenue for closed orders in 2024"),
        "DB",
        "Closed means STATUS = 'closed'. " * 200,
        _retrieved_objects(),
        max_total_chars=budget,
    )

    assert len(prompt) <= budget
    assert "Available object ids:" in prompt
    assert "table:DB.PUBLIC.ORDERS" in prompt
    assert "column:DB.PUBLIC.ORDERS#AMOUNT" in prompt


def test_retrieval_planning_prompt_raises_when_required_shell_exceeds_budget():
    with pytest.raises(PromptBudgetExceededError, match="planning prompt"):
        _retrieval_planning_user_prompt(
            Task(instance_id="local001", db="DB", question="Revenue for closed orders in 2024"),
            "DB",
            "",
            _retrieved_objects(),
            max_docs_chars=0,
            max_evidence_chars=0,
            max_total_chars=10,
        )


def test_retrieval_planning_prompt_uses_curated_summary_evidence_for_covered_tables():
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
        sample_rows=[{"SECRET_SAMPLE_MARKER": "hidden"}],
        searchable_text="github event archive",
    )
    schema_object = next(
        obj
        for obj in build_schema_objects({"GITHUB_REPOS_DATE.DAY._20240103": table})
        if obj.object_type == "table"
    )
    chunk = next(
        chunk for chunk in render_schema_chunks([schema_object]) if chunk.chunk_type == "table"
    )
    prompt = _retrieval_planning_user_prompt(
        Task(
            instance_id="sf_bq_test",
            db="GITHUB_REPOS_DATE",
            question="How many daily github archive events occurred on 2024-01-03?",
        ),
        "GITHUB_REPOS_DATE",
        "",
        [
            RetrievedSchemaObject(
                schema_object=schema_object,
                chunks=[RetrievedChunk(chunk=chunk, rank=1, score=1.0)],
                rank=1,
                score=1.0,
            )
        ],
    )

    assert "Available object ids:" in prompt
    assert "table:GITHUB_REPOS_DATE.DAY._20240103" in prompt
    assert "Large-schema summary: github_repos_day_events." in prompt
    assert "daily github archive" in prompt
    assert "CREATE TABLE" not in prompt
    assert "SECRET_DDL_MARKER" not in prompt
    assert "SECRET_SAMPLE_MARKER" not in prompt
    assert len(prompt) <= DEFAULT_MAX_SCHEMA_PROMPT_CHARS


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


def test_sql_reference_and_repair_prompts_use_large_schema_summary_context():
    table_name = "COVID19_USA.COVID19_USAFACTS.CONFIRMED_CASES"
    table_schemas = {
        table_name: TableSchema(
            name="CONFIRMED_CASES",
            database_name="COVID19_USA",
            schema_name="COVID19_USAFACTS",
            full_name=table_name,
            ddl="CREATE TABLE CONFIRMED_CASES (SECRET_DDL_MARKER TEXT);",
            columns=[ColumnSchema(name="state", type="TEXT")],
            sample_rows=[{"SECRET_SAMPLE_MARKER": "hidden"}],
            searchable_text="covid confirmed cases",
        )
    }
    schema = SchemaSelection(
        db="COVID19_USA",
        selected_object_ids=[f"table:{table_name}"],
        selected_tables=[table_name],
        expanded_tables=[table_name],
        allowed_tables=[table_name],
        rationale="selected covered table",
        confidence=0.9,
    )

    reference_context = _sql_reference_context(schema, table_schemas)
    repair_prompt = _sql_repair_prompt(
        Task(instance_id="sf_bq_test", db="COVID19_USA", question="Show confirmed cases."),
        _intent(),
        {
            "sql": f"SELECT state FROM {table_name}",
            "validation": {"ok": False, "errors": ["unknown column: bad_column"]},
            "execution_result": {"ok": False, "error": "unknown column"},
        },
        reference_context,
        "Confirmed cases are county-level.",
    )

    assert "Large-schema summary: covid19_usafacts_wide_daily_counts" in reference_context
    assert len(reference_context) <= DEFAULT_MAX_SCHEMA_PROMPT_CHARS
    assert "CONFIRMED_CASES" in reference_context
    assert "CREATE TABLE" not in reference_context
    assert "SECRET_DDL_MARKER" not in reference_context
    assert "SECRET_SAMPLE_MARKER" not in reference_context
    assert "unknown column: bad_column" in repair_prompt
    assert "Large-schema summary: covid19_usafacts_wide_daily_counts" in repair_prompt
    assert "CREATE TABLE" not in repair_prompt
    assert "SECRET_DDL_MARKER" not in repair_prompt


def test_sql_reference_budget_enforcement_preserves_large_schema_rules():
    table_name = "COVID19_USA.COVID19_USAFACTS.CONFIRMED_CASES"
    table_schemas = {
        table_name: TableSchema(
            name="CONFIRMED_CASES",
            database_name="COVID19_USA",
            schema_name="COVID19_USAFACTS",
            full_name=table_name,
            ddl="CREATE TABLE CONFIRMED_CASES (SECRET_DDL_MARKER TEXT);",
            columns=[ColumnSchema(name="state", type="TEXT")],
            sample_rows=[],
            searchable_text="covid confirmed cases",
        )
    }
    schema = SchemaSelection(
        db="COVID19_USA",
        selected_object_ids=[f"table:{table_name}"],
        selected_tables=[table_name],
        expanded_tables=[table_name],
        allowed_tables=[table_name],
        rationale="selected covered table",
        confidence=0.9,
    )

    reference_context = _sql_reference_context(schema, table_schemas)
    enforced = enforce_prompt_budget(
        "sql_reference_context",
        reference_context,
        len(reference_context),
    )

    assert enforced == reference_context
    assert table_name in enforced
    assert "CONFIRMED_CASES and DEATHS repeat daily count columns named _YYYY_MM_DD" in enforced
    assert "Wide date columns begin with an underscore and must be quoted" in enforced
    with pytest.raises(PromptBudgetExceededError, match="sql_reference_context prompt"):
        enforce_prompt_budget(
            "sql_reference_context",
            reference_context,
            len(reference_context) - 1,
        )


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
