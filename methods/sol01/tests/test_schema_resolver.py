"""Tests for resolving logical schema selections to physical tables."""

from __future__ import annotations

from sol01.execution.validation import validate_sql
from sol01.models import (
    ColumnSchema,
    HybridPlanningConstraints,
    RetrievalChunk,
    RetrievedChunk,
    RetrievedSchemaObject,
    SchemaObject,
    SelectedSchemaObject,
    TableSchema,
)
from sol01.schema.objects import build_schema_objects
from sol01.schema.resolver import resolve_schema_context


def test_resolver_maps_exact_table_selection_to_allowed_table_context():
    index = {"DB.PUBLIC.ORDERS": _table("ORDERS")}
    table = _object(index, "table")

    context = resolve_schema_context(
        db="DB",
        selected_objects=[SelectedSchemaObject(object_id=table.object_id, role="primary")],
        canonical_schema_objects=build_schema_objects(index),
        db_index=index,
        question="Show order revenue.",
    )

    assert context.allowed_tables == ["DB.PUBLIC.ORDERS"]
    assert context.resolved_tables == ["DB.PUBLIC.ORDERS"]
    assert list(context.table_schemas) == ["DB.PUBLIC.ORDERS"]
    assert "Table: DB.PUBLIC.ORDERS" in context.prompt_context


def test_resolver_applies_explicit_year_date_suffix_and_version_constraints():
    sales_index = {
        f"DB.PUBLIC.SALES_{year}": _table(
            f"SALES_{year}",
            ddl=f"CREATE TABLE SALES_{year} (ORDER_ID TEXT, AMOUNT NUMBER);",
        )
        for year in (2022, 2023, 2024)
    }
    sales_context = _resolve_family(
        sales_index,
        question="Show sales in 2024.",
        constraints=HybridPlanningConstraints(years=[2024]),
    )

    daily_index = {
        "DB.PUBLIC.DAILY_20240101": _table("DAILY_20240101"),
        "DB.PUBLIC.DAILY_20240102": _table("DAILY_20240102"),
        "DB.PUBLIC.DAILY_20240103": _table("DAILY_20240103"),
    }
    daily_context = _resolve_family(
        daily_index,
        question="Show daily sales from 2024-01-01 to 2024-01-02.",
        constraints=HybridPlanningConstraints(
            date_start="2024-01-01",
            date_end="2024-01-02",
        ),
    )

    model_index = {
        "DB.PUBLIC.MODEL_v1": _table("MODEL_v1"),
        "DB.PUBLIC.MODEL_v2": _table("MODEL_v2"),
    }
    model_context = _resolve_family(
        model_index,
        question="Use model v2.",
        constraints=HybridPlanningConstraints(version="v2", suffixes=["_v2"]),
    )

    assert sales_context.allowed_tables == ["DB.PUBLIC.SALES_2024"]
    assert daily_context.allowed_tables == [
        "DB.PUBLIC.DAILY_20240101",
        "DB.PUBLIC.DAILY_20240102",
    ]
    assert model_context.allowed_tables == ["DB.PUBLIC.MODEL_v2"]


def test_resolver_include_all_and_broad_questions_select_all_family_members():
    index = {f"DB.PUBLIC.SALES_{year}": _table(f"SALES_{year}") for year in (2022, 2023, 2024)}

    include_all = _resolve_family(
        index,
        question="Show sales.",
        constraints=HybridPlanningConstraints(include_all=True),
    )
    broad_question = _resolve_family(index, question="Show every historical sales table.")
    year_span = _resolve_family(index, question="Show sales from 2022 to 2024.")

    assert include_all.allowed_tables == [
        "DB.PUBLIC.SALES_2022",
        "DB.PUBLIC.SALES_2023",
        "DB.PUBLIC.SALES_2024",
    ]
    assert broad_question.allowed_tables == include_all.allowed_tables
    assert year_span.allowed_tables == include_all.allowed_tables


def test_resolver_defaults_ambiguous_family_to_canonical_member_with_warning():
    index = {f"DB.PUBLIC.SALES_{year}": _table(f"SALES_{year}") for year in (2022, 2023, 2024)}

    context = _resolve_family(index, question="Show sales amount.")

    assert context.allowed_tables == ["DB.PUBLIC.SALES_2022"]
    assert context.resolution_diagnostics["warnings"] == [
        "No family member constraint was provided for "
        f"{context.selected_objects[0].object_id}; using canonical member DB.PUBLIC.SALES_2022."
    ]


def test_resolver_uses_canonical_member_when_constraints_match_no_family_member():
    index = {f"DB.PUBLIC.SALES_{year}": _table(f"SALES_{year}") for year in (2022, 2023, 2024)}

    context = _resolve_family(
        index,
        question="Show sales in 2030.",
        constraints=HybridPlanningConstraints(years=[2030]),
    )

    assert context.allowed_tables == ["DB.PUBLIC.SALES_2022"]
    assert "No family members matched constraints" in context.resolution_diagnostics["warnings"][0]
    assert (
        context.resolution_diagnostics["resolution_entries"][0]["reason"] == "constraints_no_match"
    )


def test_resolver_has_deterministic_table_order_and_compact_family_prompt():
    index = {
        "DB.PUBLIC.SALES_2024": _table(
            "SALES_2024",
            ddl="CREATE TABLE SALES_2024 (ORDER_ID TEXT, AMOUNT NUMBER);",
        ),
        "DB.PUBLIC.SALES_2022": _table(
            "SALES_2022",
            ddl="CREATE TABLE SALES_2022 (ORDER_ID TEXT, AMOUNT NUMBER);",
        ),
        "DB.PUBLIC.SALES_2023": _table(
            "SALES_2023",
            ddl="CREATE TABLE SALES_2023 (ORDER_ID TEXT, AMOUNT NUMBER);",
        ),
    }

    context = _resolve_family(
        index,
        question="Show all sales history.",
        retrieval_evidence=[_retrieved_join_evidence()],
    )

    assert context.allowed_tables == [
        "DB.PUBLIC.SALES_2022",
        "DB.PUBLIC.SALES_2023",
        "DB.PUBLIC.SALES_2024",
    ]
    assert context.prompt_context.count("CREATE TABLE SALES_") == 1
    assert (
        "Physical members: DB.PUBLIC.SALES_2022, DB.PUBLIC.SALES_2023, DB.PUBLIC.SALES_2024"
        in context.prompt_context
    )
    assert "Common columns: ORDER_ID, AMOUNT" in context.prompt_context
    assert "join_candidate: Join candidate:" in context.prompt_context


def test_validation_accepts_non_canonical_family_member_from_resolved_allowed_tables():
    index = {f"DB.PUBLIC.SALES_{year}": _table(f"SALES_{year}") for year in (2022, 2023, 2024)}
    context = _resolve_family(
        index,
        question="Show all sales history.",
        constraints=HybridPlanningConstraints(include_all=True),
    )

    report = validate_sql(
        "SELECT ORDER_ID FROM DB.PUBLIC.SALES_2024",
        allowed_tables=context.allowed_tables,
        table_schemas=context.table_schemas,
    )

    assert report.ok is True
    assert report.referenced_tables == ["DB.PUBLIC.SALES_2024"]


def test_resolver_renders_large_schema_summary_without_raw_table_metadata():
    table_name = "COVID19_USA.COVID19_USAFACTS.CONFIRMED_CASES"
    index = {
        table_name: TableSchema(
            name="CONFIRMED_CASES",
            database_name="COVID19_USA",
            schema_name="COVID19_USAFACTS",
            full_name=table_name,
            ddl="CREATE TABLE CONFIRMED_CASES (SECRET_DDL_MARKER TEXT);",
            columns=[
                ColumnSchema(name="state", type="TEXT"),
                ColumnSchema(name="county_name", type="TEXT"),
                ColumnSchema(name="_2020_01_01", type="NUMBER"),
            ],
            sample_rows=[{"SECRET_SAMPLE_MARKER": "hidden"}],
            searchable_text="covid confirmed cases",
        )
    }
    table = _object(index, "table")

    context = resolve_schema_context(
        db="COVID19_USA",
        selected_objects=[SelectedSchemaObject(object_id=table.object_id, role="primary")],
        canonical_schema_objects=build_schema_objects(index),
        db_index=index,
        question="Show confirmed cases by county.",
    )

    assert context.allowed_tables == [table_name]
    assert "Large-schema summary: covid19_usafacts_wide_daily_counts" in context.prompt_context
    assert "CONFIRMED_CASES and DEATHS repeat daily count columns named _YYYY_MM_DD" in (
        context.prompt_context
    )
    assert "Wide date columns begin with an underscore and must be quoted" in (
        context.prompt_context
    )
    assert "CREATE TABLE" not in context.prompt_context
    assert "SECRET_DDL_MARKER" not in context.prompt_context
    assert "SECRET_SAMPLE_MARKER" not in context.prompt_context

    valid = validate_sql(
        f'SELECT "state" FROM {table_name}',
        allowed_tables=context.allowed_tables,
        table_schemas=context.table_schemas,
    )
    invented = validate_sql(
        f"SELECT invented_column FROM {table_name}",
        allowed_tables=context.allowed_tables,
        table_schemas=context.table_schemas,
    )

    assert valid.ok is True
    assert invented.ok is False


def _resolve_family(
    index: dict[str, TableSchema],
    *,
    question: str,
    constraints: HybridPlanningConstraints | None = None,
    retrieval_evidence: list[RetrievedSchemaObject] | None = None,
):
    objects = build_schema_objects(index)
    family = _object(index, "family")
    return resolve_schema_context(
        db="DB",
        selected_objects=[SelectedSchemaObject(object_id=family.object_id, role="primary")],
        canonical_schema_objects=objects,
        db_index=index,
        question=question,
        constraints=constraints,
        retrieval_evidence=retrieval_evidence or [],
    )


def _object(index: dict[str, TableSchema], object_type: str) -> SchemaObject:
    return next(obj for obj in build_schema_objects(index) if obj.object_type == object_type)


def _table(
    name: str,
    *,
    ddl: str | None = None,
    columns: list[ColumnSchema] | None = None,
) -> TableSchema:
    table_columns = columns or [
        ColumnSchema(name="ORDER_ID", type="TEXT"),
        ColumnSchema(name="AMOUNT", type="NUMBER"),
    ]
    full_name = f"DB.PUBLIC.{name}"
    return TableSchema(
        name=name,
        database_name="DB",
        schema_name="PUBLIC",
        full_name=full_name,
        ddl=ddl or f"CREATE TABLE {name} (ORDER_ID TEXT, AMOUNT NUMBER);",
        columns=table_columns,
        searchable_text=name.lower(),
    )


def _retrieved_join_evidence() -> RetrievedSchemaObject:
    join_object = SchemaObject(
        object_id="join_candidate:DB.PUBLIC.SALES_2022#ORDER_ID->DB.PUBLIC.ORDERS#ORDER_ID:12345678",
        object_type="join_candidate",
        name="SALES_2022.ORDER_ID to ORDERS.ORDER_ID",
        db="DB",
        searchable_text="join sales orders order id",
        metadata={
            "left": {"table_full_name": "DB.PUBLIC.SALES_2022", "column_name": "ORDER_ID"},
            "right": {"table_full_name": "DB.PUBLIC.ORDERS", "column_name": "ORDER_ID"},
        },
    )
    chunk = RetrievalChunk(
        chunk_id=f"{join_object.object_id}::join_candidate",
        object_id=join_object.object_id,
        chunk_type="join_candidate",
        prompt_text="Join candidate: DB.PUBLIC.SALES_2022.ORDER_ID = DB.PUBLIC.ORDERS.ORDER_ID.",
    )
    return RetrievedSchemaObject(
        schema_object=join_object,
        chunks=[RetrievedChunk(chunk=chunk, rank=1)],
        rank=1,
    )
