"""Tests for resolving logical schema selections to physical tables."""

from __future__ import annotations

from datetime import date, timedelta

from sol01.execution.validation import validate_sql
from sol01.models import (
    ColumnSchema,
    SchemaContextObject,
    SchemaObject,
    SchemaPlanningConstraints,
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

    assert context.resolved_tables == ["DB.PUBLIC.ORDERS"]
    assert context.resolved_tables == ["DB.PUBLIC.ORDERS"]
    assert list(context.table_schemas) == ["DB.PUBLIC.ORDERS"]
    assert "Table: DB.PUBLIC.ORDERS" in context.sql_prompt_context


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
        constraints=SchemaPlanningConstraints(years=[2024]),
    )

    daily_index = {
        "DB.PUBLIC.DAILY_20240101": _table("DAILY_20240101"),
        "DB.PUBLIC.DAILY_20240102": _table("DAILY_20240102"),
        "DB.PUBLIC.DAILY_20240103": _table("DAILY_20240103"),
    }
    daily_context = _resolve_family(
        daily_index,
        question="Show daily sales from 2024-01-01 to 2024-01-02.",
        constraints=SchemaPlanningConstraints(
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
        constraints=SchemaPlanningConstraints(version="v2", suffixes=["_v2"]),
    )

    assert sales_context.resolved_tables == ["DB.PUBLIC.SALES_2024"]
    assert daily_context.resolved_tables == [
        "DB.PUBLIC.DAILY_20240101",
        "DB.PUBLIC.DAILY_20240102",
    ]
    assert model_context.resolved_tables == ["DB.PUBLIC.MODEL_v2"]


def test_resolver_include_all_and_broad_questions_select_all_family_members():
    index = {f"DB.PUBLIC.SALES_{year}": _table(f"SALES_{year}") for year in (2022, 2023, 2024)}

    include_all = _resolve_family(
        index,
        question="Show sales.",
        constraints=SchemaPlanningConstraints(include_all=True),
    )
    broad_question = _resolve_family(index, question="Show every historical sales table.")
    year_span = _resolve_family(index, question="Show sales from 2022 to 2024.")

    assert include_all.resolved_tables == [
        "DB.PUBLIC.SALES_2022",
        "DB.PUBLIC.SALES_2023",
        "DB.PUBLIC.SALES_2024",
    ]
    assert broad_question.resolved_tables == include_all.resolved_tables
    assert year_span.resolved_tables == include_all.resolved_tables


def test_resolver_defaults_ambiguous_family_to_canonical_member_with_warning():
    index = {f"DB.PUBLIC.SALES_{year}": _table(f"SALES_{year}") for year in (2022, 2023, 2024)}

    context = _resolve_family(index, question="Show sales amount.")

    assert context.resolved_tables == ["DB.PUBLIC.SALES_2022"]
    assert context.diagnostics["warnings"] == [
        "No family member constraint was provided for "
        f"{context.selected_objects[0].object_id}; using canonical member DB.PUBLIC.SALES_2022."
    ]


def test_resolver_uses_canonical_member_when_constraints_match_no_family_member():
    index = {f"DB.PUBLIC.SALES_{year}": _table(f"SALES_{year}") for year in (2022, 2023, 2024)}

    context = _resolve_family(
        index,
        question="Show sales in 2030.",
        constraints=SchemaPlanningConstraints(years=[2030]),
    )

    assert context.resolved_tables == ["DB.PUBLIC.SALES_2022"]
    assert "No family members matched constraints" in context.diagnostics["warnings"][0]
    assert context.diagnostics["resolution_entries"][0]["reason"] == "constraints_no_match"


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
        schema_context_evidence=[_schema_context_join_evidence()],
    )

    assert context.resolved_tables == [
        "DB.PUBLIC.SALES_2022",
        "DB.PUBLIC.SALES_2023",
        "DB.PUBLIC.SALES_2024",
    ]
    assert "CREATE TABLE SALES_" not in context.sql_prompt_context
    assert (
        "Physical members: DB.PUBLIC.SALES_2022, DB.PUBLIC.SALES_2023, DB.PUBLIC.SALES_2024"
        in context.sql_prompt_context
    )
    assert "Common columns: ORDER_ID, AMOUNT" in context.sql_prompt_context
    assert "join_candidate: Join candidate:" in context.sql_prompt_context
    assert "Exact columns by type:" in context.sql_prompt_context


def test_resolver_expands_large_date_family_only_to_matching_member():
    family, members = _github_daily_family(120)
    matched_table = members[2]["table_full_name"]
    index = {
        family.metadata["canonical_member"]: _github_table("_20240101"),
        matched_table: _github_table("_20240103"),
    }

    context = resolve_schema_context(
        db="GITHUB_REPOS_DATE",
        selected_objects=[SelectedSchemaObject(object_id=family.object_id, role="primary")],
        canonical_schema_objects=[family],
        db_index=index,
        question="Show GitHub repo events for 2024-01-03.",
        constraints=SchemaPlanningConstraints(
            date_start="2024-01-03",
            date_end="2024-01-03",
        ),
    )

    assert context.resolved_tables == [matched_table]
    assert context.diagnostics["resolution_entries"][0]["reason"] == ("explicit_constraints")
    assert f"Physical members: {matched_table}" in context.sql_prompt_context
    assert "GITHUB_REPOS_DATE.DAY._20240104" not in context.sql_prompt_context


def test_resolver_keeps_large_broad_github_family_symbolic_and_budgeted():
    family, members = _github_daily_family(1500)
    index = {family.metadata["canonical_member"]: _github_table("_20240101")}

    broad_context = resolve_schema_context(
        db="GITHUB_REPOS_DATE",
        selected_objects=[SelectedSchemaObject(object_id=family.object_id, role="primary")],
        canonical_schema_objects=[family],
        db_index=index,
        question="Show every historical GitHub repository event table.",
    )
    include_all_context = resolve_schema_context(
        db="GITHUB_REPOS_DATE",
        selected_objects=[SelectedSchemaObject(object_id=family.object_id, role="primary")],
        canonical_schema_objects=[family],
        db_index=index,
        question="Show GitHub repository event tables.",
        constraints=SchemaPlanningConstraints(include_all=True),
    )

    entry = broad_context.diagnostics["resolution_entries"][0]
    include_all_entry = include_all_context.diagnostics["resolution_entries"][0]
    prompt = broad_context.sql_prompt_context

    assert broad_context.resolved_tables == []
    assert include_all_context.resolved_tables == []
    assert entry["reason"] == "symbolic_broad_question"
    assert include_all_entry["reason"] == "symbolic_include_all"
    assert entry["symbolic"] is True
    assert entry["matched_member_count"] == len(members)
    assert broad_context.diagnostics["warnings"]
    assert "expansion budget" in broad_context.diagnostics["warnings"][0]
    assert "Physical members: kept symbolic (1500 matched; expansion budget 64)" in prompt
    assert "GITHUB_REPOS_DATE.DAY._20240101" in prompt
    assert "GITHUB_REPOS_DATE.DAY._20240410" not in prompt
    assert prompt.count("GITHUB_REPOS_DATE.DAY._") < 20


def test_validation_select_alias_in_group_by_is_not_a_column_error():
    """GROUP BY on a SELECT alias must not be reported as an unknown column."""
    table = _table(
        "BIKESHARE_STATIONS",
        columns=[
            ColumnSchema(name="modified_date", type="NUMBER"),
            ColumnSchema(name="status", type="TEXT"),
            ColumnSchema(name="station_id", type="TEXT"),
        ],
    )
    index = {"DB.PUBLIC.BIKESHARE_STATIONS": table}
    sql = (
        'SELECT DATE_PART(YEAR, TO_TIMESTAMP("modified_date" / 1000000)) AS yr, '
        '"status", COUNT(DISTINCT "station_id") '
        "FROM DB.PUBLIC.BIKESHARE_STATIONS "
        "GROUP BY yr, \"status\""
    )
    report = validate_sql(
        sql,
        allowed_tables=["DB.PUBLIC.BIKESHARE_STATIONS"],
        table_schemas=index,
    )
    assert report.ok is True, report.errors


def test_validation_cte_output_column_unqualified_in_outer_scope_is_not_an_error():
    """Unqualified CTE output column mixed with a base table must not be a hard error."""
    sales_table = _table(
        "SALESORDERHEADER",
        columns=[
            ColumnSchema(name="salespersonid", type="TEXT"),
            ColumnSchema(name="orderdate", type="TEXT"),
            ColumnSchema(name="subtotal", type="NUMBER"),
        ],
    )
    person_table = _table(
        "SALESPERSON",
        columns=[ColumnSchema(name="businessentityid", type="TEXT")],
    )
    index = {
        "DB.PUBLIC.SALESORDERHEADER": sales_table,
        "DB.PUBLIC.SALESPERSON": person_table,
    }
    # yr is a CTE output alias; referencing it unqualified in outer scope must not error
    sql = (
        "WITH spy AS (SELECT \"salespersonid\", "
        "DATE_PART(YEAR, \"orderdate\") AS yr, SUM(\"subtotal\") AS total "
        "FROM DB.PUBLIC.SALESORDERHEADER GROUP BY \"salespersonid\", yr) "
        "SELECT sp.\"businessentityid\", spy.yr, spy.total "
        "FROM DB.PUBLIC.SALESPERSON AS sp "
        "LEFT JOIN spy ON sp.\"businessentityid\" = spy.\"salespersonid\""
    )
    report = validate_sql(
        sql,
        allowed_tables=["DB.PUBLIC.SALESORDERHEADER", "DB.PUBLIC.SALESPERSON"],
        table_schemas=index,
    )
    assert report.ok is True, report.errors


def test_validation_accepts_non_canonical_family_member_from_resolved_allowed_tables():
    index = {f"DB.PUBLIC.SALES_{year}": _table(f"SALES_{year}") for year in (2022, 2023, 2024)}
    context = _resolve_family(
        index,
        question="Show all sales history.",
        constraints=SchemaPlanningConstraints(include_all=True),
    )

    report = validate_sql(
        "SELECT ORDER_ID FROM DB.PUBLIC.SALES_2024",
        allowed_tables=context.resolved_tables,
        table_schemas=context.table_schemas,
    )

    assert report.ok is True
    assert report.referenced_tables == ["DB.PUBLIC.SALES_2024"]


def test_resolver_renders_exact_selected_table_metadata_without_raw_ddl():
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

    assert context.resolved_tables == [table_name]
    assert "Table: COVID19_USA.COVID19_USAFACTS.CONFIRMED_CASES" in context.sql_prompt_context
    assert "Column count: 3" in context.sql_prompt_context
    assert 'TEXT: "state" [TEXT], "county_name" [TEXT]' in context.sql_prompt_context
    assert "NUMERIC: _2020_01_01 [NUMBER]" in context.sql_prompt_context
    assert "CREATE TABLE" not in context.sql_prompt_context
    assert "SECRET_DDL_MARKER" not in context.sql_prompt_context
    assert "SECRET_SAMPLE_MARKER" not in context.sql_prompt_context

    valid = validate_sql(
        f'SELECT "state" FROM {table_name}',
        allowed_tables=context.resolved_tables,
        table_schemas=context.table_schemas,
    )
    invented = validate_sql(
        f"SELECT invented_column FROM {table_name}",
        allowed_tables=context.resolved_tables,
        table_schemas=context.table_schemas,
    )

    assert valid.ok is True
    assert invented.ok is False


def _resolve_family(
    index: dict[str, TableSchema],
    *,
    question: str,
    constraints: SchemaPlanningConstraints | None = None,
    schema_context_evidence: list[SchemaContextObject] | None = None,
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
        schema_context_evidence=schema_context_evidence or [],
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


def _github_daily_family(
    member_count: int,
) -> tuple[SchemaObject, list[dict[str, object]]]:
    start = date(2024, 1, 1)
    members = [_github_member(start + timedelta(days=offset)) for offset in range(member_count)]
    member_refs = [str(member["table_full_name"]) for member in members]
    raw_values = [
        str(member["suffix_dimension"]["raw_value"])
        for member in members
        if isinstance(member.get("suffix_dimension"), dict)
    ]
    values = [
        str(member["suffix_dimension"]["value"])
        for member in members
        if isinstance(member.get("suffix_dimension"), dict)
    ]
    family = SchemaObject(
        object_id="family:GITHUB_REPOS_DATE.DAY:github_repos:12345678",
        object_type="family",
        name="GITHUB_REPOS_DATE.DAY github repos table family",
        db="GITHUB_REPOS_DATE",
        table_name=member_refs[0],
        searchable_text="github repos daily family",
        metadata={
            "canonical_member": member_refs[0],
            "member_table_refs": member_refs,
            "members": members,
            "member_count": member_count,
            "common_columns": ["public", "actor", "created_at", "type", "repo"],
            "suffix_dimensions": [
                {
                    "kind": "YYYYMMDD",
                    "raw_values": raw_values,
                    "values": values,
                }
            ],
        },
    )
    return family, members


def _github_member(day: date) -> dict[str, object]:
    raw_value = day.strftime("%Y%m%d")
    short_name = f"_{raw_value}"
    return {
        "table_full_name": f"GITHUB_REPOS_DATE.DAY.{short_name}",
        "short_name": short_name,
        "suffix_dimension": {
            "kind": "YYYYMMDD",
            "raw_stem": "_",
            "normalized_stem": "github_repos",
            "raw_value": raw_value,
            "value": day.isoformat(),
            "source_table_name": short_name,
        },
    }


def _github_table(name: str) -> TableSchema:
    full_name = f"GITHUB_REPOS_DATE.DAY.{name}"
    return TableSchema(
        name=name,
        database_name="GITHUB_REPOS_DATE",
        schema_name="DAY",
        full_name=full_name,
        ddl=f"CREATE TABLE {name} (public BOOLEAN, actor VARIANT, created_at TIMESTAMP);",
        columns=[
            ColumnSchema(name="public", type="BOOLEAN"),
            ColumnSchema(name="actor", type="VARIANT"),
            ColumnSchema(name="created_at", type="TIMESTAMP"),
            ColumnSchema(name="type", type="TEXT"),
            ColumnSchema(name="repo", type="VARIANT"),
        ],
        searchable_text="github archive events",
    )


def _schema_context_join_evidence() -> SchemaContextObject:
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
    return SchemaContextObject(
        schema_object=join_object,
        planning_text="Join candidate: DB.PUBLIC.SALES_2022.ORDER_ID = DB.PUBLIC.ORDERS.ORDER_ID.",
        position=1,
    )
