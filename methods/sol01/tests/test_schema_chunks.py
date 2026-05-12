"""Tests for deterministic schema object chunk rendering."""

from __future__ import annotations

from sol01.models import ColumnSchema, SchemaObject, TableSchema
from sol01.schema.chunks import render_schema_chunks
from sol01.schema.objects import build_schema_objects


def _table(name: str, columns: list[ColumnSchema], *, ddl: str = "") -> TableSchema:
    full_name = f"DB.PUBLIC.{name}"
    return TableSchema(
        name=name,
        database_name="DB",
        schema_name="PUBLIC",
        full_name=full_name,
        ddl=ddl,
        columns=columns,
        sample_rows=[],
        searchable_text=name.lower(),
    )


def _schema_objects() -> list[SchemaObject]:
    columns = [
        ColumnSchema(name="ORDER_ID", type="TEXT"),
        ColumnSchema(name="CUSTOMER_ID", type="TEXT"),
        ColumnSchema(name="STATUS", type="TEXT", description="Order state"),
        ColumnSchema(name="ORDER_DATE", type="DATE"),
        ColumnSchema(name="TOTAL_AMOUNT", type="NUMBER"),
    ]
    index = {
        "DB.PUBLIC.ORDERS": TableSchema(
            name="ORDERS",
            database_name="DB",
            schema_name="PUBLIC",
            full_name="DB.PUBLIC.ORDERS",
            ddl="CREATE TABLE ORDERS (SECRET_DDL_MARKER TEXT);",
            columns=columns,
            sample_rows=[
                {"STATUS": "open"},
                {"STATUS": "closed"},
                {"STATUS": "open"},
            ],
            searchable_text="orders",
        ),
        "DB.PUBLIC.CUSTOMERS": _table(
            "CUSTOMERS",
            [
                ColumnSchema(name="CUSTOMER_ID", type="TEXT"),
                ColumnSchema(name="CUSTOMER_NAME", type="TEXT"),
            ],
        ),
        "DB.PUBLIC.SALES_2022": _table(
            "SALES_2022",
            [
                ColumnSchema(name="ORDER_ID", type="TEXT"),
                ColumnSchema(name="AMOUNT", type="NUMBER"),
            ],
        ),
        "DB.PUBLIC.SALES_2023": _table(
            "SALES_2023",
            [
                ColumnSchema(name="ORDER_ID", type="TEXT"),
                ColumnSchema(name="AMOUNT", type="NUMBER"),
            ],
        ),
        "DB.PUBLIC.SALES_2024": _table(
            "SALES_2024",
            [
                ColumnSchema(name="ORDER_ID", type="TEXT"),
                ColumnSchema(name="AMOUNT", type="NUMBER"),
            ],
        ),
    }
    return build_schema_objects(index)


def test_renders_deterministic_chunks_for_all_schema_object_types():
    objects = _schema_objects()

    first = render_schema_chunks(objects)
    second = render_schema_chunks(reversed(objects))

    assert [chunk.model_dump(mode="json") for chunk in first] == [
        chunk.model_dump(mode="json") for chunk in second
    ]
    assert {
        "table",
        "column",
        "column_group",
        "join_candidate",
        "sample_value",
        "table_family",
    }.issubset({chunk.chunk_type for chunk in first})
    assert all(chunk.object_id in {obj.object_id for obj in objects} for chunk in first)
    assert all(chunk.chunk_id.startswith(chunk.object_id) for chunk in first)


def test_chunks_do_not_render_full_schema_blobs_or_oversized_sample_values():
    objects = _schema_objects()
    chunks = render_schema_chunks(objects)

    rendered_text = "\n".join(
        "\n".join(
            [
                chunk.bm25_text,
                chunk.prompt_text,
                chunk.source_definition,
            ]
        )
        for chunk in chunks
    )
    assert "SECRET_DDL_MARKER" not in rendered_text

    long_value_object = SchemaObject(
        object_id="sample_value:DB.PUBLIC.ORDERS#STATUS:12345678",
        object_type="sample_value",
        name="x" * 140,
        db="DB",
        table_name="DB.PUBLIC.ORDERS",
        column_name="STATUS",
        metadata={
            "value": "x" * 140,
            "sample_size": 3,
            "distinct_count": 2,
        },
    )
    sample_chunk = render_schema_chunks([long_value_object])[0]

    assert "x" * 100 not in sample_chunk.bm25_text
    assert len(sample_chunk.prompt_text) < 160


def test_source_definition_and_inferred_usage_stay_separate():
    join_chunk = next(
        chunk
        for chunk in render_schema_chunks(_schema_objects())
        if chunk.chunk_type == "join_candidate"
    )

    assert "Source columns" in join_chunk.source_definition
    assert "foreign key" not in join_chunk.source_definition.lower()
    assert "not a declared foreign key" in join_chunk.inferred_usage
    assert "Inferred join evidence" in join_chunk.inferred_usage


def test_sample_value_chunks_are_exact_oriented():
    sample_chunk = next(
        chunk
        for chunk in render_schema_chunks(_schema_objects())
        if chunk.chunk_type == "sample_value"
    )

    assert sample_chunk.source == "sample"
    assert "STATUS" in sample_chunk.bm25_text
    assert "exact filter evidence" in sample_chunk.inferred_usage


def test_family_prompt_text_renders_canonical_structure_compactly():
    family_chunk = next(
        chunk
        for chunk in render_schema_chunks(_schema_objects())
        if chunk.chunk_type == "table_family"
    )

    assert "canonical=DB.PUBLIC.SALES_2022" in family_chunk.prompt_text
    assert "members=3" in family_chunk.prompt_text
    assert "Common columns: ORDER_ID, AMOUNT." in family_chunk.prompt_text
    assert "Partition dimensions: YYYY values 2022, 2023, 2024." in family_chunk.prompt_text


def test_covered_large_table_chunk_uses_curated_summary_not_raw_metadata():
    objects = build_schema_objects(
        {
            "GITHUB_REPOS_DATE.DAY._20240103": TableSchema(
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
                    ColumnSchema(name="SECRET_COLUMN_MARKER", type="TEXT"),
                ],
                sample_rows=[{"SECRET_SAMPLE_MARKER": "hidden"}],
                searchable_text="github archive events",
            )
        }
    )

    table_chunk = next(
        chunk for chunk in render_schema_chunks(objects) if chunk.chunk_type == "table"
    )

    assert "Large-schema summary: github_repos_day_events." in table_chunk.prompt_text
    assert "daily github archive" in table_chunk.bm25_text
    assert "CREATE TABLE" not in table_chunk.prompt_text
    assert "SECRET_DDL_MARKER" not in table_chunk.prompt_text
    assert "SECRET_COLUMN_MARKER" not in table_chunk.prompt_text
    assert "SECRET_SAMPLE_MARKER" not in table_chunk.prompt_text
    assert table_chunk.metadata["summary_ids"] == ["github_repos_day_events"]
    assert "daily github archive" in table_chunk.metadata["summary_aliases"]


def test_covered_large_table_family_chunk_uses_curated_summary_not_member_dump():
    family = SchemaObject(
        object_id="family:GITHUB_REPOS_DATE.DAY:github_repos:12345678",
        object_type="family",
        name="GITHUB_REPOS_DATE.DAY github repos table family",
        db="GITHUB_REPOS_DATE",
        table_name="GITHUB_REPOS_DATE.DAY._20240101",
        searchable_text="github repos daily family",
        metadata={
            "canonical_member": "GITHUB_REPOS_DATE.DAY._20240101",
            "member_table_refs": [
                "GITHUB_REPOS_DATE.DAY._20240101",
                "GITHUB_REPOS_DATE.DAY._20240102",
                "GITHUB_REPOS_DATE.DAY._20240103",
            ],
            "common_columns": ["public", "actor", "created_at", "type", "repo"],
        },
    )

    family_chunk = next(
        chunk for chunk in render_schema_chunks([family]) if chunk.chunk_type == "table_family"
    )

    assert "Large-schema summary: github_repos_day_events." in family_chunk.prompt_text
    assert "Member preview" not in family_chunk.prompt_text
    assert "daily github archive" in family_chunk.bm25_text
    assert family_chunk.metadata["summary_ids"] == ["github_repos_day_events"]
