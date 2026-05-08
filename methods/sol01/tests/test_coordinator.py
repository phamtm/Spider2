"""Tests for the sol01 per-task execution loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from sol01.coordinator import run_task
from sol01.infra.config import RuntimeConfig
from sol01.llm.client import PromptSpec
from sol01.models import (
    CandidateReviewReport,
    ColumnSchema,
    FinalAnswer,
    Intent,
    PlanningDecision,
    SQLCandidate,
    SQLCandidateBatch,
    TableSchema,
    Task,
)
from sol01.output.output import ensure_run_paths

SALES_TABLE = "TEST_DB.PUBLIC.SALES"
ORDERS_TABLE = "TEST_DB.PUBLIC.ORDERS"


@dataclass
class FakeLLMClient:
    """Minimal fake LLM that returns queued outputs by prompt name."""

    outputs: dict[str, list[Any]]
    prompts: dict[str, list[str]] | None = None

    def load_prompt(self, prompt_name: str) -> PromptSpec:
        return PromptSpec(
            name=prompt_name, text=f"{prompt_name} prompt", sha256=f"hash-{prompt_name}"
        )

    def run_structured(
        self,
        user_prompt: str,
        *,
        prompt_name: str,
        output_type: type[Any],
        model: Any = None,
    ) -> Any:
        if self.prompts is not None:
            self.prompts.setdefault(prompt_name, []).append(user_prompt)
        queue = self.outputs.get(prompt_name, [])
        if not queue:
            raise AssertionError(f"No fake output queued for prompt {prompt_name}")
        output = queue.pop(0)
        assert isinstance(output, output_type)
        return output

    def run_structured_with_prompt(
        self,
        user_prompt: str,
        *,
        prompt: PromptSpec,
        output_type: type[Any],
        model: Any = None,
    ) -> Any:
        return self.run_structured(
            user_prompt,
            prompt_name=prompt.name,
            output_type=output_type,
            model=model,
        )


@pytest.fixture
def fake_snowflake(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return deterministic DataFrames instead of opening Snowflake connections."""

    def fake_fetch_query_dataframe(sql: str, *, db: str) -> pd.DataFrame:
        if db != "TEST_DB":
            raise AssertionError(f"Unexpected database: {db}")
        if "MISSING_COLUMN" in sql.upper():
            raise RuntimeError("invalid identifier 'MISSING_COLUMN'")
        if "COUNT" in sql.upper():
            return pd.DataFrame([{"count": 1}])
        if "ORDERS" in sql:
            return pd.DataFrame([{"ORDER_ID": 1, "AMOUNT": 12.0}])
        return pd.DataFrame(
            [
                {"CUSTOMER": "bob", "AMOUNT": 12.0},
                {"CUSTOMER": "alice", "AMOUNT": 10.5},
            ]
        )

    monkeypatch.setattr(
        "sol01.candidates.evaluator.fetch_query_dataframe", fake_fetch_query_dataframe
    )
    monkeypatch.setattr(
        "sol01.candidates.verification._fetch_query_dataframe", fake_fetch_query_dataframe
    )


@pytest.fixture
def db_index(monkeypatch: pytest.MonkeyPatch) -> dict[str, TableSchema]:
    """Patch a compact schema index for coordinator tests."""

    schema = {
        SALES_TABLE: TableSchema(
            name=SALES_TABLE,
            full_name=SALES_TABLE,
            ddl=f"CREATE TABLE {SALES_TABLE} (CUSTOMER TEXT, AMOUNT NUMBER)",
            columns=[
                ColumnSchema(name="CUSTOMER", type="TEXT"),
                ColumnSchema(name="AMOUNT", type="NUMBER"),
            ],
            sample_rows=[{"customer": "bob", "amount": 12}],
            searchable_text="sales customer amount",
        ),
        ORDERS_TABLE: TableSchema(
            name=ORDERS_TABLE,
            full_name=ORDERS_TABLE,
            ddl=f"CREATE TABLE {ORDERS_TABLE} (ORDER_ID NUMBER, AMOUNT NUMBER)",
            columns=[
                ColumnSchema(name="ORDER_ID", type="NUMBER"),
                ColumnSchema(name="AMOUNT", type="NUMBER"),
            ],
            sample_rows=[{"order_id": 1, "amount": 12}],
            searchable_text="orders order amount",
        ),
    }
    monkeypatch.setattr("sol01.coordinator.load_db_index", lambda *args, **kwargs: schema)
    monkeypatch.setattr(
        "sol01.candidates.verification.load_db_index", lambda *args, **kwargs: schema
    )
    return schema


def _planning(
    *, tables: list[str] | None = None, summary: str = "Find totals."
) -> PlanningDecision:
    return PlanningDecision(
        selected_tables=tables or [SALES_TABLE],
        rationale="selected needed tables",
        confidence=0.9,
        intent=Intent(
            summary=summary,
            entities=["sales"],
            metrics=["amount"],
            filters=[],
            time_constraints=[],
            output_expectation="customer and amount columns",
            assumptions=[],
        ),
    )


def _candidate(sql: str, *, confidence: float = 0.8) -> SQLCandidate:
    return SQLCandidate(
        sql=sql,
        explanation="candidate",
        assumptions=[],
        constraint_ledger=[],
        unsupported_assumptions=[],
        confidence=confidence,
    )


def test_run_task_uses_planning_and_batched_generation(
    tmp_path: Path,
    fake_snowflake: None,
    db_index: dict[str, TableSchema],
):
    task = Task(instance_id="sf_local003", db="TEST_DB", question="Show customer totals.")
    run_paths = ensure_run_paths("success-run", outputs_root=tmp_path)
    prompts: dict[str, list[str]] = {}
    llm = FakeLLMClient(
        outputs={
            "planning": [_planning()],
            "sql_generation_batch": [
                SQLCandidateBatch(
                    candidates=[
                        _candidate(
                            f"SELECT CUSTOMER, AMOUNT FROM {SALES_TABLE} ORDER BY AMOUNT DESC"
                        )
                    ]
                )
            ],
        },
        prompts=prompts,
    )

    answer = run_task(
        task,
        run_paths=run_paths,
        config=RuntimeConfig(api_key="test-key"),
        llm_client=llm,
        initial_candidates=1,
    )

    assert isinstance(answer, FinalAnswer)
    assert answer.status == "success"
    trace = json.loads((run_paths.traces_dir / "sf_local003.json").read_text(encoding="utf-8"))
    assert trace["prompt_hashes"] == {
        "planning": "hash-planning",
        "sql_generation_batch": "hash-sql_generation_batch",
    }
    assert trace["schema_selection"]["selected_tables"] == [SALES_TABLE]
    assert len(trace["attempts"]) == 1
    assert "candidate_review" not in trace
    assert set(prompts) == {"planning", "sql_generation_batch"}


def test_close_executable_candidates_use_one_candidate_review(
    tmp_path: Path,
    fake_snowflake: None,
    db_index: dict[str, TableSchema],
):
    task = Task(instance_id="sf_review", db="TEST_DB", question="Show customer totals.")
    run_paths = ensure_run_paths("review-run", outputs_root=tmp_path)
    llm = FakeLLMClient(
        outputs={
            "planning": [_planning()],
            "sql_generation_batch": [
                SQLCandidateBatch(
                    candidates=[
                        _candidate(f"SELECT CUSTOMER, AMOUNT FROM {SALES_TABLE}", confidence=0.7),
                        _candidate(
                            f"SELECT CUSTOMER, AMOUNT FROM {SALES_TABLE} ORDER BY AMOUNT DESC",
                            confidence=0.8,
                        ),
                    ]
                )
            ],
            "candidate_review": [
                CandidateReviewReport(
                    baseline_stage="initial_2",
                    preferred_stage="initial_2",
                    compared_stages=["initial_1", "initial_2"],
                    reasons=["initial_2 preserves ordering"],
                    confidence=0.9,
                    issues=[],
                    should_repair=False,
                )
            ],
        }
    )

    answer = run_task(
        task,
        run_paths=run_paths,
        config=RuntimeConfig(api_key="test-key"),
        llm_client=llm,
        initial_candidates=2,
    )

    assert answer.status == "success"
    trace = json.loads((run_paths.traces_dir / "sf_review.json").read_text(encoding="utf-8"))
    assert trace["candidate_review"]["preferred_stage"] == "initial_2"
    assert trace["final_sql"].endswith("ORDER BY AMOUNT DESC")


def test_tiny_aggregate_is_reviewed_by_candidate_review(
    tmp_path: Path,
    fake_snowflake: None,
    db_index: dict[str, TableSchema],
):
    task = Task(instance_id="sf_count", db="TEST_DB", question="How many customers?")
    run_paths = ensure_run_paths("aggregate-run", outputs_root=tmp_path)
    llm = FakeLLMClient(
        outputs={
            "planning": [_planning(summary="Count customers.")],
            "sql_generation_batch": [
                SQLCandidateBatch(
                    candidates=[_candidate(f"SELECT COUNT(*) AS COUNT FROM {SALES_TABLE}")]
                )
            ],
            "candidate_review": [
                CandidateReviewReport(
                    baseline_stage="initial_1",
                    preferred_stage="initial_1",
                    compared_stages=["initial_1"],
                    reasons=["tiny aggregate may be a false count"],
                    confidence=0.6,
                    issues=["Aggregate returned a suspiciously tiny result."],
                    should_repair=True,
                    repair_focus="Check filters and aggregation grain.",
                )
            ],
            "sql_repair": [
                _candidate(f"SELECT CUSTOMER, AMOUNT FROM {SALES_TABLE} ORDER BY AMOUNT DESC")
            ],
        }
    )

    answer = run_task(
        task,
        run_paths=run_paths,
        config=RuntimeConfig(api_key="test-key"),
        llm_client=llm,
        initial_candidates=1,
    )

    assert answer.status == "success"
    trace = json.loads((run_paths.traces_dir / "sf_count.json").read_text(encoding="utf-8"))
    assert "aggregate_verification" not in trace
    assert trace["candidate_review"]["should_repair"] is True
    assert [attempt["stage"] for attempt in trace["attempts"]] == ["initial_1", "critic_repair"]


def test_schema_expansion_uses_deterministic_table_name_and_reuses_intent(
    tmp_path: Path,
    fake_snowflake: None,
    db_index: dict[str, TableSchema],
):
    task = Task(instance_id="sf_expand", db="TEST_DB", question="Show order totals.")
    run_paths = ensure_run_paths("expand-run", outputs_root=tmp_path)
    prompts: dict[str, list[str]] = {}
    llm = FakeLLMClient(
        outputs={
            "planning": [_planning()],
            "sql_generation_batch": [
                SQLCandidateBatch(candidates=[_candidate("SELECT MISSING_COLUMN FROM ORDERS")]),
                SQLCandidateBatch(
                    candidates=[_candidate(f"SELECT ORDER_ID, AMOUNT FROM {ORDERS_TABLE}")]
                ),
            ],
        },
        prompts=prompts,
    )

    answer = run_task(
        task,
        run_paths=run_paths,
        config=RuntimeConfig(api_key="test-key"),
        llm_client=llm,
        initial_candidates=1,
        max_attempts=1,
        semantic_repairs=0,
    )

    assert answer.status == "success"
    trace = json.loads((run_paths.traces_dir / "sf_expand.json").read_text(encoding="utf-8"))
    assert trace["schema_expansion"]["decision"]["source"] == "deterministic"
    assert trace["schema_selection"]["expanded_tables"] == [SALES_TABLE, ORDERS_TABLE]
    assert "schema_expansion" not in prompts
    assert [attempt["stage"] for attempt in trace["attempts"]] == ["initial_1", "schema_expansion"]
