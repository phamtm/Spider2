"""Tests for the per-task execution loop."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from sol01.config import RuntimeConfig
from sol01.coordinator import (
    _attempt_score,
    _critic_prompt,
    _intent_user_prompt,
    _metric_source_guidance,
    _semantic_repair_prompt,
    _sql_generation_prompt,
    _sql_reference_context,
    _sql_repair_prompt,
    run_task,
    run_tasks,
)
from sol01.llm import PromptSpec
from sol01.models import (
    CandidateComparisonReport,
    ColumnSchema,
    ConfidenceReport,
    ExecutionResult,
    FinalAnswer,
    Intent,
    SchemaSelection,
    SQLCandidate,
    TableSchema,
    Task,
    ValidationReport,
)
from sol01.output import ensure_run_paths
from sol01.retrieval import load_db_index

SALES_TABLE = "TEST_DB.PUBLIC.SALES"
DICOM_TEST_TABLE = "TEST_DB.PUBLIC.DICOM_PIVOT"
BIKE_TABLE = "TEST_DB.PUBLIC.BIKESHARE_STATIONS"


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
        if "missing_column" in sql or "still_missing" in sql:
            raise RuntimeError("invalid identifier 'MISSING_COLUMN'")
        return pd.DataFrame(
            [
                {"customer": "bob", "amount": 12.0},
                {"customer": "alice", "amount": 10.5},
                {"customer": "carol", "amount": 7.5},
            ]
        )

    monkeypatch.setattr("sol01.coordinator.fetch_query_dataframe", fake_fetch_query_dataframe)


def test_run_task_success_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_snowflake: None
):
    task = Task(instance_id="sf_local003", db="TEST_DB", question="Show customer totals.")
    run_paths = ensure_run_paths("success-run", outputs_root=tmp_path)
    llm = FakeLLMClient(
        outputs={
            "intent": [
                Intent(
                    summary="Find customer totals.",
                    entities=["sales"],
                    metrics=[],
                    filters=[],
                    time_constraints=[],
                    output_expectation="customer and total columns",
                    assumptions=["Use all rows."],
                )
            ],
            "sql_generation": [
                SQLCandidate(
                    sql=f"SELECT customer, amount FROM {SALES_TABLE} ORDER BY amount DESC",
                    explanation="Read customer amounts directly.",
                    assumptions=["amount already stores totals"],
                    confidence=0.8,
                )
            ],
            "result_critic": [
                ConfidenceReport(confidence=0.9, issues=[], should_repair=False, repair_focus=None)
            ],
        }
    )

    monkeypatch.setattr(
        "sol01.coordinator.retrieve_schema",
        lambda *args, **kwargs: SchemaSelection(
            db="test_db",
            selected_tables=[SALES_TABLE],
            expanded_tables=[SALES_TABLE],
            rationale="sales is enough",
            confidence=0.9,
        ),
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
    assert answer.csv_path is not None
    assert Path(answer.csv_path).exists()
    assert (run_paths.sql_dir / "sf_local003.sql").exists()

    trace = json.loads((run_paths.traces_dir / "sf_local003.json").read_text(encoding="utf-8"))
    assert trace["status"] == "success"
    assert trace["retrieval_mode"] == "llm_only"
    assert trace["prompt_hashes"]["intent"] == "hash-intent"
    assert len(trace["attempts"]) == 1
    assert trace["attempts"][0]["validation"]["ok"] is True
    assert trace["attempts"][0]["execution_result"]["ok"] is True


def test_run_task_live_client_wires_llm_call_log_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_snowflake: None
):
    task = Task(instance_id="sf_local003", db="TEST_DB", question="Show customer totals.")
    run_paths = ensure_run_paths("logged-run", outputs_root=tmp_path)
    llm = FakeLLMClient(
        outputs={
            "intent": [
                Intent(
                    summary="Find customer totals.",
                    entities=["sales"],
                    metrics=[],
                    filters=[],
                    time_constraints=[],
                    output_expectation="customer and total columns",
                    assumptions=["Use all rows."],
                )
            ],
            "sql_generation": [
                SQLCandidate(
                    sql=f"SELECT customer, amount FROM {SALES_TABLE} ORDER BY amount DESC",
                    explanation="Read customer amounts directly.",
                    assumptions=["amount already stores totals"],
                    confidence=0.8,
                )
            ],
            "result_critic": [
                ConfidenceReport(confidence=0.9, issues=[], should_repair=False, repair_focus=None)
            ],
        }
    )
    created: dict[str, Path] = {}

    def fake_llm_client(config: RuntimeConfig, *, call_logger):
        created["path"] = call_logger.path
        return llm

    monkeypatch.setattr("sol01.coordinator.LLMClient", fake_llm_client)
    monkeypatch.setattr(
        "sol01.coordinator.retrieve_schema",
        lambda *args, **kwargs: SchemaSelection(
            db="test_db",
            selected_tables=[SALES_TABLE],
            expanded_tables=[SALES_TABLE],
            rationale="sales is enough",
            confidence=0.9,
        ),
    )

    answer = run_task(
        task,
        run_paths=run_paths,
        config=RuntimeConfig(api_key="test-key"),
        initial_candidates=1,
    )

    log_path = run_paths.llm_calls_dir / "sf_local003.jsonl"
    assert answer.status == "success"
    assert created["path"] == log_path
    trace = json.loads((run_paths.traces_dir / "sf_local003.json").read_text(encoding="utf-8"))
    assert trace["llm_call_log_path"] == str(log_path)


def test_run_task_compares_executable_candidates_before_critic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_snowflake: None
):
    task = Task(instance_id="sf_local007", db="TEST_DB", question="Show customer totals.")
    run_paths = ensure_run_paths("comparison-run", outputs_root=tmp_path)
    captured_prompts: dict[str, list[str]] = {}
    llm = FakeLLMClient(
        outputs={
            "intent": [
                Intent(
                    summary="Find customer totals.",
                    entities=["sales"],
                    metrics=[],
                    filters=[],
                    time_constraints=[],
                    output_expectation="customer and total columns",
                    assumptions=["Use all rows."],
                )
            ],
            "sql_generation": [
                SQLCandidate(
                    sql=f"SELECT customer, amount FROM {SALES_TABLE} ORDER BY amount DESC",
                    explanation="Read customer amounts directly.",
                    assumptions=["amount already stores totals"],
                    confidence=0.95,
                ),
                SQLCandidate(
                    sql=(
                        f"SELECT customer, SUM(amount) AS total FROM {SALES_TABLE} "
                        "GROUP BY customer ORDER BY total DESC"
                    ),
                    explanation="Aggregate by customer.",
                    assumptions=["amount is additive"],
                    confidence=0.55,
                ),
            ],
            "result_comparison": [
                CandidateComparisonReport(
                    baseline_stage="initial_1",
                    preferred_stage="initial_2",
                    compared_stages=["initial_1", "initial_2"],
                    reasons=[
                        "initial_2 matches the requested customer-total contract better.",
                    ],
                )
            ],
            "result_critic": [
                ConfidenceReport(confidence=0.9, issues=[], should_repair=False, repair_focus=None)
            ],
        },
        prompts=captured_prompts,
    )

    monkeypatch.setattr(
        "sol01.coordinator.retrieve_schema",
        lambda *args, **kwargs: SchemaSelection(
            db="test_db",
            selected_tables=[SALES_TABLE],
            expanded_tables=[SALES_TABLE],
            rationale="sales is enough",
            confidence=0.9,
        ),
    )

    answer = run_task(
        task,
        run_paths=run_paths,
        config=RuntimeConfig(api_key="test-key"),
        llm_client=llm,
        initial_candidates=2,
    )

    assert answer.status == "success"
    trace = json.loads((run_paths.traces_dir / "sf_local007.json").read_text(encoding="utf-8"))
    assert trace["candidate_comparison"]["preferred_stage"] == "initial_2"
    assert [candidate["stage"] for candidate in trace["candidate_comparison"]["candidates"]] == [
        "initial_1",
        "initial_2",
    ]
    assert trace["final_sql"].startswith("SELECT customer, SUM(amount) AS total")
    assert "initial_1" in captured_prompts["result_comparison"][0]
    assert "initial_2" in captured_prompts["result_comparison"][0]
    assert trace["attempts"][0]["candidate_confidence"] == 0.95
    assert trace["attempts"][1]["candidate_confidence"] == 0.55


def test_run_task_default_lifecycle_persists_only_selected_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_snowflake: None
):
    task = Task(instance_id="sf_local008", db="TEST_DB", question="Show customer totals.")
    run_paths = ensure_run_paths("default-lifecycle-run", outputs_root=tmp_path)
    captured_prompts: dict[str, list[str]] = {}
    llm = FakeLLMClient(
        outputs={
            "intent": [
                Intent(
                    summary="Find customer totals.",
                    entities=["sales"],
                    metrics=["total amount"],
                    filters=[],
                    time_constraints=[],
                    output_expectation="customer and total columns",
                    assumptions=["Use all rows."],
                )
            ],
            "sql_generation": [
                SQLCandidate(
                    sql=f"SELECT customer, amount FROM {SALES_TABLE} ORDER BY amount DESC",
                    explanation="Read customer amounts directly.",
                    assumptions=[],
                    confidence=0.8,
                ),
                SQLCandidate(
                    sql=(
                        f"SELECT customer, SUM(amount) AS total FROM {SALES_TABLE} "
                        "GROUP BY customer"
                    ),
                    explanation="Aggregate by customer.",
                    assumptions=[],
                    confidence=0.7,
                ),
                SQLCandidate(
                    sql=(
                        f"SELECT customer, SUM(amount) AS total FROM {SALES_TABLE} "
                        "GROUP BY customer ORDER BY total DESC"
                    ),
                    explanation="Aggregate and order by total.",
                    assumptions=[],
                    confidence=0.6,
                ),
            ],
            "result_comparison": [
                CandidateComparisonReport(
                    baseline_stage="initial_1",
                    preferred_stage="initial_3",
                    compared_stages=["initial_1", "initial_2", "initial_3"],
                    reasons=["initial_3 best matches the requested output and ordering."],
                )
            ],
            "result_critic": [
                ConfidenceReport(confidence=0.9, issues=[], should_repair=False, repair_focus=None)
            ],
        },
        prompts=captured_prompts,
    )

    monkeypatch.setattr(
        "sol01.coordinator.retrieve_schema",
        lambda *args, **kwargs: SchemaSelection(
            db="test_db",
            selected_tables=[SALES_TABLE],
            expanded_tables=[SALES_TABLE],
            rationale="sales is enough",
            confidence=0.9,
        ),
    )

    answer = run_task(
        task,
        run_paths=run_paths,
        config=RuntimeConfig(api_key="test-key"),
        llm_client=llm,
    )

    trace = json.loads((run_paths.traces_dir / "sf_local008.json").read_text(encoding="utf-8"))
    sql_path = run_paths.sql_dir / "sf_local008.sql"
    csv_path = run_paths.csv_dir / "sf_local008.csv"

    assert answer.status == "success"
    assert len(captured_prompts["sql_generation"]) == 3
    assert [attempt["stage"] for attempt in trace["attempts"]] == [
        "initial_1",
        "initial_2",
        "initial_3",
    ]
    assert trace["candidate_comparison"]["preferred_stage"] == "initial_3"
    assert trace["final_sql"] == sql_path.read_text(encoding="utf-8").strip()
    assert trace["final_sql"].endswith("ORDER BY total DESC")
    assert trace["csv_path"] == str(csv_path)
    assert answer.csv_path == str(csv_path)
    assert csv_path.exists()
    assert sorted(path.name for path in run_paths.sql_dir.glob("*.sql")) == ["sf_local008.sql"]
    assert sorted(path.name for path in run_paths.csv_dir.glob("*.csv")) == ["sf_local008.csv"]


def test_run_tasks_keeps_going_after_unexpected_task_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    tasks = [
        Task(instance_id="sf_local001", db="TEST_DB", question="First task."),
        Task(instance_id="sf_local002", db="TEST_DB", question="Second task."),
        Task(instance_id="sf_local003", db="TEST_DB", question="Third task."),
    ]
    run_paths = ensure_run_paths("batch-crash-run", outputs_root=tmp_path)
    called: list[str] = []

    def fake_run_task(
        task: Task,
        *,
        run_paths,
        config,
        llm_client=None,
        force=False,
        skip_failed=False,
        initial_candidates=3,
        max_attempts=4,
        semantic_repairs=1,
    ):
        called.append(task.instance_id)
        if task.instance_id == "sf_local002":
            raise RuntimeError("unexpected task failure")
        return FinalAnswer(
            instance_id=task.instance_id,
            status="success",
            sql=f"SELECT '{task.instance_id}'",
            csv_path=f"/tmp/{task.instance_id}.csv",
            trace_path=str(run_paths.traces_dir / f"{task.instance_id}.json"),
        )

    monkeypatch.setattr("sol01.coordinator.ensure_run_paths", lambda *args, **kwargs: run_paths)
    monkeypatch.setattr("sol01.coordinator.load_db_index", lambda *args, **kwargs: {})
    monkeypatch.setattr("sol01.coordinator.run_task", fake_run_task)

    results = run_tasks(
        tasks,
        run_id="batch-crash-run",
        config=RuntimeConfig(api_key="test-key", concurrency=1),
    )

    assert called == ["sf_local001", "sf_local002", "sf_local003"]
    assert [result.status for result in results] == ["success", "failed", "success"]

    trace = json.loads((run_paths.traces_dir / "sf_local002.json").read_text(encoding="utf-8"))
    assert trace["status"] == "failed"
    assert trace["error"]["type"] == "RuntimeError"
    assert trace["error"]["message"] == "unexpected task failure"
    assert trace["attempts"] == []
    assert trace["csv_path"] is None


def test_run_tasks_bounded_executor_preserves_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    tasks = [
        Task(instance_id="sf_local001", db="TEST_DB", question="First task."),
        Task(instance_id="sf_local002", db="TEST_DB", question="Second task."),
        Task(instance_id="sf_local003", db="TEST_DB", question="Third task."),
    ]
    run_paths = ensure_run_paths("bounded-run", outputs_root=tmp_path)
    active = 0
    max_active = 0
    lock = threading.Lock()
    release_pair = threading.Barrier(2)
    prewarmed: list[str] = []

    def fake_run_task(
        task: Task,
        *,
        run_paths,
        config,
        llm_client=None,
        force=False,
        skip_failed=False,
        initial_candidates=3,
        max_attempts=4,
        semantic_repairs=1,
    ):
        nonlocal active, max_active
        with lock:
            assert prewarmed == ["TEST_DB"]
            active += 1
            max_active = max(max_active, active)
        try:
            if task.instance_id in {"sf_local001", "sf_local002"}:
                release_pair.wait(timeout=2)
            time.sleep(0.05 if task.instance_id == "sf_local001" else 0.01)
            return FinalAnswer(
                instance_id=task.instance_id,
                status="success",
                sql=f"SELECT '{task.instance_id}'",
                csv_path=f"/tmp/{task.instance_id}.csv",
                trace_path=str(run_paths.traces_dir / f"{task.instance_id}.json"),
            )
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr("sol01.coordinator.ensure_run_paths", lambda *args, **kwargs: run_paths)
    monkeypatch.setattr(
        "sol01.coordinator.load_db_index",
        lambda db, *, cache_path=None: prewarmed.append(db) or {},
    )
    monkeypatch.setattr("sol01.coordinator.run_task", fake_run_task)

    results = run_tasks(
        tasks,
        run_id="bounded-run",
        config=RuntimeConfig(api_key="test-key", concurrency=2),
    )

    assert max_active == 2
    assert [result.instance_id for result in results] == [task.instance_id for task in tasks]


def test_run_tasks_prewarms_unique_databases_before_workers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    tasks = [
        Task(instance_id="sf_local001", db="DB_ONE", question="First task."),
        Task(instance_id="sf_local002", db="DB_TWO", question="Second task."),
        Task(instance_id="sf_local003", db="DB_ONE", question="Third task."),
    ]
    run_paths = ensure_run_paths("prewarm-run", outputs_root=tmp_path)
    prewarmed: list[str] = []

    def fake_run_task(
        task: Task,
        *,
        run_paths,
        config,
        llm_client=None,
        force=False,
        skip_failed=False,
        initial_candidates=3,
        max_attempts=4,
        semantic_repairs=1,
    ):
        assert prewarmed == ["DB_ONE", "DB_TWO"]
        return FinalAnswer(
            instance_id=task.instance_id,
            status="success",
            sql=f"SELECT '{task.instance_id}'",
            csv_path=f"/tmp/{task.instance_id}.csv",
            trace_path=str(run_paths.traces_dir / f"{task.instance_id}.json"),
        )

    monkeypatch.setattr("sol01.coordinator.ensure_run_paths", lambda *args, **kwargs: run_paths)
    monkeypatch.setattr(
        "sol01.coordinator.load_db_index",
        lambda db, *, cache_path=None: prewarmed.append(db) or {},
    )
    monkeypatch.setattr("sol01.coordinator.run_task", fake_run_task)

    results = run_tasks(
        tasks,
        run_id="prewarm-run",
        config=RuntimeConfig(api_key="test-key", concurrency=2),
    )

    assert prewarmed == ["DB_ONE", "DB_TWO"]
    assert [result.instance_id for result in results] == [task.instance_id for task in tasks]


def test_run_tasks_bounded_executor_keeps_running_after_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    tasks = [
        Task(instance_id="sf_local001", db="TEST_DB", question="First task."),
        Task(instance_id="sf_local002", db="TEST_DB", question="Second task."),
        Task(instance_id="sf_local003", db="TEST_DB", question="Third task."),
    ]
    run_paths = ensure_run_paths("bounded-failure-run", outputs_root=tmp_path)
    started: list[str] = []
    prewarmed: list[str] = []

    def fake_run_task(
        task: Task,
        *,
        run_paths,
        config,
        llm_client=None,
        force=False,
        skip_failed=False,
        initial_candidates=3,
        max_attempts=4,
        semantic_repairs=1,
    ):
        started.append(task.instance_id)
        assert prewarmed == ["TEST_DB"]
        if task.instance_id == "sf_local002":
            raise RuntimeError("unexpected task failure")
        return FinalAnswer(
            instance_id=task.instance_id,
            status="success",
            sql=f"SELECT '{task.instance_id}'",
            csv_path=f"/tmp/{task.instance_id}.csv",
            trace_path=str(run_paths.traces_dir / f"{task.instance_id}.json"),
        )

    monkeypatch.setattr("sol01.coordinator.ensure_run_paths", lambda *args, **kwargs: run_paths)
    monkeypatch.setattr(
        "sol01.coordinator.load_db_index",
        lambda db, *, cache_path=None: prewarmed.append(db) or {},
    )
    monkeypatch.setattr("sol01.coordinator.run_task", fake_run_task)

    results = run_tasks(
        tasks,
        run_id="bounded-failure-run",
        config=RuntimeConfig(api_key="test-key", concurrency=3),
    )

    assert set(started) == {"sf_local001", "sf_local002", "sf_local003"}
    assert [result.status for result in results] == ["success", "failed", "success"]

    trace = json.loads((run_paths.traces_dir / "sf_local002.json").read_text(encoding="utf-8"))
    assert trace["status"] == "failed"
    assert trace["error"]["type"] == "RuntimeError"
    assert trace["error"]["message"] == "unexpected task failure"


def test_run_task_repairs_validation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_snowflake: None
):
    task = Task(instance_id="sf_local004", db="TEST_DB", question="Show the first customer.")
    run_paths = ensure_run_paths("repair-run", outputs_root=tmp_path)
    llm = FakeLLMClient(
        outputs={
            "intent": [
                Intent(
                    summary="Find one customer.",
                    entities=["sales"],
                    metrics=[],
                    filters=[],
                    time_constraints=[],
                    output_expectation="one customer row",
                    assumptions=[],
                )
            ],
            "sql_generation": [
                SQLCandidate(
                    sql="SELECT * FROM missing_table",
                    explanation="Broken first guess.",
                    assumptions=[],
                    confidence=0.95,
                )
            ],
            "sql_repair": [
                SQLCandidate(
                    sql=f"SELECT customer FROM {SALES_TABLE} LIMIT 1",
                    explanation="Repair to the real table.",
                    assumptions=[],
                    confidence=0.8,
                )
            ],
            "result_critic": [
                ConfidenceReport(confidence=0.9, issues=[], should_repair=False, repair_focus=None)
            ],
        }
    )

    monkeypatch.setattr(
        "sol01.coordinator.retrieve_schema",
        lambda *args, **kwargs: SchemaSelection(
            db="test_db",
            selected_tables=[SALES_TABLE],
            expanded_tables=[SALES_TABLE],
            rationale="sales is enough",
            confidence=0.9,
        ),
    )

    answer = run_task(
        task,
        run_paths=run_paths,
        config=RuntimeConfig(api_key="test-key"),
        llm_client=llm,
        initial_candidates=1,
    )

    assert answer.status == "success"

    trace = json.loads((run_paths.traces_dir / "sf_local004.json").read_text(encoding="utf-8"))
    assert trace["status"] == "success"
    assert len(trace["attempts"]) == 2
    assert trace["attempts"][0]["validation"]["ok"] is False
    assert trace["attempts"][1]["execution_result"]["ok"] is True


def test_run_task_failure_writes_trace_without_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_snowflake: None
):
    task = Task(instance_id="sf_local005", db="TEST_DB", question="Show a missing table.")
    run_paths = ensure_run_paths("failed-run", outputs_root=tmp_path)
    llm = FakeLLMClient(
        outputs={
            "intent": [
                Intent(
                    summary="Broken question path.",
                    entities=[],
                    metrics=[],
                    filters=[],
                    time_constraints=[],
                    output_expectation="none",
                    assumptions=[],
                )
            ],
            "sql_generation": [
                SQLCandidate(
                    sql="DROP TABLE sales",
                    explanation="Unsafe query.",
                    assumptions=[],
                    confidence=0.9,
                )
            ],
            "sql_repair": [
                SQLCandidate(
                    sql="SELECT * FROM missing_table",
                    explanation="Still broken.",
                    assumptions=[],
                    confidence=0.4,
                )
            ],
        }
    )

    monkeypatch.setattr(
        "sol01.coordinator.retrieve_schema",
        lambda *args, **kwargs: SchemaSelection(
            db="test_db",
            selected_tables=[SALES_TABLE],
            expanded_tables=[SALES_TABLE],
            rationale="sales is enough",
            confidence=0.9,
        ),
    )

    answer = run_task(
        task,
        run_paths=run_paths,
        config=RuntimeConfig(api_key="test-key"),
        llm_client=llm,
        initial_candidates=1,
    )

    assert answer.status == "failed"
    assert answer.csv_path is None
    assert not (run_paths.csv_dir / "sf_local005.csv").exists()

    trace = json.loads((run_paths.traces_dir / "sf_local005.json").read_text(encoding="utf-8"))
    assert trace["status"] == "failed"
    assert trace["csv_path"] is None
    assert len(trace["attempts"]) == 2


def test_run_task_verifies_zero_aggregate_results_before_finalizing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_snowflake: None
):
    task = Task(
        instance_id="sf_bq327",
        db="TEST_DB",
        question="Count distinct indicators for a country filter that might use a label variant.",
    )
    run_paths = ensure_run_paths("aggregate-verification-run", outputs_root=tmp_path)
    captured_prompts: dict[str, list[str]] = {}

    def fake_fetch_query_dataframe(sql: str, *, db: str) -> pd.DataFrame:
        if db != "TEST_DB":
            raise AssertionError(f"Unexpected database: {db}")
        if "\"country_name\" = 'Russia'" in sql:
            return pd.DataFrame([{"INDICATOR_COUNT": 0}])
        if "LIKE LOWER('%Russia%')" in sql and "MATCHED_VALUE" in sql:
            return pd.DataFrame([{"MATCHED_VALUE": "Russian Federation"}])
        if "\"country_name\" IN ('Russia', 'Russian Federation')" in sql:
            return pd.DataFrame([{"INDICATOR_COUNT": 4}])
        raise AssertionError(f"Unexpected SQL: {sql}")

    llm = FakeLLMClient(
        outputs={
            "intent": [
                Intent(
                    summary="Count distinct indicators for the requested country bucket.",
                    entities=["international debt", "country"],
                    metrics=["count distinct indicators"],
                    filters=["country_name = 'Russia'"],
                    time_constraints=[],
                    output_expectation="one distinct indicator count",
                    assumptions=["The label may need variant matching."],
                )
            ],
            "sql_generation": [
                SQLCandidate(
                    sql=(
                        'SELECT COUNT(DISTINCT "indicator_code") AS indicator_count '
                        "FROM WORLD_BANK.WORLD_BANK_INTL_DEBT.INTERNATIONAL_DEBT "
                        'WHERE "country_name" = \'Russia\' AND "value" = 0'
                    ),
                    explanation="A direct country filter over distinct indicator codes.",
                    assumptions=["country stores a single canonical label"],
                    confidence=0.95,
                )
            ],
            "aggregate_verification": [
                ConfidenceReport(
                    confidence=0.2,
                    issues=["The zero count looks too small to trust."],
                    should_repair=True,
                    repair_focus="check country value variants and aggregation grain",
                )
            ],
            "sql_repair": [
                SQLCandidate(
                    sql=(
                        'SELECT COUNT(DISTINCT "indicator_code") AS indicator_count '
                        "FROM WORLD_BANK.WORLD_BANK_INTL_DEBT.INTERNATIONAL_DEBT "
                        "WHERE \"country_name\" IN ('Russia', 'Russian Federation') "
                        'AND "value" = 0'
                    ),
                    explanation="Widen the filter to a known country variant.",
                    assumptions=["country may be stored under a longer canonical label"],
                    confidence=0.7,
                )
            ],
            "result_critic": [
                ConfidenceReport(confidence=0.9, issues=[], should_repair=False, repair_focus=None)
            ],
        },
        prompts=captured_prompts,
    )

    monkeypatch.setattr(
        "sol01.coordinator.fetch_query_dataframe",
        fake_fetch_query_dataframe,
    )
    monkeypatch.setattr(
        "sol01.coordinator.retrieve_schema",
        lambda *args, **kwargs: SchemaSelection(
            db="test_db",
            selected_tables=["WORLD_BANK.WORLD_BANK_INTL_DEBT.INTERNATIONAL_DEBT"],
            expanded_tables=["WORLD_BANK.WORLD_BANK_INTL_DEBT.INTERNATIONAL_DEBT"],
            rationale="international debt is enough",
            confidence=0.9,
        ),
    )
    monkeypatch.setattr(
        "sol01.coordinator.load_db_index",
        lambda *args, **kwargs: {
            "WORLD_BANK.WORLD_BANK_INTL_DEBT.INTERNATIONAL_DEBT": TableSchema(
                name="INTERNATIONAL_DEBT",
                full_name="WORLD_BANK.WORLD_BANK_INTL_DEBT.INTERNATIONAL_DEBT",
                ddl=(
                    'create table INTERNATIONAL_DEBT ("country_name" text, '
                    '"indicator_code" text, "indicator_name" text, "value" number);'
                ),
                columns=[
                    ColumnSchema(name="country_name", type="TEXT"),
                    ColumnSchema(name="indicator_code", type="TEXT"),
                    ColumnSchema(name="indicator_name", type="TEXT"),
                    ColumnSchema(name="value", type="NUMBER"),
                ],
                searchable_text="international debt country indicator",
            )
        },
    )

    answer = run_task(
        task,
        run_paths=run_paths,
        config=RuntimeConfig(api_key="test-key"),
        llm_client=llm,
        initial_candidates=1,
    )

    assert answer.status == "success"
    trace = json.loads((run_paths.traces_dir / "sf_bq327.json").read_text(encoding="utf-8"))
    assert trace["aggregate_verification"]["reason"] == (
        "Aggregate query returned a single very small numeric result."
    )
    assert trace["attempts"][0]["filter_grounding_report"]["reason"] == (
        "Empty result but probe values suggest a stored label variant."
    )
    assert trace["attempts"][0]["filter_grounding_report"]["value_rewrites"][0]["rewrite"] == (
        "Russian Federation"
    )
    assert (
        "LIKE LOWER('%Russia%')"
        in trace["attempts"][0]["filter_grounding_report"]["probes"][0]["probe_sql"]
    )
    assert 'COUNT(DISTINCT "indicator_code")' in trace["attempts"][0]["sql"]
    assert trace["attempts"][0]["aggregate_verification"]["should_repair"] is True
    assert trace["attempts"][1]["stage"] == "aggregate_repair"
    assert trace["attempts"][1]["execution_result"]["ok"] is True
    assert trace["final_sql"].startswith(
        'SELECT COUNT(DISTINCT "indicator_code") AS indicator_count'
    )
    assert trace["final_sql"].endswith(
        "WHERE \"country_name\" IN ('Russia', 'Russian Federation') AND \"value\" = 0"
    )
    assert "value variants" in captured_prompts["aggregate_verification"][0]
    assert "grain" in captured_prompts["aggregate_verification"][0]
    assert "Verification:" in captured_prompts["sql_repair"][0]


def test_run_task_prefers_row_count_over_distinct_entity_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_snowflake: None
):
    task = Task(
        instance_id="sf_local358",
        db="TEST_DB",
        question="How many users are in each age bucket?",
    )
    run_paths = ensure_run_paths("grain-run", outputs_root=tmp_path)
    captured_prompts: dict[str, list[str]] = {}
    llm = FakeLLMClient(
        outputs={
            "intent": [
                Intent(
                    summary="Count users by age bucket.",
                    entities=["MST_USERS", "age bucket"],
                    metrics=["count users"],
                    filters=[],
                    time_constraints=[],
                    output_expectation="age bucket and user count",
                    assumptions=["MST_USERS is a master table of users."],
                )
            ],
            "sql_generation": [
                SQLCandidate(
                    sql=(
                        "SELECT age_bucket, COUNT(DISTINCT user_id) AS users "
                        "FROM TEST_DB.PUBLIC.MST_USERS GROUP BY age_bucket"
                    ),
                    explanation="Count distinct users by bucket.",
                    assumptions=["Distinct keeps duplicates out."],
                    confidence=0.95,
                ),
                SQLCandidate(
                    sql=(
                        "SELECT age_bucket, COUNT(*) AS users "
                        "FROM TEST_DB.PUBLIC.MST_USERS GROUP BY age_bucket"
                    ),
                    explanation="Count rows by bucket.",
                    assumptions=["Each row is one user."],
                    confidence=0.55,
                ),
            ],
            "result_comparison": [
                CandidateComparisonReport(
                    baseline_stage="initial_1",
                    preferred_stage="initial_2",
                    compared_stages=["initial_1", "initial_2"],
                    reasons=["COUNT(*) matches the single entity table grain better."],
                )
            ],
            "result_critic": [
                ConfidenceReport(confidence=0.9, issues=[], should_repair=False, repair_focus=None)
            ],
        },
        prompts=captured_prompts,
    )

    monkeypatch.setattr(
        "sol01.coordinator.fetch_query_dataframe",
        lambda sql, *, db: pd.DataFrame(
            [
                {"age_bucket": "18-25", "users": 7},
                {"age_bucket": "26-35", "users": 9},
            ]
        ),
    )
    monkeypatch.setattr(
        "sol01.coordinator.retrieve_schema",
        lambda *args, **kwargs: SchemaSelection(
            db="test_db",
            selected_tables=["TEST_DB.PUBLIC.MST_USERS"],
            expanded_tables=["TEST_DB.PUBLIC.MST_USERS"],
            rationale="MST_USERS is the user master table.",
            confidence=0.9,
        ),
    )

    answer = run_task(
        task,
        run_paths=run_paths,
        config=RuntimeConfig(api_key="test-key"),
        llm_client=llm,
        initial_candidates=2,
    )

    assert answer.status == "success"
    trace = json.loads((run_paths.traces_dir / "sf_local358.json").read_text(encoding="utf-8"))
    assert trace["final_sql"].startswith("SELECT age_bucket, COUNT(*) AS users")
    assert trace["attempts"][0]["aggregate_grain"]["inferred_grain"] == "row_count"
    assert "unnecessary" in trace["attempts"][0]["aggregate_grain"]["distinct_reason"]
    assert trace["attempts"][1]["aggregate_grain"]["inferred_grain"] == "row_count"
    assert trace["attempts"][1]["aggregate_grain"].get("distinct_reason") is None
    assert "row-count style aggregation" in captured_prompts["sql_generation"][0]
    assert "COUNT(*) per group" in captured_prompts["sql_generation"][0]
    assert "COUNT(*)" in captured_prompts["result_comparison"][0]


def test_run_task_catches_execution_errors_and_writes_failed_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_snowflake: None
):
    task = Task(instance_id="sf_local006", db="TEST_DB", question="Select a missing column.")
    run_paths = ensure_run_paths("execution-fail-run", outputs_root=tmp_path)
    llm = FakeLLMClient(
        outputs={
            "intent": [
                Intent(
                    summary="Try to read a column that is not there.",
                    entities=["sales"],
                    metrics=[],
                    filters=[],
                    time_constraints=[],
                    output_expectation="one column",
                    assumptions=[],
                )
            ],
            "sql_generation": [
                SQLCandidate(
                    sql=f"SELECT missing_column FROM {SALES_TABLE}",
                    explanation="Runtime error path.",
                    assumptions=[],
                    confidence=0.9,
                )
            ],
            "sql_repair": [
                SQLCandidate(
                    sql=f"SELECT still_missing FROM {SALES_TABLE}",
                    explanation="Still broken after repair.",
                    assumptions=[],
                    confidence=0.4,
                )
            ],
        }
    )

    monkeypatch.setattr(
        "sol01.coordinator.retrieve_schema",
        lambda *args, **kwargs: SchemaSelection(
            db="test_db",
            selected_tables=[SALES_TABLE],
            expanded_tables=[SALES_TABLE],
            rationale="sales is enough",
            confidence=0.9,
        ),
    )

    answer = run_task(
        task,
        run_paths=run_paths,
        config=RuntimeConfig(api_key="test-key"),
        llm_client=llm,
        initial_candidates=1,
    )

    assert answer.status == "failed"
    trace = json.loads((run_paths.traces_dir / "sf_local006.json").read_text(encoding="utf-8"))
    assert trace["status"] == "failed"
    assert "invalid identifier" in trace["attempts"][0]["execution_result"]["error"].lower()


def test_run_task_repair_prompt_includes_schema_context_for_quoted_identifier_fix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_snowflake: None
):
    task = Task(
        instance_id="sf_bq320",
        db="TEST_DB",
        question="Count unique StudyInstanceUID for matching segment code and collection.",
    )
    run_paths = ensure_run_paths("quoted-identifier-repair-run", outputs_root=tmp_path)
    captured_prompts: dict[str, list[str]] = {}
    llm = FakeLLMClient(
        outputs={
            "intent": [
                Intent(
                    summary="Count matching studies.",
                    entities=[
                        "StudyInstanceUID",
                        "SegmentedPropertyTypeCodeSequence",
                        "collection_id",
                    ],
                    metrics=["COUNT(DISTINCT StudyInstanceUID)"],
                    filters=[
                        "SegmentedPropertyTypeCodeSequence = '15825003'",
                        "collection_id IN ('Community', 'nsclc_radiomics')",
                    ],
                    time_constraints=[],
                    output_expectation="one count",
                    assumptions=[],
                )
            ],
            "sql_generation": [
                SQLCandidate(
                    sql=(
                        f"SELECT COUNT(DISTINCT StudyInstanceUID) FROM {DICOM_TEST_TABLE} "
                        "WHERE LOWER(SegmentedPropertyTypeCodeSequence) = '15825003' "
                        "AND collection_id IN ('Community', 'nsclc_radiomics')"
                    ),
                    explanation="Initial query leaves quoted columns bare.",
                    assumptions=[],
                    confidence=0.9,
                )
            ],
            "sql_repair": [
                SQLCandidate(
                    sql=(
                        f'SELECT COUNT(DISTINCT "StudyInstanceUID") FROM {DICOM_TEST_TABLE} '
                        "WHERE LOWER(\"SegmentedPropertyTypeCodeSequence\") = '15825003' "
                        "AND \"collection_id\" IN ('Community', 'nsclc_radiomics')"
                    ),
                    explanation="Quote all Snowflake columns shown as quoted in DDL.",
                    assumptions=[],
                    confidence=0.8,
                )
            ],
            "result_critic": [
                ConfidenceReport(confidence=0.9, issues=[], should_repair=False, repair_focus=None)
            ],
        },
        prompts=captured_prompts,
    )
    table_schema = TableSchema(
        name="DICOM_PIVOT",
        database_name="TEST_DB",
        schema_name="PUBLIC",
        full_name=DICOM_TEST_TABLE,
        ddl=(
            'create or replace TABLE DICOM_PIVOT ("StudyInstanceUID" VARCHAR, '
            '"SegmentedPropertyTypeCodeSequence" VARCHAR, "collection_id" VARCHAR);'
        ),
        columns=[
            ColumnSchema(name="StudyInstanceUID", type="TEXT"),
            ColumnSchema(name="SegmentedPropertyTypeCodeSequence", type="TEXT"),
            ColumnSchema(name="collection_id", type="TEXT"),
        ],
        searchable_text="DICOM_PIVOT",
    )

    monkeypatch.setattr(
        "sol01.coordinator.retrieve_schema",
        lambda *args, **kwargs: SchemaSelection(
            db="TEST_DB",
            selected_tables=[DICOM_TEST_TABLE],
            expanded_tables=[DICOM_TEST_TABLE],
            rationale="DICOM_PIVOT has the needed columns",
            confidence=0.9,
        ),
    )
    monkeypatch.setattr(
        "sol01.coordinator.load_db_index",
        lambda *args, **kwargs: {DICOM_TEST_TABLE: table_schema},
    )

    answer = run_task(
        task,
        run_paths=run_paths,
        config=RuntimeConfig(api_key="test-key"),
        llm_client=llm,
        initial_candidates=1,
    )

    assert answer.status == "success"
    repair_prompt = captured_prompts["sql_repair"][0]
    assert repair_prompt.startswith("SQL reference context:")
    assert repair_prompt.index("Table: TEST_DB.PUBLIC.DICOM_PIVOT") < repair_prompt.index(
        "Question:"
    )
    assert '"StudyInstanceUID"' in repair_prompt
    assert '"SegmentedPropertyTypeCodeSequence"' in repair_prompt
    assert '"collection_id"' in repair_prompt

    trace = json.loads((run_paths.traces_dir / "sf_bq320.json").read_text(encoding="utf-8"))
    assert trace["attempts"][0]["validation"]["ok"] is False
    assert (
        'Use "StudyInstanceUID" instead of StudyInstanceUID'
        in trace["attempts"][0]["validation"]["errors"][0]
    )


def test_sql_generation_prompt_keeps_reference_context_before_dynamic_task_content():
    task = Task(instance_id="sf_local001", db="TEST_DB", question="Show customer totals.")
    intent = Intent(
        summary="Find customer totals.",
        entities=["sales"],
        metrics=[],
        filters=[],
        time_constraints=[],
        output_expectation="customer and total columns",
        assumptions=[],
    )
    prompt = _sql_generation_prompt(
        task,
        intent,
        "SQL reference context:\nDatabase: TEST_DB\nSelected tables:\n- TEST_DB.PUBLIC.SALES",
        "No task-linked document context.",
    )

    assert prompt.startswith("SQL reference context:")
    assert prompt.index("Document context:") < prompt.index("Question:")
    assert prompt.index("Question:") < prompt.index("Intent:")


def test_metric_source_guidance_prefers_native_metric_at_answer_grain():
    task = Task(
        instance_id="sf_local141",
        db="ADVENTUREWORKS",
        question=(
            "How did each salesperson's annual total sales compare to their annual sales quota?"
        ),
    )
    intent = Intent(
        summary="Compare annual sales and quota by salesperson.",
        entities=["salesperson", "year"],
        metrics=["total sales", "sales quota", "difference"],
        filters=[],
        time_constraints=["annual"],
        output_expectation="salesperson, year, total sales, quota, difference",
        assumptions=[],
    )
    table_schemas = {
        "ADVENTUREWORKS.ADVENTUREWORKS.SALESORDERHEADER": TableSchema(
            name="SALESORDERHEADER",
            database_name="ADVENTUREWORKS",
            schema_name="ADVENTUREWORKS",
            full_name="ADVENTUREWORKS.ADVENTUREWORKS.SALESORDERHEADER",
            ddl="",
            columns=[
                ColumnSchema(name="salespersonid", type="VARCHAR"),
                ColumnSchema(name="orderdate", type="VARCHAR"),
                ColumnSchema(name="subtotal", type="FLOAT"),
                ColumnSchema(name="totaldue", type="FLOAT"),
            ],
            searchable_text="",
        ),
        "ADVENTUREWORKS.ADVENTUREWORKS.SALESORDERDETAIL": TableSchema(
            name="SALESORDERDETAIL",
            database_name="ADVENTUREWORKS",
            schema_name="ADVENTUREWORKS",
            full_name="ADVENTUREWORKS.ADVENTUREWORKS.SALESORDERDETAIL",
            ddl="",
            columns=[
                ColumnSchema(name="salesorderid", type="NUMBER"),
                ColumnSchema(name="orderqty", type="NUMBER"),
                ColumnSchema(name="unitprice", type="FLOAT"),
            ],
            searchable_text="",
        ),
    }

    guidance = _metric_source_guidance(task, intent, table_schemas)

    assert guidance is not None
    assert "requested answer grain" in guidance
    assert "ADVENTUREWORKS.ADVENTUREWORKS.SALESORDERHEADER" in guidance
    assert "subtotal" in guidance
    assert "native metrics [totaldue, subtotal]" in guidance
    assert "column-name semantics" in guidance
    assert "Join lower-grain detail tables only" in guidance


def test_sql_generation_prompt_includes_metric_source_guidance():
    task = Task(instance_id="sf_local141", db="ADVENTUREWORKS", question="Show total sales.")
    intent = Intent(
        summary="Show total sales.",
        entities=[],
        metrics=["total sales"],
        filters=[],
        time_constraints=[],
        output_expectation="total sales",
        assumptions=[],
    )

    prompt = _sql_generation_prompt(
        task,
        intent,
        "SQL reference context:\nDatabase: ADVENTUREWORKS",
        "No task-linked document context.",
        metric_source_guidance="Prefer SALESORDERHEADER.subtotal.",
    )

    assert "Metric source guidance:" in prompt
    assert "Prefer SALESORDERHEADER.subtotal." in prompt


def test_intent_prompt_surfaces_sample_value_groundings():
    task = Task(
        instance_id="sf_bq279",
        db="TEST_DB",
        question="How many active and closed stations were there in 2013 and 2014?",
    )
    schema = SchemaSelection(
        db="TEST_DB",
        selected_tables=[BIKE_TABLE],
        expanded_tables=[BIKE_TABLE],
        rationale="station status is needed",
        confidence=1.0,
    )
    table_schemas = {
        BIKE_TABLE: TableSchema(
            name="BIKESHARE_STATIONS",
            full_name=BIKE_TABLE,
            ddl='CREATE TABLE BIKESHARE_STATIONS ("status" TEXT, "year" NUMBER);',
            columns=[
                ColumnSchema(name="status", type="TEXT", sample_values=["active", "closed"]),
                ColumnSchema(name="year", type="NUMBER", sample_values=["2013", "2014"]),
            ],
            searchable_text="bike stations status year",
        )
    }

    prompt = _intent_user_prompt(task, schema, "No task-linked document context.", table_schemas)

    assert "Grounded literal values:" in prompt
    assert f"{BIKE_TABLE}.status=active" in prompt
    assert f"{BIKE_TABLE}.status=closed" in prompt
    assert "native column values" in prompt


def test_intent_prompt_does_not_ground_one_character_substrings():
    task = Task(
        instance_id="sf_local141",
        db="TEST_DB",
        question="How did each salesperson's annual total sales compare to quota?",
    )
    schema = SchemaSelection(
        db="TEST_DB",
        selected_tables=[SALES_TABLE],
        expanded_tables=[SALES_TABLE],
        rationale="sales is needed",
        confidence=1.0,
    )
    table_schemas = {
        SALES_TABLE: TableSchema(
            name="SALES",
            database_name="TEST_DB",
            schema_name="PUBLIC",
            full_name=SALES_TABLE,
            ddl="",
            columns=[
                ColumnSchema(name="onlineorderflag", type="VARCHAR", sample_values=["f", "t"]),
                ColumnSchema(name="amount", type="FLOAT"),
            ],
            searchable_text="",
        )
    }

    prompt = _intent_user_prompt(task, schema, "No task-linked document context.", table_schemas)

    assert "onlineorderflag=f" not in prompt
    assert "Grounded literal values:" not in prompt


def test_sql_repair_prompt_keeps_reference_context_before_failed_sql():
    task = Task(instance_id="sf_local001", db="TEST_DB", question="Show customer totals.")
    intent = Intent(
        summary="Find customer totals.",
        entities=["sales"],
        metrics=["count customers"],
        filters=[],
        native_value_terms=[f"{SALES_TABLE}.status=active"],
        derived_behavioral_definitions=["active means has at least one trip"],
        time_constraints=[],
        output_expectation="customer totals",
        assumptions=[],
    )
    attempt = {
        "sql": f"SELECT missing_column FROM {SALES_TABLE}",
        "validation": {"ok": True, "errors": [], "warnings": []},
        "execution_result": {"ok": False, "error": "invalid identifier"},
    }

    prompt = _sql_repair_prompt(
        task,
        intent,
        attempt,
        "SQL reference context:\nDatabase: TEST_DB\nSelected tables:\n- TEST_DB.PUBLIC.SALES",
        "No task-linked document context.",
    )

    assert prompt.startswith("SQL reference context:")
    assert prompt.index("Question:") < prompt.index("Failed SQL:")
    assert prompt.index("Failed SQL:") < prompt.index("Validation:")
    assert f"{SALES_TABLE}.status=active" in prompt


def test_sql_reference_context_includes_dicom_pivot_quoted_columns():
    table = load_db_index("IDC")["IDC.IDC_V17.DICOM_PIVOT"]
    context = _sql_reference_context(
        SchemaSelection(
            db="IDC",
            selected_tables=["IDC.IDC_V17.DICOM_PIVOT"],
            expanded_tables=["IDC.IDC_V17.DICOM_PIVOT"],
            rationale="DICOM_PIVOT has the needed columns",
            confidence=1.0,
        ),
        {"IDC.IDC_V17.DICOM_PIVOT": table},
    )

    assert '"StudyInstanceUID"' in context
    assert '"SegmentedPropertyTypeCodeSequence"' in context
    assert '"collection_id"' in context


def test_run_task_uses_only_external_knowledge_for_document_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_snowflake: None
):
    task = Task(
        instance_id="sf_local003",
        db="TEST_DB",
        question="Show customer totals.",
        external_knowledge="RFM.md",
    )
    run_paths = ensure_run_paths("external-knowledge-run", outputs_root=tmp_path)
    captured_prompts: dict[str, list[str]] = {}
    llm = FakeLLMClient(
        outputs={
            "intent": [
                Intent(
                    summary="Find customer totals.",
                    entities=["sales"],
                    metrics=["frequency"],
                    filters=[],
                    time_constraints=[],
                    output_expectation="customer and total columns",
                    assumptions=["Use all rows."],
                )
            ],
            "sql_generation": [
                SQLCandidate(
                    sql=f"SELECT customer, amount FROM {SALES_TABLE} ORDER BY amount DESC",
                    explanation="Read customer amounts directly.",
                    assumptions=["amount already stores totals"],
                    confidence=0.8,
                )
            ],
            "result_critic": [
                ConfidenceReport(confidence=0.9, issues=[], should_repair=False, repair_focus=None)
            ],
        },
        prompts=captured_prompts,
    )

    monkeypatch.setattr(
        "sol01.coordinator.retrieve_schema",
        lambda *args, **kwargs: SchemaSelection(
            db="test_db",
            selected_tables=[SALES_TABLE],
            expanded_tables=[SALES_TABLE],
            rationale="sales is enough",
            confidence=0.9,
        ),
    )
    monkeypatch.setattr(
        "sol01.coordinator.load_document_text",
        lambda file_name: f"WHOLE DOC FOR {file_name}",
    )

    answer = run_task(
        task,
        run_paths=run_paths,
        config=RuntimeConfig(api_key="test-key"),
        llm_client=llm,
        initial_candidates=1,
    )

    assert answer.status == "success"
    intent_prompt = captured_prompts["intent"][0]
    assert "Document context:\nWHOLE DOC FOR RFM.md" in intent_prompt
    sql_prompt = captured_prompts["sql_generation"][0]
    assert "Document context:\nWHOLE DOC FOR RFM.md" in sql_prompt


def test_run_task_grounds_status_literals_before_sql_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    task = Task(
        instance_id="sf_bq279",
        db="TEST_DB",
        question="How many active and closed stations were there in 2013 and 2014?",
    )
    run_paths = ensure_run_paths("status-grounding-run", outputs_root=tmp_path)
    captured_prompts: dict[str, list[str]] = {}
    llm = FakeLLMClient(
        outputs={
            "intent": [
                Intent(
                    summary="Count stations by status.",
                    entities=["stations"],
                    metrics=["count stations"],
                    filters=[],
                    time_constraints=[],
                    output_expectation="status and count columns",
                    assumptions=["Use the selected station table."],
                )
            ],
            "sql_generation": [
                SQLCandidate(
                    sql=(
                        f'SELECT "status", COUNT(*) AS station_count FROM {BIKE_TABLE} '
                        'GROUP BY "status" ORDER BY "status"'
                    ),
                    explanation="Count stations by stored status.",
                    assumptions=["status stores the station state"],
                    confidence=0.9,
                )
            ],
            "result_critic": [
                ConfidenceReport(confidence=0.95, issues=[], should_repair=False, repair_focus=None)
            ],
        },
        prompts=captured_prompts,
    )

    bike_table = TableSchema(
        name="BIKESHARE_STATIONS",
        full_name=BIKE_TABLE,
        ddl='CREATE TABLE BIKESHARE_STATIONS ("status" TEXT, "year" NUMBER);',
        columns=[
            ColumnSchema(name="status", type="TEXT", sample_values=["active", "closed"]),
            ColumnSchema(name="year", type="NUMBER"),
        ],
        searchable_text="bike stations status year",
    )

    monkeypatch.setattr(
        "sol01.coordinator.retrieve_schema",
        lambda *args, **kwargs: SchemaSelection(
            db="TEST_DB",
            selected_tables=[BIKE_TABLE],
            expanded_tables=[BIKE_TABLE],
            rationale="station status is needed",
            confidence=0.9,
        ),
    )
    monkeypatch.setattr(
        "sol01.coordinator.load_db_index",
        lambda *args, **kwargs: {BIKE_TABLE: bike_table},
    )
    monkeypatch.setattr(
        "sol01.coordinator.fetch_query_dataframe",
        lambda sql, db: pd.DataFrame(
            [
                {"status": "active", "station_count": 10},
                {"status": "closed", "station_count": 1},
            ]
        ),
    )

    answer = run_task(
        task,
        run_paths=run_paths,
        config=RuntimeConfig(api_key="test-key"),
        llm_client=llm,
        initial_candidates=1,
    )

    assert answer.status == "success"
    intent_prompt = captured_prompts["intent"][0]
    assert "Grounded literal values:" in intent_prompt
    assert f"{BIKE_TABLE}.status=active" in intent_prompt
    assert f"{BIKE_TABLE}.status=closed" in intent_prompt

    sql_prompt = captured_prompts["sql_generation"][0]
    assert '"native_value_terms": [' in sql_prompt
    assert f"{BIKE_TABLE}.status=active" in sql_prompt
    assert f"{BIKE_TABLE}.status=closed" in sql_prompt

    trace = json.loads((run_paths.traces_dir / "sf_bq279.json").read_text(encoding="utf-8"))
    assert trace["intent"]["native_value_terms"] == [
        f"{BIKE_TABLE}.status=active",
        f"{BIKE_TABLE}.status=closed",
    ]
    assert trace["intent"]["derived_behavioral_definitions"] == []


def test_run_task_keeps_successful_critic_repair_on_score_tie(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    task = Task(instance_id="sf_local099", db="TEST_DB", question="Show customer totals.")
    run_paths = ensure_run_paths("critic-repair-tie-run", outputs_root=tmp_path)
    llm = FakeLLMClient(
        outputs={
            "intent": [
                Intent(
                    summary="Find customer totals.",
                    entities=["customers"],
                    metrics=["total amount"],
                    filters=[],
                    time_constraints=[],
                    output_expectation="customer and total columns",
                    assumptions=[],
                )
            ],
            "sql_generation": [
                SQLCandidate(
                    sql=f"SELECT customer, amount FROM {SALES_TABLE}",
                    explanation="Initial answer.",
                    assumptions=[],
                    confidence=0.9,
                )
            ],
            "result_critic": [
                ConfidenceReport(
                    confidence=0.9,
                    issues=["Use the repaired source."],
                    should_repair=True,
                    repair_focus="replace initial SQL",
                )
            ],
            "sql_repair": [
                SQLCandidate(
                    sql=f"SELECT customer, amount FROM {SALES_TABLE} /* fixed */",
                    explanation="Repaired answer.",
                    assumptions=[],
                    confidence=0.9,
                )
            ],
        }
    )

    monkeypatch.setattr(
        "sol01.coordinator.retrieve_schema",
        lambda *args, **kwargs: SchemaSelection(
            db="TEST_DB",
            selected_tables=[SALES_TABLE],
            expanded_tables=[SALES_TABLE],
            rationale="sales is enough",
            confidence=0.9,
        ),
    )
    monkeypatch.setattr("sol01.coordinator.load_db_index", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        "sol01.coordinator.fetch_query_dataframe",
        lambda sql, db: pd.DataFrame([{"customer": "bob", "amount": 12.0}]),
    )

    answer = run_task(
        task,
        run_paths=run_paths,
        config=RuntimeConfig(api_key="test-key"),
        llm_client=llm,
        initial_candidates=1,
    )

    trace = json.loads((run_paths.traces_dir / "sf_local099.json").read_text(encoding="utf-8"))
    assert answer.status == "success"
    assert trace["final_sql"].endswith("/* fixed */")


def test_attempt_score_prefers_output_shape_over_candidate_confidence():
    intent = Intent(
        summary="Count customer totals.",
        entities=["sales"],
        metrics=["count"],
        filters=[],
        time_constraints=[],
        output_expectation="one count",
        assumptions=[],
    )
    validation = ValidationReport(ok=True, errors=[], warnings=[], referenced_tables=["sales"])
    good_execution = ExecutionResult(
        ok=True,
        row_count=1,
        columns=["total"],
        sample_rows=[{"total": 12}],
        csv_path=None,
        error=None,
    )
    bad_execution = ExecutionResult(
        ok=True,
        row_count=1,
        columns=["customer", "total"],
        sample_rows=[{"customer": "bob", "total": 12}],
        csv_path=None,
        error=None,
    )

    good_score = _attempt_score(
        candidate=SQLCandidate(
            sql="SELECT COUNT(*) AS total FROM TEST_DB.PUBLIC.SALES",
            explanation="Scalar count.",
            assumptions=[],
            confidence=0.2,
        ),
        intent=intent,
        validation=validation,
        execution=good_execution,
        result_profile={
            "row_count": 1,
            "columns": ["total"],
            "sample_rows": [{"total": 12}],
        },
    )
    bad_score = _attempt_score(
        candidate=SQLCandidate(
            sql="SELECT customer, COUNT(*) AS total FROM TEST_DB.PUBLIC.SALES GROUP BY customer",
            explanation="Too many columns for the contract.",
            assumptions=[],
            confidence=0.99,
        ),
        intent=intent,
        validation=validation,
        execution=bad_execution,
        result_profile={
            "row_count": 1,
            "columns": ["customer", "total"],
            "sample_rows": [{"customer": "bob", "total": 12}],
        },
    )

    assert good_score > bad_score


def test_attempt_score_penalizes_ungrounded_filters_that_return_no_rows():
    intent = Intent(
        summary="Find matching countries.",
        entities=["countries"],
        metrics=["count"],
        filters=["country = 'Russia'"],
        time_constraints=[],
        output_expectation="one count",
        assumptions=[],
    )
    validation = ValidationReport(ok=True, errors=[], warnings=[], referenced_tables=["countries"])
    grounded_execution = ExecutionResult(
        ok=True,
        row_count=4,
        columns=["total"],
        sample_rows=[{"total": 4}],
        csv_path=None,
        error=None,
    )
    empty_execution = ExecutionResult(
        ok=True,
        row_count=0,
        columns=["total"],
        sample_rows=[],
        csv_path=None,
        error=None,
    )

    grounded_score = _attempt_score(
        candidate=SQLCandidate(
            sql=(
                "SELECT COUNT(*) AS total FROM TEST_DB.PUBLIC.COUNTRIES "
                "WHERE country = 'Russian Federation'"
            ),
            explanation="Uses the stored value variant.",
            assumptions=[],
            confidence=0.3,
        ),
        intent=intent,
        validation=validation,
        execution=grounded_execution,
        result_profile={
            "row_count": 4,
            "columns": ["total"],
            "sample_rows": [{"total": 4}],
        },
    )
    empty_score = _attempt_score(
        candidate=SQLCandidate(
            sql="SELECT COUNT(*) AS total FROM TEST_DB.PUBLIC.COUNTRIES WHERE country = 'Russia'",
            explanation="Exact string filter returns nothing.",
            assumptions=[],
            confidence=0.99,
        ),
        intent=intent,
        validation=validation,
        execution=empty_execution,
        result_profile={
            "row_count": 0,
            "columns": ["total"],
            "sample_rows": [],
        },
    )

    assert grounded_score > empty_score


def test_run_task_flags_missing_grouped_identifier_in_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    task = Task(
        instance_id="sf_local131",
        db="ENTERTAINMENTAGENCY",
        question=(
            "Could you list each musical style with the number of times it appears as a 1st, "
            "2nd, or 3rd preference in a single row per style?"
        ),
    )
    run_paths = ensure_run_paths("shape-run", outputs_root=tmp_path)
    captured_prompts: dict[str, list[str]] = {}
    styles_table = TableSchema(
        name="MUSICAL_STYLES",
        full_name="ENTERTAINMENTAGENCY.ENTERTAINMENTAGENCY.MUSICAL_STYLES",
        ddl='create table MUSICAL_STYLES ("StyleID" number, "StyleName" text);',
        columns=[
            ColumnSchema(name="StyleID"),
            ColumnSchema(name="StyleName"),
        ],
        searchable_text="musical styles styleid stylename",
    )
    preferences_table = TableSchema(
        name="MUSICAL_PREFERENCES",
        full_name="ENTERTAINMENTAGENCY.ENTERTAINMENTAGENCY.MUSICAL_PREFERENCES",
        ddl=('create table MUSICAL_PREFERENCES ("StyleID" number, "PreferenceSeq" number);'),
        columns=[
            ColumnSchema(name="StyleID"),
            ColumnSchema(name="PreferenceSeq"),
        ],
        searchable_text="musical preferences styleid preferenceseq",
    )
    llm = FakeLLMClient(
        outputs={
            "intent": [
                Intent(
                    summary="Count preferences by musical style.",
                    entities=["musical styles", "musical preferences"],
                    metrics=[
                        "count of 1st preference",
                        "count of 2nd preference",
                        "count of 3rd preference",
                    ],
                    filters=[],
                    time_constraints=[],
                    output_expectation=(
                        "One row per musical style with columns: musical style name, "
                        "count as 1st preference, count as 2nd preference, "
                        "count as 3rd preference."
                    ),
                    assumptions=[
                        "MUSICAL_PREFERENCES contains PreferenceSeq (1,2,3) and StyleID.",
                        (
                            "All musical styles should be included, with zero counts if "
                            "no preferences."
                        ),
                    ],
                )
            ],
            "sql_generation": [
                SQLCandidate(
                    sql=(
                        'SELECT ms."StyleName", '
                        'COUNT(CASE WHEN mp."PreferenceSeq" = 1 THEN 1 END) '
                        'AS "1stPreferenceCount", '
                        'COUNT(CASE WHEN mp."PreferenceSeq" = 2 THEN 1 END) '
                        'AS "2ndPreferenceCount", '
                        'COUNT(CASE WHEN mp."PreferenceSeq" = 3 THEN 1 END) '
                        'AS "3rdPreferenceCount" '
                        "FROM ENTERTAINMENTAGENCY.ENTERTAINMENTAGENCY.MUSICAL_STYLES ms "
                        "LEFT JOIN ENTERTAINMENTAGENCY.ENTERTAINMENTAGENCY.MUSICAL_PREFERENCES mp "
                        'ON ms."StyleID" = mp."StyleID" '
                        'GROUP BY ms."StyleID", ms."StyleName" '
                        'ORDER BY ms."StyleName"'
                    ),
                    explanation="Omit the grouped key from the output.",
                    assumptions=[],
                    confidence=0.95,
                ),
                SQLCandidate(
                    sql=(
                        'SELECT ms."StyleID", ms."StyleName", '
                        'COUNT(CASE WHEN mp."PreferenceSeq" = 1 THEN 1 END) '
                        'AS "1stPreferenceCount", '
                        'COUNT(CASE WHEN mp."PreferenceSeq" = 2 THEN 1 END) '
                        'AS "2ndPreferenceCount", '
                        'COUNT(CASE WHEN mp."PreferenceSeq" = 3 THEN 1 END) '
                        'AS "3rdPreferenceCount" '
                        "FROM ENTERTAINMENTAGENCY.ENTERTAINMENTAGENCY.MUSICAL_STYLES ms "
                        "LEFT JOIN ENTERTAINMENTAGENCY.ENTERTAINMENTAGENCY.MUSICAL_PREFERENCES mp "
                        'ON ms."StyleID" = mp."StyleID" '
                        'GROUP BY ms."StyleID", ms."StyleName" '
                        'ORDER BY ms."StyleName"'
                    ),
                    explanation="Keep the grouped key in the output.",
                    assumptions=[],
                    confidence=0.4,
                ),
            ],
            "result_comparison": [
                CandidateComparisonReport(
                    baseline_stage="initial_1",
                    preferred_stage="initial_2",
                    compared_stages=["initial_1", "initial_2"],
                    reasons=["The second candidate keeps the grouped key visible."],
                )
            ],
            "result_critic": [
                ConfidenceReport(confidence=0.9, issues=[], should_repair=False, repair_focus=None)
            ],
        },
        prompts=captured_prompts,
    )

    def fake_fetch_query_dataframe(sql: str, *, db: str) -> pd.DataFrame:
        if db != "ENTERTAINMENTAGENCY":
            raise AssertionError(f"Unexpected database: {db}")
        if 'SELECT ms."StyleID",' in sql:
            return pd.DataFrame(
                [
                    {
                        "StyleID": 1,
                        "StyleName": "40's Ballroom Music",
                        "1stPreferenceCount": 0,
                        "2ndPreferenceCount": 3,
                        "3rdPreferenceCount": 4,
                    },
                    {
                        "StyleID": 2,
                        "StyleName": "50's Music",
                        "1stPreferenceCount": 1,
                        "2ndPreferenceCount": 2,
                        "3rdPreferenceCount": 3,
                    },
                    {
                        "StyleID": 3,
                        "StyleName": "60's Music",
                        "1stPreferenceCount": 2,
                        "2ndPreferenceCount": 1,
                        "3rdPreferenceCount": 0,
                    },
                ]
            )
        return pd.DataFrame(
            [
                {
                    "StyleName": "40's Ballroom Music",
                    "1stPreferenceCount": 0,
                    "2ndPreferenceCount": 3,
                    "3rdPreferenceCount": 4,
                },
                {
                    "StyleName": "50's Music",
                    "1stPreferenceCount": 1,
                    "2ndPreferenceCount": 2,
                    "3rdPreferenceCount": 3,
                },
                {
                    "StyleName": "60's Music",
                    "1stPreferenceCount": 2,
                    "2ndPreferenceCount": 1,
                    "3rdPreferenceCount": 0,
                },
            ]
        )

    monkeypatch.setattr("sol01.coordinator.fetch_query_dataframe", fake_fetch_query_dataframe)
    monkeypatch.setattr(
        "sol01.coordinator.retrieve_schema",
        lambda *args, **kwargs: SchemaSelection(
            db="ENTERTAINMENTAGENCY",
            selected_tables=[
                styles_table.full_name or styles_table.name,
                preferences_table.full_name or preferences_table.name,
            ],
            expanded_tables=[
                styles_table.full_name or styles_table.name,
                preferences_table.full_name or preferences_table.name,
            ],
            rationale="Musical styles and preferences are enough.",
            confidence=0.95,
        ),
    )
    monkeypatch.setattr(
        "sol01.coordinator.load_db_index",
        lambda *args, **kwargs: {
            styles_table.full_name or styles_table.name: styles_table,
            preferences_table.full_name or preferences_table.name: preferences_table,
        },
    )

    answer = run_task(
        task,
        run_paths=run_paths,
        config=RuntimeConfig(api_key="test-key"),
        llm_client=llm,
        initial_candidates=2,
    )

    assert answer.status == "success"
    trace = json.loads((run_paths.traces_dir / "sf_local131.json").read_text(encoding="utf-8"))
    assert trace["final_sql"].startswith('SELECT ms."StyleID", ms."StyleName"')
    assert trace["attempts"][0]["shape_report"]["expected_columns"] == [
        "StyleName",
        "1stPreferenceCount",
        "2ndPreferenceCount",
        "3rdPreferenceCount",
        "StyleID",
    ]
    assert "missing grouped key StyleID" in trace["attempts"][0]["shape_report"]["violations"]
    assert (
        trace["attempts"][0]["score_breakdown"]["shape"]
        < trace["attempts"][1]["score_breakdown"]["shape"]
    )
    assert "StyleID" not in trace["attempts"][0]["result_profile"]["columns"]
    assert trace["attempts"][1]["shape_report"]["violations"] == []


def test_run_task_without_external_knowledge_uses_no_document_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_snowflake: None
):
    task = Task(
        instance_id="sf_local999",
        db="TEST_DB",
        question="Show customer totals.",
        external_knowledge=None,
    )
    run_paths = ensure_run_paths("no-external-knowledge-run", outputs_root=tmp_path)
    captured_prompts: dict[str, list[str]] = {}
    llm = FakeLLMClient(
        outputs={
            "intent": [
                Intent(
                    summary="Find customer totals.",
                    entities=["sales"],
                    metrics=["frequency"],
                    filters=[],
                    time_constraints=[],
                    output_expectation="customer and total columns",
                    assumptions=["Use all rows."],
                )
            ],
            "sql_generation": [
                SQLCandidate(
                    sql=f"SELECT customer, amount FROM {SALES_TABLE} ORDER BY amount DESC",
                    explanation="Read customer amounts directly.",
                    assumptions=["amount already stores totals"],
                    confidence=0.8,
                )
            ],
            "result_critic": [
                ConfidenceReport(confidence=0.9, issues=[], should_repair=False, repair_focus=None)
            ],
        },
        prompts=captured_prompts,
    )

    monkeypatch.setattr(
        "sol01.coordinator.retrieve_schema",
        lambda *args, **kwargs: SchemaSelection(
            db="test_db",
            selected_tables=[SALES_TABLE],
            expanded_tables=[SALES_TABLE],
            rationale="sales is enough",
            confidence=0.9,
        ),
    )

    answer = run_task(
        task,
        run_paths=run_paths,
        config=RuntimeConfig(api_key="test-key"),
        llm_client=llm,
        initial_candidates=1,
    )

    assert answer.status == "success"
    intent_prompt = captured_prompts["intent"][0]
    assert "Document context:\nNo task-linked document context." in intent_prompt
    sql_prompt = captured_prompts["sql_generation"][0]
    assert "Document context:\nNo task-linked document context." in sql_prompt


def test_critic_prompt_includes_answer_contract_and_candidate_assumptions():
    task = Task(
        instance_id="sf_bq062",
        db="DEPS_DEV_V1",
        question="What is the most frequently used license by packages in each system?",
    )
    intent = Intent(
        summary="Find the most frequent license per system.",
        entities=["packages", "systems", "licenses"],
        metrics=["count of packages by license"],
        filters=[],
        native_value_terms=["DEPS_DEV_V1.DEPS_DEV_V1.PACKAGEVERSIONS.System=active"],
        derived_behavioral_definitions=["active packages means packages with recent updates"],
        time_constraints=[],
        answer_grain="one row per package system",
        requested_ordering=["rank licenses within each system by package count"],
        output_expectation="system, license, package count",
        assumptions=[],
        evidence=["Question asks by packages in each system."],
        unsupported_assumptions=["Latest snapshot is not stated."],
        do_not_assume=["Do not restrict to the latest snapshot."],
    )
    attempt = {
        "sql": 'SELECT "System", license FROM DEPS_DEV_V1.DEPS_DEV_V1.PACKAGEVERSIONS',
        "assumptions": ["Use latest snapshot."],
        "constraint_ledger": ["Filter to MAX(SnapshotAt) as latest data."],
        "unsupported_assumptions": ["Latest snapshot was chosen from schema, not task text."],
        "execution_result": {"ok": True, "row_count": 2},
        "result_profile": {"row_count": 2, "columns": ["System", "LICENSE"]},
    }

    prompt = _critic_prompt(
        task,
        intent,
        attempt,
        "SQL reference context:\nDatabase: DEPS_DEV_V1",
        "No task-linked document context.",
    )

    assert "Answer contract:" in prompt
    assert "Latest snapshot is not stated." in prompt
    assert "native_value_terms" in prompt
    assert "active packages means packages with recent updates" in prompt
    assert "Candidate constraint ledger:" in prompt
    assert "Filter to MAX(SnapshotAt) as latest data." in prompt
    assert "Candidate unsupported assumptions:" in prompt


def test_semantic_repair_prompt_rederives_from_original_task_contract():
    task = Task(
        instance_id="sf_bq062",
        db="DEPS_DEV_V1",
        question="What is the most frequently used license by packages in each system?",
    )
    intent = Intent(
        summary="Find the most frequent license per system.",
        entities=["packages", "systems", "licenses"],
        metrics=["count of packages by license"],
        filters=[],
        native_value_terms=["DEPS_DEV_V1.DEPS_DEV_V1.PACKAGEVERSIONS.System=active"],
        derived_behavioral_definitions=["active packages means packages with recent updates"],
        time_constraints=[],
        answer_grain="one row per system",
        output_expectation="system, license, package count",
        unsupported_assumptions=["Latest snapshot is not stated."],
        do_not_assume=["Do not restrict to current/latest rows without task evidence."],
    )
    attempt = {
        "sql": 'SELECT "System" FROM DEPS_DEV_V1.DEPS_DEV_V1.PACKAGEVERSIONS',
        "assumptions": ["Use latest snapshot."],
        "constraint_ledger": ["Filter to MAX(SnapshotAt)."],
        "unsupported_assumptions": ["Latest snapshot was ungrounded."],
    }
    critic = ConfidenceReport(
        confidence=0.9,
        issues=["SQL adds an ungrounded latest snapshot filter."],
        should_repair=True,
        repair_focus="Remove ungrounded narrowing and answer from the task text.",
    )

    prompt = _semantic_repair_prompt(
        task,
        intent,
        attempt,
        critic,
        "SQL reference context:\nDatabase: DEPS_DEV_V1",
        "No task-linked document context.",
    )

    assert "Question:" in prompt
    assert "Current answer contract:" in prompt
    assert "Do not restrict to current/latest rows without task evidence." in prompt
    assert "native_value_terms" in prompt
    assert "Candidate constraint ledger:" in prompt
    assert "SQL adds an ungrounded latest snapshot filter." in prompt
