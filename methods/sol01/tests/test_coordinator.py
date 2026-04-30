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
    _sql_generation_prompt,
    _sql_reference_context,
    _sql_repair_prompt,
    run_task,
    run_tasks,
)
from sol01.llm import PromptSpec
from sol01.models import (
    ColumnSchema,
    ConfidenceReport,
    FinalAnswer,
    Intent,
    SchemaSelection,
    SQLCandidate,
    TableSchema,
    Task,
)
from sol01.output import ensure_run_paths
from sol01.retrieval import load_db_index

SALES_TABLE = "TEST_DB.PUBLIC.SALES"
DICOM_TEST_TABLE = "TEST_DB.PUBLIC.DICOM_PIVOT"


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


def test_sql_repair_prompt_keeps_reference_context_before_failed_sql():
    task = Task(instance_id="sf_local001", db="TEST_DB", question="Show customer totals.")
    attempt = {
        "sql": f"SELECT missing_column FROM {SALES_TABLE}",
        "validation": {"ok": True, "errors": [], "warnings": []},
        "execution_result": {"ok": False, "error": "invalid identifier"},
    }

    prompt = _sql_repair_prompt(
        task,
        attempt,
        "SQL reference context:\nDatabase: TEST_DB\nSelected tables:\n- TEST_DB.PUBLIC.SALES",
        "No task-linked document context.",
    )

    assert prompt.startswith("SQL reference context:")
    assert prompt.index("Question:") < prompt.index("Failed SQL:")
    assert prompt.index("Failed SQL:") < prompt.index("Validation:")


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
    sql_prompt = captured_prompts["sql_generation"][0]
    assert "Document context:\nWHOLE DOC FOR RFM.md" in sql_prompt


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
    sql_prompt = captured_prompts["sql_generation"][0]
    assert "Document context:\nNo task-linked document context." in sql_prompt
