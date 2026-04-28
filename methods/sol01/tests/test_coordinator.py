"""Tests for the per-task execution loop."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from sol01.config import RuntimeConfig
from sol01.coordinator import run_task
from sol01.llm import PromptSpec
from sol01.models import (
    ConfidenceReport,
    FinalAnswer,
    Intent,
    SchemaSelection,
    SQLCandidate,
    Task,
)
from sol01.output import ensure_run_paths


@dataclass
class FakeLLMClient:
    """Minimal fake LLM that returns queued outputs by prompt name."""

    outputs: dict[str, list[Any]]

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
def temp_db(tmp_path: Path) -> Path:
    """Create a tiny SQLite database for coordinator tests."""

    path = tmp_path / "sample.sqlite"
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE sales (
                id INTEGER PRIMARY KEY,
                customer TEXT,
                amount REAL
            )
            """
        )
        connection.executemany(
            "INSERT INTO sales (customer, amount) VALUES (?, ?)",
            [
                ("alice", 10.5),
                ("bob", 12.0),
                ("carol", 7.5),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return path


def test_run_task_success_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, temp_db: Path):
    task = Task(instance_id="local003", db="test_db", question="Show customer totals.")
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
                    sql="SELECT customer, amount FROM sales ORDER BY amount DESC",
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
            selected_tables=["sales"],
            expanded_tables=["sales"],
            rationale="sales is enough",
            confidence=0.9,
        ),
    )

    answer = run_task(
        task,
        run_paths=run_paths,
        config=RuntimeConfig(api_key="test-key"),
        llm_client=llm,
        db_path=temp_db,
        initial_candidates=1,
    )

    assert isinstance(answer, FinalAnswer)
    assert answer.status == "success"
    assert answer.csv_path is not None
    assert Path(answer.csv_path).exists()
    assert (run_paths.sql_dir / "local003.sql").exists()

    trace = json.loads((run_paths.traces_dir / "local003.json").read_text(encoding="utf-8"))
    assert trace["status"] == "success"
    assert trace["retrieval_mode"] == "llm_only"
    assert trace["prompt_hashes"]["intent"] == "hash-intent"
    assert len(trace["attempts"]) == 1
    assert trace["attempts"][0]["validation"]["ok"] is True
    assert trace["attempts"][0]["execution_result"]["ok"] is True


def test_run_task_repairs_validation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, temp_db: Path
):
    task = Task(instance_id="local004", db="test_db", question="Show the first customer.")
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
                    sql="SELECT customer FROM sales ORDER BY id LIMIT 1",
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
            selected_tables=["sales"],
            expanded_tables=["sales"],
            rationale="sales is enough",
            confidence=0.9,
        ),
    )

    answer = run_task(
        task,
        run_paths=run_paths,
        config=RuntimeConfig(api_key="test-key"),
        llm_client=llm,
        db_path=temp_db,
        initial_candidates=1,
    )

    assert answer.status == "success"

    trace = json.loads((run_paths.traces_dir / "local004.json").read_text(encoding="utf-8"))
    assert trace["status"] == "success"
    assert len(trace["attempts"]) == 2
    assert trace["attempts"][0]["validation"]["ok"] is False
    assert trace["attempts"][1]["execution_result"]["ok"] is True


def test_run_task_failure_writes_trace_without_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, temp_db: Path
):
    task = Task(instance_id="local005", db="test_db", question="Show a missing table.")
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
            selected_tables=["sales"],
            expanded_tables=["sales"],
            rationale="sales is enough",
            confidence=0.9,
        ),
    )

    answer = run_task(
        task,
        run_paths=run_paths,
        config=RuntimeConfig(api_key="test-key"),
        llm_client=llm,
        db_path=temp_db,
        initial_candidates=1,
    )

    assert answer.status == "failed"
    assert answer.csv_path is None
    assert not (run_paths.csv_dir / "local005.csv").exists()

    trace = json.loads((run_paths.traces_dir / "local005.json").read_text(encoding="utf-8"))
    assert trace["status"] == "failed"
    assert trace["csv_path"] is None
    assert len(trace["attempts"]) == 2


def test_run_task_catches_execution_errors_and_writes_failed_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, temp_db: Path
):
    task = Task(instance_id="local006", db="test_db", question="Select a missing column.")
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
                    sql="SELECT missing_column FROM sales",
                    explanation="Runtime error path.",
                    assumptions=[],
                    confidence=0.9,
                )
            ],
            "sql_repair": [
                SQLCandidate(
                    sql="SELECT still_missing FROM sales",
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
            selected_tables=["sales"],
            expanded_tables=["sales"],
            rationale="sales is enough",
            confidence=0.9,
        ),
    )

    answer = run_task(
        task,
        run_paths=run_paths,
        config=RuntimeConfig(api_key="test-key"),
        llm_client=llm,
        db_path=temp_db,
        initial_candidates=1,
    )

    assert answer.status == "failed"
    trace = json.loads((run_paths.traces_dir / "local006.json").read_text(encoding="utf-8"))
    assert trace["status"] == "failed"
    assert "no such column" in trace["attempts"][0]["execution_result"]["error"].lower()
