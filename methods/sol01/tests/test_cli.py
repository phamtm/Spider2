"""Tests for CLI command wiring and argument parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from sol01 import cli
from sol01.config import RuntimeConfig
from sol01.models import FinalAnswer, Task
from sol01.output import AskPaths, ensure_run_paths, eval_input_csv_dir_for

runner = CliRunner()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


@dataclass
class DummyAnswer:
    """Small return object for the ask command test."""

    status: str
    csv_path: str | None = None


def test_index_command_dispatches(monkeypatch):
    """The index command should call the index handler once."""

    called: dict[str, Any] = {}

    def fake_handle_index() -> dict[str, Any]:
        called["ok"] = True
        return {"db_one": {}}

    monkeypatch.setattr(cli, "handle_index", fake_handle_index)

    result = runner.invoke(cli.app, ["index"])

    assert result.exit_code == 0
    assert called == {"ok": True}
    assert "Indexed" in result.output


def test_run_command_dispatches_expected_filters(monkeypatch):
    """The run command should pass through the task filters and run ID."""

    called: dict[str, Any] = {}

    def fake_handle_run(**kwargs: Any) -> list[Any]:
        called.update(kwargs)
        return {
            "tasks": [Task(instance_id="local003", db="db", question="Question text")],
            "results": [
                FinalAnswer(
                    instance_id="local003",
                    status="success",
                    sql="SELECT 1",
                    csv_path="out.csv",
                    trace_path="trace.json",
                )
            ],
            "eval_summary": {
                "correct_tasks": 1,
                "attempted_tasks": 1,
                "missing_csv_count": 0,
                "per_instance": [
                    {"instance_id": "local003", "passed": True, "score": 1, "csv_present": True}
                ],
            },
        }

    monkeypatch.setattr(cli, "handle_run", fake_handle_run)

    result = runner.invoke(
        cli.app,
        ["run", "--instance-id", "local003", "--run-id", "smoke-local003"],
    )

    assert result.exit_code == 0
    assert called == {
        "run_id": "smoke-local003",
        "instance_id": "local003",
        "db": None,
        "question_contains": None,
        "limit": None,
        "force": False,
        "skip_failed": False,
    }
    assert "Eval summary: 1/1 correct, missing CSV 0" in result.output
    assert "Exec time:" in result.output
    assert "- local003: PASS | task success | Question text" in result.output


def test_run_command_forwards_explicit_concurrency(monkeypatch):
    """The run command should forward an explicit concurrency value."""

    called: dict[str, Any] = {}

    def fake_handle_run(**kwargs: Any) -> dict[str, Any]:
        called.update(kwargs)
        return {
            "tasks": [Task(instance_id="local003", db="db", question="Question text")],
            "results": [
                FinalAnswer(
                    instance_id="local003",
                    status="success",
                    sql="SELECT 1",
                    csv_path="out.csv",
                    trace_path="trace.json",
                )
            ],
            "eval_summary": {
                "correct_tasks": 1,
                "attempted_tasks": 1,
                "missing_csv_count": 0,
                "per_instance": [
                    {"instance_id": "local003", "passed": True, "score": 1, "csv_present": True}
                ],
            },
        }

    monkeypatch.setattr(cli, "handle_run", fake_handle_run)

    result = runner.invoke(
        cli.app,
        [
            "run",
            "--instance-id",
            "local003",
            "--run-id",
            "smoke-local003",
            "--concurrency",
            "8",
        ],
    )

    assert result.exit_code == 0
    assert called["concurrency"] == 8


def test_handle_run_passes_default_dotenv_path(monkeypatch):
    """The run handler should opt into the method-local dotenv file."""

    called: dict[str, Any] = {}

    monkeypatch.setattr(
        cli,
        "_load_filtered_tasks",
        lambda **kwargs: [Task(instance_id="local003", db="db", question="q")],
    )

    def fake_from_env(cls, require_api_key=False, dotenv_path=None, concurrency=None):
        called["require_api_key"] = require_api_key
        called["dotenv_path"] = dotenv_path
        called["concurrency"] = concurrency
        return RuntimeConfig(api_key="test-key")

    def fake_run_tasks(tasks, *, run_id, config, force, skip_failed):
        called["task_ids"] = [task.instance_id for task in tasks]
        called["run_id"] = run_id
        return [
            FinalAnswer(
                instance_id="local003",
                status="success",
                sql="SELECT 1",
                csv_path="out.csv",
                trace_path="trace.json",
            )
        ]

    monkeypatch.setattr(cli.RuntimeConfig, "from_env", classmethod(fake_from_env))
    monkeypatch.setattr(cli, "run_tasks", fake_run_tasks)

    def fake_run_persisted_eval(run_id, *, expected_instance_ids=None, **kwargs):
        called["eval_run_id"] = run_id
        called["expected_instance_ids"] = expected_instance_ids
        return {
            "correct_tasks": 1,
            "attempted_tasks": 1,
            "missing_csv_count": 0,
            "result_dir": f"/persisted/{run_id}/eval/scored_csv",
            "per_instance": [
                {
                    "instance_id": "local003",
                    "passed": True,
                    "score": 1,
                    "csv_present": True,
                    "failure_reason": None,
                }
            ],
        }

    monkeypatch.setattr(cli, "run_persisted_eval", fake_run_persisted_eval)

    result = cli.handle_run(
        concurrency=None,
        run_id="smoke-local003",
        instance_id="local003",
        db=None,
        question_contains=None,
        limit=None,
        force=False,
        skip_failed=False,
    )

    assert called["require_api_key"] is True
    assert called["dotenv_path"] == cli.DEFAULT_DOTENV_PATH
    assert called["concurrency"] is None
    assert called["task_ids"] == ["local003"]
    assert called["run_id"] == "smoke-local003"
    assert called["eval_run_id"] == "smoke-local003"
    assert called["expected_instance_ids"] == ["local003"]
    assert result["eval_summary"]["correct_tasks"] == 1


def test_handle_run_forwards_all_expected_ids_to_persisted_eval(monkeypatch, tmp_path: Path):
    """Post-run eval should account for every selected task, even missing CSVs."""

    called: dict[str, Any] = {}

    monkeypatch.setattr(
        cli,
        "_load_filtered_tasks",
        lambda **kwargs: [
            Task(instance_id="local003", db="db", question="q1"),
            Task(instance_id="local004", db="db", question="q2"),
        ],
    )
    monkeypatch.setattr(
        cli.RuntimeConfig,
        "from_env",
        classmethod(
            lambda cls, require_api_key=False, dotenv_path=None, concurrency=None: RuntimeConfig(
                api_key="k"
            )
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_tasks",
        lambda *args, **kwargs: [
            FinalAnswer(
                instance_id="local003",
                status="success",
                sql="SELECT 1",
                csv_path=str(tmp_path / "local003.csv"),
                trace_path="trace-1.json",
            ),
            FinalAnswer(
                instance_id="local004",
                status="failed",
                sql="SELECT 2",
                csv_path=None,
                trace_path="trace-2.json",
            ),
        ],
    )

    def fake_run_persisted_eval(run_id, *, expected_instance_ids=None, **kwargs):
        called["expected_instance_ids"] = expected_instance_ids
        return {
            "correct_tasks": 1,
            "attempted_tasks": 1,
            "missing_csv_count": 1,
            "per_instance": [],
        }

    monkeypatch.setattr(cli, "run_persisted_eval", fake_run_persisted_eval)

    cli.handle_run(
        concurrency=None,
        run_id="smoke-local003",
        instance_id=None,
        db=None,
        question_contains=None,
        limit=None,
        force=False,
        skip_failed=False,
    )

    assert called["expected_instance_ids"] == ["local003", "local004"]


def test_eval_command_dispatches_filters(monkeypatch):
    """The eval command should pass through its debug filters."""

    called: dict[str, Any] = {}

    def fake_handle_eval(**kwargs: Any) -> dict[str, Any]:
        called.update(kwargs)
        return {
            "correct_tasks": 1,
            "attempted_tasks": 1,
            "missing_csv_count": 0,
        }

    monkeypatch.setattr(cli, "handle_eval", fake_handle_eval)

    result = runner.invoke(
        cli.app,
        ["eval", "--run-id", "smoke-local003", "--instance-id", "local003"],
    )

    assert result.exit_code == 0
    assert called == {
        "run_id": "smoke-local003",
        "instance_id": "local003",
        "db": None,
        "question_contains": None,
        "limit": None,
    }


def test_handle_eval_passes_filtered_ids_without_rewriting_manifest(monkeypatch, tmp_path: Path):
    """Filtered eval should pass expected IDs directly instead of mutating the manifest."""

    called: dict[str, Any] = {}
    run_paths = ensure_run_paths("smoke-local003", outputs_root=tmp_path)
    run_paths.manifest_path.write_text(
        '{"task_ids": ["local003", "local004"]}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        cli,
        "_load_filtered_tasks",
        lambda **kwargs: [Task(instance_id="local003", db="db", question="q")],
    )
    monkeypatch.setattr(cli, "ensure_run_paths", lambda run_id: run_paths)
    (run_paths.csv_dir / "local003.csv").write_text("answer\n1\n", encoding="utf-8")
    (run_paths.csv_dir / "local004.csv").write_text("answer\n2\n", encoding="utf-8")

    def fake_run_official_eval(
        run_id: str,
        *,
        expected_instance_ids: list[str] | None = None,
        artifact_tag: str | None = None,
        result_dir: Path | None = None,
    ) -> dict[str, Any]:
        called["run_id"] = run_id
        called["expected_instance_ids"] = expected_instance_ids
        called["artifact_tag"] = artifact_tag
        called["result_dir_files"] = sorted(path.name for path in result_dir.iterdir())
        return {
            "correct_tasks": 1,
            "attempted_tasks": 1,
            "missing_csv_count": 0,
        }

    monkeypatch.setattr(cli, "run_official_eval", fake_run_official_eval)

    summary = cli.handle_eval(
        run_id="smoke-local003",
        instance_id="local003",
        db=None,
        question_contains=None,
        limit=None,
    )

    assert summary["correct_tasks"] == 1
    assert called["result_dir_files"] == ["local003.csv"]
    artifact_tag = cli._filtered_eval_tag(
        instance_id="local003",
        db=None,
        question_contains=None,
        limit=None,
    )
    assert called == {
        "run_id": "smoke-local003",
        "expected_instance_ids": ["local003"],
        "artifact_tag": artifact_tag,
        "result_dir_files": ["local003.csv"],
    }
    assert (
        run_paths.manifest_path.read_text(encoding="utf-8")
        == '{"task_ids": ["local003", "local004"]}\n'
    )
    assert (
        eval_input_csv_dir_for(run_paths, eval_id=artifact_tag)
        == run_paths.eval_dir / "runs" / artifact_tag / "input_csv"
    )


def test_handle_eval_rejects_missing_filtered_csv(monkeypatch, tmp_path: Path):
    """Filtered eval should fail fast when a requested CSV is missing."""

    run_paths = ensure_run_paths("smoke-local003", outputs_root=tmp_path)
    monkeypatch.setattr(cli, "ensure_run_paths", lambda run_id: run_paths)
    monkeypatch.setattr(
        cli,
        "_load_filtered_tasks",
        lambda **kwargs: [Task(instance_id="local003", db="db", question="q")],
    )

    with pytest.raises(typer.BadParameter) as exc_info:
        cli.handle_eval(
            run_id="smoke-local003",
            instance_id="local003",
            db=None,
            question_contains=None,
            limit=None,
        )
    assert "Missing CSV for local003" in str(exc_info.value)
    assert exc_info.value.param_hint == "--run-id"


def test_filtered_eval_tag_adds_a_disambiguating_hash():
    """Different raw filters should not collapse to the same artifact tag."""

    first = cli._filtered_eval_tag(
        instance_id=None,
        db=None,
        question_contains="A B",
        limit=None,
    )
    second = cli._filtered_eval_tag(
        instance_id=None,
        db=None,
        question_contains="A-B",
        limit=None,
    )

    assert first != second


def test_analyze_command_dispatches(monkeypatch):
    """The analyze command should call the analysis handler."""

    called: dict[str, Any] = {}

    def fake_handle_analyze(**kwargs: Any) -> dict[str, Any]:
        called.update(kwargs)
        return {
            "trace_count": 3,
            "status_counts": {"success": 1, "failed": 2, "skipped": 0},
        }

    monkeypatch.setattr(cli, "handle_analyze", fake_handle_analyze)

    result = runner.invoke(cli.app, ["analyze", "--run-id", "smoke-local003"])

    assert result.exit_code == 0
    assert called == {"run_id": "smoke-local003"}


def test_llm_calls_command_summarizes_rows(monkeypatch, tmp_path: Path):
    """The llm-calls command should print a compact summary by default."""

    monkeypatch.setattr(cli, "OUTPUTS_ROOT", tmp_path)
    log_path = tmp_path / "smoke-local003" / "llm_calls" / "local003.jsonl"
    _write_jsonl(
        log_path,
        [
            {
                "sequence": 1,
                "call_id": "0001-intent",
                "prompt_name": "intent",
                "status": "success",
                "duration_ms": 1000,
                "model": "deepseek/deepseek-v4-pro",
                "attempts": [{"status": "success"}],
                "error": None,
            },
            {
                "sequence": 2,
                "call_id": "0002-sql_generation",
                "prompt_name": "sql_generation",
                "status": "error",
                "duration_ms": 3000,
                "model": "deepseek/deepseek-v4-pro",
                "attempts": [{"status": "error"}],
                "error": {"type": "ModelHTTPError", "message": "bad request", "status_code": 400},
            },
        ],
    )

    result = runner.invoke(
        cli.app,
        [
            "llm-calls",
            "--run-id",
            "smoke-local003",
            "--instance-id",
            "local003",
        ],
    )

    assert result.exit_code == 0
    expected_first = "- 1: intent | success | 1s | deepseek/deepseek-v4-pro | 1 attempt | no error"
    expected_second = "- 2: sql_generation | error | 3s | deepseek/deepseek-v4-pro | 1 attempt |"
    assert expected_first in result.output
    assert expected_second in result.output
    assert "ModelHTTPError" in result.output


def test_llm_calls_command_prints_full_selected_call(monkeypatch, tmp_path: Path):
    """The llm-calls command should print the full payload for one selected call."""

    monkeypatch.setattr(cli, "OUTPUTS_ROOT", tmp_path)
    log_path = tmp_path / "smoke-local003" / "llm_calls" / "local003.jsonl"
    _write_jsonl(
        log_path,
        [
            {
                "sequence": 1,
                "call_id": "0001-intent",
                "prompt_name": "intent",
                "status": "success",
                "duration_ms": 1000,
                "model": "deepseek/deepseek-v4-pro",
                "request": {
                    "system_prompt": "# Sample Prompt\n\nDo the thing.",
                    "user_prompt": "Question one.",
                    "output_schema": "SampleOutput",
                },
                "response": {"validated_output": {"value": "hello"}},
                "attempts": [{"status": "success"}],
                "error": None,
            },
            {
                "sequence": 2,
                "call_id": "0002-sql_generation",
                "prompt_name": "sql_generation",
                "status": "error",
                "duration_ms": 3000,
                "model": "deepseek/deepseek-v4-pro",
                "request": {
                    "system_prompt": "# SQL Prompt",
                    "user_prompt": "Question two.",
                    "output_schema": "SqlOutput",
                },
                "response": None,
                "attempts": [{"status": "error", "error": {"status_code": 400}}],
                "error": {"type": "ModelHTTPError", "message": "bad request", "status_code": 400},
            },
        ],
    )

    result = runner.invoke(
        cli.app,
        [
            "llm-calls",
            "--run-id",
            "smoke-local003",
            "--instance-id",
            "local003",
            "--call-id",
            "0001-intent",
        ],
    )

    assert result.exit_code == 0
    assert "System prompt:" in result.output
    assert "# Sample Prompt" in result.output
    assert "User prompt:" in result.output
    assert "Question one." in result.output
    assert "Validated response:" in result.output
    assert '"value": "hello"' in result.output
    assert "Attempts:" in result.output
    assert '"status": "success"' in result.output
    assert "Error:" in result.output
    assert "—" in result.output


def test_llm_calls_command_exits_cleanly_when_log_is_missing(monkeypatch, tmp_path: Path):
    """The llm-calls command should fail with a useful message when no log exists."""

    monkeypatch.setattr(cli, "OUTPUTS_ROOT", tmp_path)

    result = runner.invoke(
        cli.app,
        [
            "llm-calls",
            "--run-id",
            "smoke-local003",
            "--instance-id",
            "local003",
        ],
    )

    assert result.exit_code == 1
    assert "No usable LLM call rows were found" in result.output


def test_ask_command_dispatches(monkeypatch):
    """The ask command should pass db and question to the ask handler."""

    called: dict[str, Any] = {}

    def fake_handle_ask(**kwargs: Any) -> DummyAnswer:
        called.update(kwargs)
        return DummyAnswer(status="success", csv_path="/tmp/result.csv")

    monkeypatch.setattr(cli, "handle_ask", fake_handle_ask)

    result = runner.invoke(
        cli.app,
        ["ask", "--db", "E_commerce", "Which customers have the highest AOV?"],
    )

    assert result.exit_code == 0
    assert called == {
        "db": "E_commerce",
        "question": "Which customers have the highest AOV?",
    }
    assert "Ask status: success" in result.output


def test_handle_ask_uses_ask_layout(monkeypatch, tmp_path: Path):
    """Ask should use the dedicated ask paths and move final artifacts there."""

    ask_paths = AskPaths(
        root=tmp_path / "ask" / "20260427T120000.000000Z",
        sql_path=tmp_path / "ask" / "20260427T120000.000000Z" / "sql.sql",
        csv_path=tmp_path / "ask" / "20260427T120000.000000Z" / "result.csv",
        trace_path=tmp_path / "ask" / "20260427T120000.000000Z" / "trace.json",
    )
    ask_paths.root.mkdir(parents=True, exist_ok=True)
    called: dict[str, Any] = {}

    def fake_from_env(cls, require_api_key=False, dotenv_path=None, concurrency=None):
        called["require_api_key"] = require_api_key
        called["dotenv_path"] = dotenv_path
        called["concurrency"] = concurrency
        return object()

    monkeypatch.setattr(cli.RuntimeConfig, "from_env", classmethod(fake_from_env))
    monkeypatch.setattr(cli, "ensure_ask_paths", lambda outputs_root: ask_paths)

    def fake_run_task(task: Task, *, run_paths, config, force: bool):
        (run_paths.sql_dir / "ask.sql").parent.mkdir(parents=True, exist_ok=True)
        (run_paths.csv_dir / "ask.csv").parent.mkdir(parents=True, exist_ok=True)
        (run_paths.traces_dir / "ask.json").parent.mkdir(parents=True, exist_ok=True)
        (run_paths.sql_dir / "ask.sql").write_text("SELECT 1\n", encoding="utf-8")
        (run_paths.csv_dir / "ask.csv").write_text("answer\n1\n", encoding="utf-8")
        (run_paths.traces_dir / "ask.json").write_text('{"status":"success"}\n', encoding="utf-8")
        return FinalAnswer(
            instance_id=task.instance_id,
            status="success",
            sql="SELECT 1",
            csv_path=str(run_paths.csv_dir / "ask.csv"),
            trace_path=str(run_paths.traces_dir / "ask.json"),
        )

    monkeypatch.setattr(cli, "run_task", fake_run_task)

    answer = cli.handle_ask(db="E_commerce", question="Which customers have the highest AOV?")

    assert called["require_api_key"] is True
    assert called["dotenv_path"] == cli.DEFAULT_DOTENV_PATH
    assert called["concurrency"] is None
    assert answer.status == "success"
    assert answer.csv_path == str(ask_paths.csv_path)
    assert answer.trace_path == str(ask_paths.trace_path)
    assert ask_paths.sql_path.read_text(encoding="utf-8") == "SELECT 1\n"
    assert ask_paths.csv_path.read_text(encoding="utf-8") == "answer\n1\n"
    assert ask_paths.trace_path.read_text(encoding="utf-8") == '{"status":"success"}\n'


def test_handle_ask_cleans_up_internal_dir_on_failure(monkeypatch, tmp_path: Path):
    """A failed ask run should not leave the internal staging directory behind."""

    ask_paths = AskPaths(
        root=tmp_path / "ask" / "20260427T120000.000000Z",
        sql_path=tmp_path / "ask" / "20260427T120000.000000Z" / "sql.sql",
        csv_path=tmp_path / "ask" / "20260427T120000.000000Z" / "result.csv",
        trace_path=tmp_path / "ask" / "20260427T120000.000000Z" / "trace.json",
    )
    ask_paths.root.mkdir(parents=True, exist_ok=True)
    called: dict[str, Any] = {}

    def fake_from_env(cls, require_api_key=False, dotenv_path=None, concurrency=None):
        called["require_api_key"] = require_api_key
        called["dotenv_path"] = dotenv_path
        called["concurrency"] = concurrency
        return object()

    monkeypatch.setattr(cli.RuntimeConfig, "from_env", classmethod(fake_from_env))
    monkeypatch.setattr(cli, "ensure_ask_paths", lambda outputs_root: ask_paths)

    def failing_run_task(task: Task, *, run_paths, config, force: bool):
        run_paths.root.mkdir(parents=True, exist_ok=True)
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "run_task", failing_run_task)

    try:
        cli.handle_ask(db="E_commerce", question="Which customers have the highest AOV?")
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("Expected RuntimeError")

    assert called["require_api_key"] is True
    assert called["dotenv_path"] == cli.DEFAULT_DOTENV_PATH
    assert called["concurrency"] is None
    assert not (ask_paths.root / "_internal").exists()
