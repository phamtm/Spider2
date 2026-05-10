"""Tests for the single-question persisted gold runner."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import sol01.analysis.gold_run as gold_run
from sol01.models import ExecutionResult
from sol01.output.output import (
    csv_path_for,
    eval_input_csv_dir_for,
    eval_workspace_suite_dir_for,
    run_log_path_for,
    stderr_log_path_for,
    stdout_log_path_for,
)


def test_run_question_executes_gold_sql_and_scores_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    captured: dict[str, object] = {}
    run_id = "gold-sf_bq320-fixed"

    monkeypatch.setattr(gold_run, "_build_run_id", lambda question_id: run_id)

    def fake_execute_sql(
        sql: str,
        *,
        db: str,
        csv_path: Path,
        **kwargs: object,
    ) -> ExecutionResult:
        captured["sql"] = sql
        captured["db"] = db
        captured["csv_path"] = csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("unique_study_count\n1\n", encoding="utf-8")
        return ExecutionResult(
            ok=True,
            row_count=1,
            columns=["unique_study_count"],
            sample_rows=[{"unique_study_count": 1}],
            csv_path=str(csv_path),
        )

    def fake_runner(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["cwd"] = cwd
        assert cwd == eval_workspace_suite_dir_for(
            gold_run.ensure_run_paths(run_id, outputs_root=tmp_path),
            eval_id="default",
        )
        assert (cwd.parent / "spider2-snow.jsonl").exists()
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="{'sf_bq320': 1}\nFinal score: 1.0, Correct examples: 1, Total examples: 1\n",
            stderr="",
        )

    monkeypatch.setattr(gold_run, "execute_sql", fake_execute_sql)

    result = gold_run.run_question(
        "sf_bq320",
        outputs_root=tmp_path,
        evaluator_runner=fake_runner,
    )

    assert result.run_id == run_id
    assert result.question.instance_id == "sf_bq320"
    assert result.question.db == "IDC"
    assert result.score == 1
    assert result.run_paths.root == tmp_path / run_id
    assert result.run_paths.root.is_relative_to(tmp_path)
    assert "DICOM_PIVOT" in str(captured["sql"])
    assert captured["db"] == "IDC"
    assert captured["csv_path"] == tmp_path / run_id / "csv" / "sf_bq320.csv"
    assert "--mode" in captured["command"]
    assert "exec_result" in captured["command"]
    assert (
        (tmp_path / run_id / "sql" / "sf_bq320.sql")
        .read_text(encoding="utf-8")
        .startswith("SELECT")
    )
    assert csv_path_for(result.run_paths, instance_id="sf_bq320").exists()
    assert eval_input_csv_dir_for(result.run_paths, eval_id="default").is_relative_to(tmp_path)
    assert stdout_log_path_for(result.run_paths).exists()
    assert stderr_log_path_for(result.run_paths).read_text(encoding="utf-8") == ""
    assert run_log_path_for(result.run_paths).exists()
    assert (result.run_paths.manifest_path).exists()


def test_main_prints_exec_time_for_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    fake_result = SimpleNamespace(
        question=gold_run.Task(instance_id="sf_bq320", db="IDC", question="q"),
        execution=ExecutionResult(ok=True, row_count=1, columns=[], sample_rows=[]),
        score=1,
        run_id="gold-sf_bq320-260429.1200-abcdef12",
        run_paths=SimpleNamespace(root=Path("/tmp/gold-sf_bq320")),
    )

    monkeypatch.setattr(gold_run, "run_question", lambda instance_id: fake_result)

    assert gold_run.main(["sf_bq320"]) == 0
    output = capsys.readouterr().out
    assert "Exec time:" in output
    assert "Gold run passed." in output


def test_run_question_rejects_unknown_question():
    with pytest.raises(gold_run.GoldRunInputError, match="Unknown Spider2-snow question"):
        gold_run.run_question("missing-question")


def test_main_returns_distinct_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(
        gold_run,
        "run_question",
        lambda instance_id: (_ for _ in ()).throw(
            gold_run.GoldRunInputError("Unknown Spider2-snow question: missing")
        ),
    )

    assert gold_run.main(["missing"]) == 2
    assert "Unknown Spider2-snow question" in capsys.readouterr().err

    monkeypatch.setattr(
        gold_run,
        "run_question",
        lambda instance_id: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert gold_run.main(["sf_bq320"]) == 1
    assert "boom" in capsys.readouterr().err
