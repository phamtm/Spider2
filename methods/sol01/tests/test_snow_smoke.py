"""Tests for the single-question Snow smoke runner."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sol01 import snow_smoke
from sol01.models import ExecutionResult


def test_run_question_executes_gold_sql_and_scores_csv(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_execute_sql(sql: str, *, db: str, csv_path: Path, **kwargs: object) -> ExecutionResult:
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
        assert (cwd.parent / "spider2-snow.jsonl").exists()
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="{'sf_bq320': 1}\nFinal score: 1.0, Correct examples: 1, Total examples: 1\n",
            stderr="",
        )

    monkeypatch.setattr(snow_smoke, "execute_sql", fake_execute_sql)

    result = snow_smoke.run_question("sf_bq320", evaluator_runner=fake_runner)

    assert result.question.instance_id == "sf_bq320"
    assert result.question.db == "IDC"
    assert result.score == 1
    assert "DICOM_PIVOT" in str(captured["sql"])
    assert captured["db"] == "IDC"
    assert "--mode" in captured["command"]
    assert "exec_result" in captured["command"]


def test_run_question_rejects_unknown_question():
    with pytest.raises(ValueError, match="Unknown Spider2-snow question"):
        snow_smoke.run_question("missing-question")
