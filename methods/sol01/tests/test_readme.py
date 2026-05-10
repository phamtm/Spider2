"""Tests for the operator-facing README."""

from __future__ import annotations

from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_documents_persisted_eval_runs():
    text = README.read_text(encoding="utf-8")

    assert "uv run sol01 run --instance-id sf_bq320" in text
    assert "uv run sol01 run sf035 sf_bq135 sf_bq084" in text
    assert "uv run sol01 run --db E_COMMERCE --question-contains revenue" in text
    assert "just run" in text
    assert "just run-selected sf035 sf_bq135" in text
    assert "just gold sf_bq320" in text
    assert "run_mode" not in text
    assert "task selectors are ORed" not in text
    assert "repeated `tag:` filters are ANDed" not in text
    assert "`all` must stand alone" not in text
    assert "Category shortcuts are also available:" not in text
    assert "eval/runs/default/" in text
    assert "eval/runs/<filtered-tag>/" in text
    assert "command.json" in text
    assert "input_csv.csv" in text
    assert "workspace/temp/" in text
    assert "workspace/spider2-snow/evaluation_suite/log.txt" in text
    assert "just smoke" not in text
