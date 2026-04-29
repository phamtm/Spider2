"""Tests for the operator-facing README."""

from __future__ import annotations

from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_documents_persisted_eval_runs():
    text = README.read_text(encoding="utf-8")

    assert "just run <selector>" in text
    assert "eval/runs/default/" in text
    assert "eval/runs/<filtered-tag>/" in text
    assert "command.json" in text
    assert "input_csv.csv" in text
    assert "workspace/temp/" in text
    assert "workspace/spider2-snow/evaluation_suite/log.txt" in text
    assert "just smoke" not in text
