"""Tests for the sol01 justfile recipes."""

from __future__ import annotations

import subprocess
from pathlib import Path

JUSTFILE = Path(__file__).resolve().parents[1] / "justfile"


def _run_just(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["just", "--justfile", str(JUSTFILE), "--dry-run", *args],
        cwd=JUSTFILE.parent,
        check=False,
        capture_output=True,
        text=True,
    )


def _combined_output(completed: subprocess.CompletedProcess[str]) -> str:
    return completed.stdout + completed.stderr


def test_just_run_dispatches_to_solver_cli():
    completed = _run_just("run")

    assert completed.returncode == 0
    output = _combined_output(completed)
    assert "uv run sol01 run" in output
    assert "run_mode" not in output


def test_just_run_selected_dispatches_selectors_to_solver_cli():
    completed = _run_just("run-selected", "sf035", "sf_bq135")

    assert completed.returncode == 0
    output = _combined_output(completed)
    assert "uv run sol01 run sf035 sf_bq135" in output


def test_just_gold_remains_separate_from_solver_cli():
    completed = _run_just("gold", "sf_bq320")

    assert completed.returncode == 0
    output = _combined_output(completed)
    assert "python -m sol01.analysis.gold_run" in output
    assert "uv run sol01 run" not in output


def test_justfile_no_longer_exposes_legacy_selector_shortcuts():
    text = JUSTFILE.read_text(encoding="utf-8")

    assert "run_mode" not in text
    assert "tiers +" not in text
    assert "tags +" not in text
    assert "easy:" not in text
    assert "hard:" not in text
    assert "spatial:" not in text
    assert "nested_events:" not in text
    assert "anti_join:" not in text
    assert "external_formula:" not in text
    assert "all:" not in text
