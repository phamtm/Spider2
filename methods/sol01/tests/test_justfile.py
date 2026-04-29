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


def test_just_run_dispatches_to_persisted_run_mode():
    completed = _run_just("run", "sf_bq320")

    assert completed.returncode == 0
    assert "python -m sol01.run_mode" in _combined_output(completed)
    assert "sf_bq320" in _combined_output(completed)


def test_just_run_accepts_multiple_patterns_and_all_mode():
    patterns = _run_just("run", "sf_bq3*", "sf_bq4*")
    all_mode = _run_just("all")

    assert patterns.returncode == 0
    assert "python -m sol01.run_mode" in _combined_output(patterns)
    assert "sf_bq3*" in _combined_output(patterns)
    assert "sf_bq4*" in _combined_output(patterns)

    assert all_mode.returncode == 0
    assert "python -m sol01.run_mode --all" in _combined_output(all_mode)


def test_just_smoke_remains_separate_from_persisted_run_mode():
    completed = _run_just("smoke", "sf_bq320")

    assert completed.returncode == 0
    output = _combined_output(completed)
    assert "python -m sol01.snow_smoke" in output
    assert "python -m sol01.run_mode" not in output
