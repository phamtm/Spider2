"""Tests for the sol01 justfile recipes."""

from __future__ import annotations

import subprocess
from pathlib import Path

JUSTFILE = Path(__file__).resolve().parents[1] / "justfile"


def test_just_run_selected_dispatches_selectors_to_solver_cli():
    completed = subprocess.run(
        ["just", "--justfile", str(JUSTFILE), "--dry-run", "run-selected", "sf035", "sf_bq135"],
        cwd=JUSTFILE.parent,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    output = completed.stdout + completed.stderr
    assert "uv run sol01 run sf035 sf_bq135" in output
