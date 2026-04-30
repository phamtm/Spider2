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


def test_category_shortcuts_dispatch_to_persisted_run_mode():
    tiers = _run_just("tiers", "tier:1", "tier:2")
    tags = _run_just("tags", "tag:spatial", "tag:spatial_join")
    easy = _run_just("easy")
    hard = _run_just("hard")
    spatial = _run_just("spatial")
    nested_events = _run_just("nested_events")
    anti_join = _run_just("anti_join")
    external_formula = _run_just("external_formula")

    assert tiers.returncode == 0
    assert "python -m sol01.run_mode" in _combined_output(tiers)
    assert "tier:1" in _combined_output(tiers)
    assert "tier:2" in _combined_output(tiers)

    assert tags.returncode == 0
    assert "tag:spatial" in _combined_output(tags)
    assert "tag:spatial_join" in _combined_output(tags)

    assert easy.returncode == 0
    assert "tier:1" in _combined_output(easy)
    assert "tier:2" in _combined_output(easy)

    assert hard.returncode == 0
    assert "tier:6-12" in _combined_output(hard)

    assert spatial.returncode == 0
    assert "tag:spatial" in _combined_output(spatial)
    assert "tag:spatial_join" in _combined_output(spatial)
    assert "tag:spatial_adjustment" in _combined_output(spatial)

    assert nested_events.returncode == 0
    assert "tag:event_sequence" in _combined_output(nested_events)
    assert "tag:event_parsing" in _combined_output(nested_events)
    assert "tag:event_classification" in _combined_output(nested_events)

    assert anti_join.returncode == 0
    assert "tag:anti_join" in _combined_output(anti_join)

    assert external_formula.returncode == 0
    assert "tag:formula" in _combined_output(external_formula)
    assert "tag:external_knowledge" in _combined_output(external_formula)


def test_just_run_requires_at_least_one_pattern():
    completed = _run_just("run")

    assert completed.returncode != 0
    assert "takes at least 1" in _combined_output(completed)


def test_just_gold_remains_separate_from_persisted_run_mode():
    completed = _run_just("gold", "sf_bq320")

    assert completed.returncode == 0
    output = _combined_output(completed)
    assert "python -m sol01.gold_run" in output
    assert "python -m sol01.run_mode" not in output


def test_justfile_no_longer_exposes_smoke_recipe():
    assert "smoke instance_id:" not in JUSTFILE.read_text(encoding="utf-8")
