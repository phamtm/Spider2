"""Unit tests for the v2 candidate winner selector."""

from __future__ import annotations

from sol01.candidates.selection import WinnerSelection, select_winner
from sol01.models import AttemptRecord, ExecutionResult, ValidationReport


def _attempt(
    stage: str,
    *,
    ok: bool = True,
    score: float = 100.0,
) -> AttemptRecord:
    return AttemptRecord(
        stage=stage,
        sql=f"SELECT 1 -- {stage}",
        explanation=stage,
        candidate_confidence=0.8,
        validation=ValidationReport(ok=True),
        execution_result=ExecutionResult(ok=ok, row_count=1 if ok else 0),
        score=score,
    )


def test_select_winner_returns_none_for_empty():
    assert select_winner([]) is None


def test_select_winner_score_based_picks_highest():
    a1 = _attempt("initial_1", score=500.0)
    a2 = _attempt("initial_2", score=900.0)
    result = select_winner([a1, a2])
    assert isinstance(result, WinnerSelection)
    assert result.attempt is a2
    assert result.index == 1
    assert "score" in result.reason
    assert "initial_2" in result.reason


def test_select_winner_executable_beats_non_executable():
    non_exec = _attempt("initial_1", ok=False, score=-800.0)
    exec_ok = _attempt("initial_2", ok=True, score=1000.0)
    result = select_winner([non_exec, exec_ok])
    assert result is not None
    assert result.attempt is exec_ok


def test_select_winner_schema_recovery_wins_when_higher_score():
    initial = _attempt("initial_1", ok=False, score=-800.0)
    expansion = _attempt("recovery_schema", ok=True, score=1100.0)
    result = select_winner([initial, expansion])
    assert result is not None
    assert result.attempt is expansion
    assert result.index == 1
    assert "recovery_schema" in result.reason


def test_select_winner_no_executable_returns_best_score():
    a1 = _attempt("initial_1", ok=False, score=-500.0)
    a2 = _attempt("initial_2", ok=False, score=-200.0)
    result = select_winner([a1, a2])
    assert result is not None
    assert result.attempt is a2
    assert "no executable" in result.reason
