"""Single entry point for final candidate winner selection."""

from __future__ import annotations

import dataclasses

from sol01.candidates.scoring import best_attempt as _score_best
from sol01.models import AttemptRecord
from sol01.workflow import RECOVERY_STAGE_SCHEMA, RECOVERY_STAGE_SQL


@dataclasses.dataclass
class WinnerSelection:
    attempt: AttemptRecord
    index: int
    reason: str


def select_winner(attempts: list[AttemptRecord]) -> WinnerSelection | None:
    """Return the winning attempt using the v2 selection policy.

    Policy:
    - The highest-scoring attempt wins.
    - Execution success dominates via ±1000 bonus.
    """
    if not attempts:
        return None

    winner = _score_best(attempts)
    if winner is None:
        return None
    idx = next(i for i, a in enumerate(attempts) if a is winner)
    executables = sum(1 for a in attempts if a.execution_result.ok)
    if winner.execution_result.ok:
        reason = f"score: best executable (stage={winner.stage})"
    elif executables == 0:
        reason = f"score: no executable candidates (stage={winner.stage})"
    else:
        reason = f"score: best overall (stage={winner.stage})"
    return WinnerSelection(attempt=winner, index=idx, reason=reason)


def final_winner_reason(winner: AttemptRecord | None) -> str | None:
    """Derive a human-readable reason for why this attempt is the final winner."""
    if winner is None:
        return None
    stage = winner.stage
    if stage == RECOVERY_STAGE_SCHEMA:
        return "recovery: schema fix won"
    if stage == RECOVERY_STAGE_SQL:
        return "recovery: SQL fix won"
    return f"score: best executable (stage={stage})"
