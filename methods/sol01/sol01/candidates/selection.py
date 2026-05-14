"""Single entry point for final candidate winner selection."""

from __future__ import annotations

import dataclasses
from typing import Any

from sol01.candidates.scoring import best_attempt as _score_best
from sol01.models import AttemptRecord


@dataclasses.dataclass
class WinnerSelection:
    attempt: AttemptRecord
    index: int
    reason: str


def select_winner(
    attempts: list[AttemptRecord],
    *,
    preferred_stage: str | None = None,
) -> WinnerSelection | None:
    """Return the winning attempt using the v2 selection policy.

    Policy:
    - If preferred_stage is given and names an executable attempt, that attempt wins.
    - Otherwise the highest-scoring attempt wins; execution success dominates via ±1000 bonus.
    """
    if not attempts:
        return None

    if preferred_stage is not None:
        executables = [a for a in attempts if a.execution_result.ok]
        preferred = next((a for a in executables if a.stage == preferred_stage), None)
        if preferred is not None:
            idx = next(i for i, a in enumerate(attempts) if a is preferred)
            return WinnerSelection(
                attempt=preferred,
                index=idx,
                reason=f"model preferred: {preferred_stage}",
            )

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


def final_winner_reason(
    winner: AttemptRecord | None,
    *,
    candidate_review_payload: dict[str, Any] | None,
) -> str | None:
    """Derive a human-readable reason for why this attempt is the final winner."""
    if winner is None:
        return None
    stage = winner.stage
    if (
        candidate_review_payload is not None
        and candidate_review_payload.get("preferred_stage") == stage
    ):
        return f"model preferred: {stage}"
    if stage == "critic_repair":
        return "critic repair: highest scoring"
    if stage == "schema_expansion":
        return "schema expansion: highest scoring"
    return f"score: best executable (stage={stage})"
