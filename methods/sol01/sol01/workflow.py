"""Shared workflow labels and lightweight trace models for the solver."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TASK_STATUS_SUCCESS = "success"
TASK_STATUS_FAILED = "failed"
TASK_STATUS_SKIPPED = "skipped"

RECOVERY_KIND_SQL = "sql"
RECOVERY_KIND_SCHEMA = "schema"
RECOVERY_KIND_SEMANTIC = "semantic"

RECOVERY_STAGE_SQL = "recovery_sql"
RECOVERY_STAGE_SCHEMA = "recovery_schema"
RECOVERY_STAGE_SEMANTIC = "recovery_semantic"

STOP_REASON_NO_ATTEMPTS = "no_attempts"
STOP_REASON_ATTEMPT_BUDGET_EXHAUSTED = "attempt_budget_exhausted"
STOP_REASON_SEMANTIC_REPAIR_BUDGET_EXHAUSTED = "semantic_repair_budget_exhausted"
STOP_REASON_SCHEMA_RECOVERY_COMPLETE = "schema_recovery_complete"
STOP_REASON_SQL_RECOVERY_COMPLETE = "sql_recovery_complete"
STOP_REASON_REVIEW_COMPLETE = "review_complete"
STOP_REASON_NO_RECOVERY_NEEDED = "no_recovery_needed"

REPAIR_SKIPPED_REASON_ATTEMPT_BUDGET = "attempt budget exhausted"
REPAIR_SKIPPED_REASON_SEMANTIC_BUDGET = "semantic repair budget exhausted"

RECOVERY_OUTCOME_RECOVERED = "recovered"
RECOVERY_OUTCOME_STILL_FAILED = "still_failed"
RECOVERY_OUTCOME_EXPANDED = "expanded"
RECOVERY_OUTCOME_EXPANDED_FAILED = "expanded_failed"
RECOVERY_OUTCOME_EXPANDED_NO_CANDIDATE = "expanded_no_candidate"
RECOVERY_OUTCOME_NO_NEW_TABLES = "no_new_tables"
RECOVERY_OUTCOME_DB_INDEX_UNAVAILABLE = "db_index_unavailable"

TaskStatus = Literal["success", "failed", "skipped"]
RecoveryKind = Literal["sql", "schema", "semantic"]
RecoveryOutcome = Literal[
    "recovered",
    "still_failed",
    "expanded",
    "expanded_failed",
    "expanded_no_candidate",
    "no_new_tables",
    "db_index_unavailable",
]
RecoveryStopReason = Literal[
    "no_attempts",
    "attempt_budget_exhausted",
    "semantic_repair_budget_exhausted",
    "schema_recovery_complete",
    "sql_recovery_complete",
    "review_complete",
    "no_recovery_needed",
    "expanded_no_candidate",
    "no_new_tables",
    "db_index_unavailable",
]
RepairSkippedReason = Literal["attempt budget exhausted", "semantic repair budget exhausted"]


def initial_attempt_stage(index: int) -> str:
    """Return the stable stage label for one initial candidate."""

    return f"initial_{index}"


def is_initial_attempt_stage(stage: str) -> bool:
    """Return True when a stage label belongs to the initial candidate batch."""

    return stage.startswith("initial_")


class CandidateReviewTrace(BaseModel):
    """Persisted candidate-review payload attached to a solver trace."""

    review_reason: str
    baseline_stage: str | None = None
    preferred_stage: str | None = None
    compared_stages: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    confidence: float
    issues: list[str] = Field(default_factory=list)
    should_repair: bool
    repair_focus: str | None = None


class RecoveryTrace(BaseModel):
    """Persisted recovery summary attached to a solver trace."""

    priority_order: list[str] = Field(default_factory=list)
    attempts_before_recovery: int
    attempts_after_recovery: int
    max_attempts: int
    semantic_repairs_allowed: int
    semantic_repairs_remaining: int
    actions: list[dict[str, object]] = Field(default_factory=list)
    stop_reason: RecoveryStopReason
