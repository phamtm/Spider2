"""Evidence-based candidate scoring: named checks drive eligibility and ordering."""

from __future__ import annotations

from sol01.models import (
    AggregateGrainReport,
    AttemptRecord,
    CandidateEvidence,
    ExecutionResult,
    FilterGroundingReport,
    OutputShapeReport,
    ValidationReport,
)

_EXECUTABLE_SCORE = 1000.0
_ISSUE_PENALTY = 10.0
_CONFIDENCE_FACTOR = 0.01


def build_candidate_evidence(
    *,
    execution: ExecutionResult,
    validation: ValidationReport,
    shape_report: OutputShapeReport | None = None,
    filter_grounding_report: FilterGroundingReport | None = None,
    aggregate_grain: AggregateGrainReport | None = None,
) -> CandidateEvidence:
    """Collect deterministic check results into named evidence."""
    issues: list[str] = []

    if not validation.ok:
        for err in validation.errors:
            issues.append(f"validation_error: {err[:80]}")
    for warn in validation.warnings:
        issues.append(f"validation_warning: {warn[:80]}")

    if execution.ok and shape_report is not None:
        for col in shape_report.missing_columns:
            issues.append(f"shape_missing_column: {col}")
        for violation in shape_report.violations:
            issues.append(f"shape_violation: {violation[:80]}")

    if execution.ok and filter_grounding_report is not None:
        if filter_grounding_report.zero_like_result and not filter_grounding_report.value_rewrites:
            issues.append("filter_zero_like_ungrounded")

    if execution.ok and aggregate_grain is not None:
        clear_mismatch = (
            aggregate_grain.inferred_grain == "distinct_entity_count"
            and not aggregate_grain.uses_distinct
        ) or (aggregate_grain.inferred_grain == "row_count" and aggregate_grain.uses_distinct)
        if clear_mismatch:
            issues.append("aggregate_grain_mismatch")

    return CandidateEvidence(
        executable=execution.ok,
        validation_ok=validation.ok,
        validation_errors=list(validation.errors),
        validation_warnings=list(validation.warnings),
        row_count=execution.row_count,
        column_count=len(execution.columns),
        issues=issues,
    )


def score_from_evidence(evidence: CandidateEvidence, *, candidate_confidence: float = 0.0) -> float:
    """Derive a numeric score for fallback ordering when no model preference exists."""
    base = _EXECUTABLE_SCORE if evidence.executable else -_EXECUTABLE_SCORE
    return base - _ISSUE_PENALTY * len(evidence.issues) + candidate_confidence * _CONFIDENCE_FACTOR


def best_attempt(attempts: list[AttemptRecord]) -> AttemptRecord | None:
    """Return the highest-scoring attempt."""
    if not attempts:
        return None
    return max(attempts, key=lambda a: a.score)
