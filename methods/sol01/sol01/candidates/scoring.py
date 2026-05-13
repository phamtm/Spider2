"""Score SQL candidates using execution and verification evidence."""

from __future__ import annotations

from numbers import Number
from typing import Any

from sol01.models import (
    AggregateGrainReport,
    AttemptRecord,
    ExecutionResult,
    FilterGroundingReport,
    Intent,
    OutputShapeReport,
    SQLCandidate,
    ValidationReport,
)

# -- Execution status weights --
EXECUTION_SUCCESS_BONUS = 1000.0
EXECUTION_FAILURE_PENALTY = -1000.0

# -- Validation weights --
VALIDATION_CLEAN_BONUS = 120.0
VALIDATION_FAILURE_PENALTY = -180.0
VALIDATION_ERROR_PENALTY = -15.0
VALIDATION_WARNING_PENALTY = -5.0

# -- Output shape weights --
SHAPE_COLUMN_MATCH_BONUS = 45.0
SHAPE_COLUMN_NEAR_MATCH_BONUS = 10.0
SHAPE_COLUMN_MISMATCH_PENALTY = -25.0
SHAPE_SCALAR_EXACT_BONUS = 35.0
SHAPE_SCALAR_PARTIAL_BONUS = 12.0
SHAPE_SCALAR_MISS_PENALTY = -18.0
SHAPE_TABULAR_FULL_BONUS = 18.0
SHAPE_TABULAR_PARTIAL_BONUS = 6.0
SHAPE_TABULAR_EMPTY_PENALTY = -12.0
SHAPE_NONEMPTY_BONUS = 4.0
SHAPE_EMPTY_PENALTY = -4.0

# -- Shape report weights --
REPORT_COLUMN_EXACT_BONUS = 30.0
REPORT_COLUMN_NEAR_BONUS = 12.0
REPORT_MISSING_COLUMN_PENALTY = -28.0
REPORT_VIOLATION_PENALTY = -20.0
REPORT_GROUPED_KEY_PENALTY = -15.0

# -- Verification penalty weights --
PENALTY_MISSING_GROUPED_KEY = -8.0
PENALTY_MISSING_GROUPED_KEY_CAP = -16.0
PENALTY_ZERO_LIKE_UNGROUNDED = -10.0
PENALTY_AGGREGATE_GRAIN_MISMATCH = -5.0

# -- Filter grounding weights --
FILTER_UNEXECUTED_PENALTY = -30.0
FILTER_GROUNDED_REWRITE_BONUS = 16.0
FILTER_UNGROUNDED_ZERO_PENALTY = -22.0
FILTER_BASE_SCORE = 14.0
FILTER_VALIDATION_ERROR_PENALTY = -8.0
FILTER_VALIDATION_WARNING_PENALTY = -3.0
FILTER_VALUE_REWRITE_BONUS = 6.0

# -- Cardinality plausibility weights --
CARDINALITY_EMPTY_PENALTY = -16.0
CARDINALITY_SINGLE_SMALL_PENALTY = -14.0
CARDINALITY_FEW_SMALL_PENALTY = -8.0
CARDINALITY_NONEMPTY_BONUS = 3.0

# -- Aggregate grain weights --
AGGREGATE_ROW_COUNT_DISTINCT_PENALTY = -6.0
AGGREGATE_ROW_COUNT_BONUS = 2.0
AGGREGATE_DISTINCT_ENTITY_BONUS = 2.0
AGGREGATE_DISTINCT_ENTITY_MISS_PENALTY = -4.0
AGGREGATE_VALUE_COUNT_BONUS = 0.5

# -- Confidence tiebreaker --
CONFIDENCE_TIEBREAKER_FACTOR = 0.01


def attempt_score_breakdown(
    *,
    intent: Intent | None,
    candidate: SQLCandidate,
    validation: ValidationReport,
    execution: ExecutionResult,
    aggregate_grain: AggregateGrainReport | None = None,
    result_profile: dict[str, Any] | None = None,
    shape_report: OutputShapeReport | None = None,
    filter_grounding_report: FilterGroundingReport | None = None,
) -> dict[str, float]:
    """Return the score contribution for each verification check."""

    return {
        "execution_status": _execution_status_adjustment(execution),
        "validation": _validation_adjustment(validation),
        "shape": _output_shape_adjustment(intent, execution, result_profile, shape_report),
        "filter_grounding": _filter_grounding_adjustment(
            intent,
            validation,
            execution,
            filter_grounding_report,
        ),
        "aggregate_grain": _aggregate_grain_adjustment(aggregate_grain),
        "cardinality": _cardinality_plausibility_adjustment(execution, result_profile),
        "verification_penalty": sum(
            verification_penalty_reasons(
                execution=execution,
                shape_report=shape_report,
                filter_grounding_report=filter_grounding_report,
                aggregate_grain=aggregate_grain,
            ).values()
        ),
        "confidence_tiebreaker": candidate.confidence * CONFIDENCE_TIEBREAKER_FACTOR,
    }


def verification_penalty_reasons(
    *,
    execution: ExecutionResult,
    shape_report: OutputShapeReport | None = None,
    filter_grounding_report: FilterGroundingReport | None = None,
    aggregate_grain: AggregateGrainReport | None = None,
) -> dict[str, float]:
    """Named penalty contributions for known-bad verification patterns.

    Only fires on executable candidates. Penalties are bounded so that
    execution and validation still dominate non-executable SQL.
    """
    if not execution.ok:
        return {}

    reasons: dict[str, float] = {}

    if shape_report is not None:
        grouped_key_count = sum(1 for v in shape_report.violations if "grouped key" in v)
        if grouped_key_count:
            reasons["missing_grouped_key"] = max(
                PENALTY_MISSING_GROUPED_KEY * grouped_key_count,
                PENALTY_MISSING_GROUPED_KEY_CAP,
            )

    if filter_grounding_report is not None:
        if filter_grounding_report.zero_like_result and not filter_grounding_report.value_rewrites:
            reasons["zero_like_ungrounded_filter"] = PENALTY_ZERO_LIKE_UNGROUNDED

    if aggregate_grain is not None:
        clear_mismatch = (
            aggregate_grain.inferred_grain == "distinct_entity_count"
            and not aggregate_grain.uses_distinct
        ) or (aggregate_grain.inferred_grain == "row_count" and aggregate_grain.uses_distinct)
        if clear_mismatch:
            reasons["aggregate_grain_mismatch"] = PENALTY_AGGREGATE_GRAIN_MISMATCH

    return reasons


def best_attempt(attempts: list[AttemptRecord]) -> AttemptRecord | None:
    """Return the highest-scoring attempt so far."""

    if not attempts:
        return None
    return max(attempts, key=lambda attempt: attempt.score)


def _aggregate_grain_adjustment(report: AggregateGrainReport | None) -> float:
    """Score candidate aggregates according to the inferred grain."""

    if report is None:
        return 0.0
    if report.inferred_grain == "row_count":
        if report.uses_distinct:
            return AGGREGATE_ROW_COUNT_DISTINCT_PENALTY
        return AGGREGATE_ROW_COUNT_BONUS
    if report.inferred_grain == "distinct_entity_count":
        if report.uses_distinct:
            return AGGREGATE_DISTINCT_ENTITY_BONUS
        return AGGREGATE_DISTINCT_ENTITY_MISS_PENALTY
    if report.inferred_grain == "value_count":
        return AGGREGATE_VALUE_COUNT_BONUS
    return 0.0


def _execution_status_adjustment(execution: ExecutionResult) -> float:
    """Make execution success dominate every other signal."""

    return EXECUTION_SUCCESS_BONUS if execution.ok else EXECUTION_FAILURE_PENALTY


def _validation_adjustment(validation: ValidationReport) -> float:
    """Reward SQL that validates cleanly and penalize noisy validation."""

    score = VALIDATION_CLEAN_BONUS if validation.ok else VALIDATION_FAILURE_PENALTY
    score += VALIDATION_ERROR_PENALTY * len(validation.errors)
    score += VALIDATION_WARNING_PENALTY * len(validation.warnings)
    return score


def _output_shape_adjustment(
    intent: Intent | None,
    execution: ExecutionResult,
    result_profile: dict[str, Any] | None,
    shape_report: OutputShapeReport | None = None,
) -> float:
    """Prefer result shapes that match the task's stated output contract."""

    if not execution.ok:
        return 0.0

    score = 0.0
    if shape_report is not None:
        score += _shape_report_adjustment(shape_report)

    profile = result_profile or {
        "row_count": execution.row_count,
        "columns": execution.columns,
    }
    row_count = int(profile.get("row_count") or 0)
    column_count = len(profile.get("columns") or [])
    expectation = _normalized_output_expectation(intent)

    expected_columns = _expected_output_columns(expectation)
    if expected_columns is not None and shape_report is None:
        if column_count == expected_columns:
            score += SHAPE_COLUMN_MATCH_BONUS
        elif abs(column_count - expected_columns) == 1:
            score += SHAPE_COLUMN_NEAR_MATCH_BONUS
        else:
            score += SHAPE_COLUMN_MISMATCH_PENALTY

    if shape_report is None:
        if _expects_scalar_output(expectation):
            if row_count == 1 and column_count == 1:
                score += SHAPE_SCALAR_EXACT_BONUS
            elif row_count == 1 or column_count == 1:
                score += SHAPE_SCALAR_PARTIAL_BONUS
            else:
                score += SHAPE_SCALAR_MISS_PENALTY

        if _expects_tabular_output(expectation):
            if row_count > 0 and column_count >= 2:
                score += SHAPE_TABULAR_FULL_BONUS
            elif row_count > 0:
                score += SHAPE_TABULAR_PARTIAL_BONUS
            else:
                score += SHAPE_TABULAR_EMPTY_PENALTY

        if row_count > 0 and column_count > 0:
            score += SHAPE_NONEMPTY_BONUS
        else:
            score += SHAPE_EMPTY_PENALTY

    return score


def _shape_report_adjustment(report: OutputShapeReport | None) -> float:
    """Score a candidate by how well its observed columns match the inferred contract."""

    if report is None:
        return 0.0

    score = 0.0
    if report.expected_columns:
        if report.observed_columns == report.expected_columns:
            score += REPORT_COLUMN_EXACT_BONUS
        elif not report.missing_columns:
            score += REPORT_COLUMN_NEAR_BONUS
        else:
            score += REPORT_MISSING_COLUMN_PENALTY * len(report.missing_columns)

    if report.violations:
        score += REPORT_VIOLATION_PENALTY * len(report.violations)
        if any("grouped key" in violation for violation in report.violations):
            score += REPORT_GROUPED_KEY_PENALTY
    return score


def _filter_grounding_adjustment(
    intent: Intent | None,
    validation: ValidationReport,
    execution: ExecutionResult,
    report: FilterGroundingReport | None,
) -> float:
    """Prefer candidates whose filters appear grounded in observed values."""

    if intent is None or not intent.filters:
        return 0.0
    if not execution.ok:
        return FILTER_UNEXECUTED_PENALTY
    if report is not None and report.zero_like_result:
        if report.value_rewrites:
            return FILTER_GROUNDED_REWRITE_BONUS
        return FILTER_UNGROUNDED_ZERO_PENALTY
    score = FILTER_BASE_SCORE
    if validation.errors:
        score += FILTER_VALIDATION_ERROR_PENALTY
    if validation.warnings:
        score += FILTER_VALIDATION_WARNING_PENALTY * len(validation.warnings)
    if report is not None and report.value_rewrites:
        score += FILTER_VALUE_REWRITE_BONUS
    return score


def _cardinality_plausibility_adjustment(
    execution: ExecutionResult,
    result_profile: dict[str, Any] | None,
) -> float:
    """Reward results whose size looks plausible for the query type."""

    if not execution.ok:
        return 0.0

    profile = result_profile or {
        "row_count": execution.row_count,
        "sample_rows": execution.sample_rows,
    }
    row_count = int(profile.get("row_count") or 0)
    if row_count == 0:
        return CARDINALITY_EMPTY_PENALTY

    sample_rows = profile.get("sample_rows") or []
    numeric_values = [
        value
        for value in (_coerce_number(item) for row in sample_rows for item in row.values())
        if value is not None
    ]
    if not numeric_values:
        return 0.0

    max_value = max(numeric_values)
    if row_count == 1 and max_value <= 1:
        return CARDINALITY_SINGLE_SMALL_PENALTY
    if row_count <= 2 and max_value <= 2:
        return CARDINALITY_FEW_SMALL_PENALTY
    if row_count > 0:
        return CARDINALITY_NONEMPTY_BONUS
    return 0.0


def _normalized_output_expectation(intent: Intent | None) -> str:
    """Normalize the intent's output expectation for shape heuristics."""

    if intent is None:
        return ""
    return " ".join(intent.output_expectation.lower().split())


def _expected_output_columns(expectation: str) -> int | None:
    """Infer how many columns the answer should expose from the intent text."""

    if not expectation:
        return None
    if any(token in expectation for token in (" and ", " columns", " per ", " by ")):
        return 2
    if any(token in expectation for token in ("count", "how many", "one count", "single count")):
        return 1
    if any(token in expectation for token in ("one column", "single column")):
        return 1
    return None


def _expects_scalar_output(expectation: str) -> bool:
    """Return True when the answer is likely a single value."""

    return any(token in expectation for token in ("count", "how many", "one count", "single count"))


def _expects_tabular_output(expectation: str) -> bool:
    """Return True when the answer is likely a multi-column result set."""

    return any(
        token in expectation for token in (" and ", " columns", " per ", " by ", " rows", " list ")
    )


def _coerce_number(value: Any) -> float | None:
    """Convert one value to a number when it looks numeric."""

    if isinstance(value, bool):
        return None
    if isinstance(value, Number):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
