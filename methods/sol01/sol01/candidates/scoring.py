"""Score SQL candidates using execution and verification evidence."""

from __future__ import annotations

from numbers import Number
from typing import Any

from sol01.models import (
    AggregateGrainReport,
    ExecutionResult,
    FilterGroundingReport,
    Intent,
    OutputShapeReport,
    SQLCandidate,
    ValidationReport,
)


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
        "confidence_tiebreaker": candidate.confidence * 0.01,
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
            reasons["missing_grouped_key"] = max(-8.0 * grouped_key_count, -16.0)

    if filter_grounding_report is not None:
        if filter_grounding_report.zero_like_result and not filter_grounding_report.value_rewrites:
            reasons["zero_like_ungrounded_filter"] = -10.0

    if aggregate_grain is not None:
        clear_mismatch = (
            aggregate_grain.inferred_grain == "distinct_entity_count"
            and not aggregate_grain.uses_distinct
        ) or (aggregate_grain.inferred_grain == "row_count" and aggregate_grain.uses_distinct)
        if clear_mismatch:
            reasons["aggregate_grain_mismatch"] = -5.0

    return reasons


def best_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the highest-scoring attempt so far."""

    if not attempts:
        return None
    return max(attempts, key=lambda attempt: float(attempt["score"]))


_attempt_score_breakdown = attempt_score_breakdown
_verification_penalty_reasons = verification_penalty_reasons
_best_attempt = best_attempt


def _aggregate_grain_adjustment(report: AggregateGrainReport | None) -> float:
    """Score candidate aggregates according to the inferred grain."""

    if report is None:
        return 0.0
    if report.inferred_grain == "row_count":
        return -6.0 if report.uses_distinct else 2.0
    if report.inferred_grain == "distinct_entity_count":
        return 2.0 if report.uses_distinct else -4.0
    if report.inferred_grain == "value_count":
        return 0.5
    return 0.0


def _execution_status_adjustment(execution: ExecutionResult) -> float:
    """Make execution success dominate every other signal."""

    return 1000.0 if execution.ok else -1000.0


def _validation_adjustment(validation: ValidationReport) -> float:
    """Reward SQL that validates cleanly and penalize noisy validation."""

    score = 120.0 if validation.ok else -180.0
    score -= 15.0 * len(validation.errors)
    score -= 5.0 * len(validation.warnings)
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
            score += 45.0
        elif abs(column_count - expected_columns) == 1:
            score += 10.0
        else:
            score -= 25.0

    if shape_report is None:
        if _expects_scalar_output(expectation):
            if row_count == 1 and column_count == 1:
                score += 35.0
            elif row_count == 1 or column_count == 1:
                score += 12.0
            else:
                score -= 18.0

        if _expects_tabular_output(expectation):
            if row_count > 0 and column_count >= 2:
                score += 18.0
            elif row_count > 0:
                score += 6.0
            else:
                score -= 12.0

        if row_count > 0 and column_count > 0:
            score += 4.0
        else:
            score -= 4.0

    return score


def _shape_report_adjustment(report: OutputShapeReport | None) -> float:
    """Score a candidate by how well its observed columns match the inferred contract."""

    if report is None:
        return 0.0

    score = 0.0
    if report.expected_columns:
        if report.observed_columns == report.expected_columns:
            score += 30.0
        elif not report.missing_columns:
            score += 12.0
        else:
            score -= 28.0 * len(report.missing_columns)

    if report.violations:
        score -= 20.0 * len(report.violations)
        if any("grouped key" in violation for violation in report.violations):
            score -= 15.0
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
        return -30.0
    if report is not None and report.zero_like_result:
        if report.value_rewrites:
            return 16.0
        return -22.0
    score = 14.0
    if validation.errors:
        score -= 8.0
    if validation.warnings:
        score -= 3.0 * len(validation.warnings)
    if report is not None and report.value_rewrites:
        score += 6.0
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
        return -16.0

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
        return -14.0
    if row_count <= 2 and max_value <= 2:
        return -8.0
    if row_count > 0:
        return 3.0
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
