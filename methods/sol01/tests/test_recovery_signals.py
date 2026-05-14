"""Focused tests for schema-recovery trigger detection."""

from __future__ import annotations

from sol01.models import AttemptRecord, ExecutionResult, ValidationReport
from sol01.recovery_signals import schema_expansion_trigger


def _attempt(
    *,
    validation_errors: list[str] | None = None,
    validation_warnings: list[str] | None = None,
    execution_error: str | None = None,
    critic: dict[str, object] | None = None,
) -> AttemptRecord:
    return AttemptRecord(
        stage="initial_1",
        sql="SELECT 1",
        explanation="test",
        candidate_confidence=0.8,
        validation=ValidationReport(
            ok=not validation_errors,
            errors=validation_errors or [],
            warnings=validation_warnings or [],
        ),
        execution_result=ExecutionResult(
            ok=execution_error is None,
            row_count=0,
            error=execution_error,
        ),
        score=0.0,
        critic=critic,
    )


def test_schema_expansion_trigger_prefers_validation_unknown_column() -> None:
    attempt = _attempt(validation_errors=["Unknown column CUSTOMER_ID"])

    assert (
        schema_expansion_trigger(attempt) == "validation_unknown_column: Unknown column CUSTOMER_ID"
    )


def test_schema_expansion_trigger_detects_missing_table_execution_errors() -> None:
    attempt = _attempt(
        execution_error="000904 (42000): SQL compilation error: Object 'ORDERS' does not exist"
    )

    assert schema_expansion_trigger(attempt).startswith("execution_unknown_table:")


def test_schema_expansion_trigger_detects_invalid_identifier_execution_errors() -> None:
    attempt = _attempt(execution_error="SQL compilation error: invalid identifier 'MISSING_COLUMN'")

    assert schema_expansion_trigger(attempt).startswith("execution_unknown_column:")


def test_schema_expansion_trigger_detects_critic_schema_issue() -> None:
    attempt = _attempt(
        critic={"issues": ["The query is missing a join between orders and customers."]}
    )

    assert schema_expansion_trigger(attempt) == (
        "critic_issue: The query is missing a join between orders and customers."
    )
