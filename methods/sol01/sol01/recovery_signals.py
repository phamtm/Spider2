"""Signals that decide whether schema-aware recovery should run."""

from __future__ import annotations

from sol01.infra.policy import DEFAULT_RECOVERY_SIGNAL_POLICY
from sol01.models import AttemptRecord

_MISSING_TABLE_ERROR_CODES = ("000904", "002003")
_MISSING_TABLE_ERROR_PHRASES = ("does not exist", "object does not exist", "table not found")


def schema_expansion_trigger(attempt: AttemptRecord) -> str | None:
    """Return why schema expansion is warranted from validation or execution feedback."""

    for error in attempt.validation.errors:
        error_lower = error.lower()
        if "unknown table referenced" in error_lower:
            return f"validation_unknown_table: {error}"
        if "unknown column" in error_lower:
            return f"validation_unknown_column: {error}"

    for warning in attempt.validation.warnings:
        if "no selected table has it" in warning.lower():
            return f"validation_missing_column: {warning}"

    execution_signal = _execution_schema_signal(attempt.execution_result.error)
    if execution_signal is not None:
        return execution_signal

    return None


def _execution_schema_signal(error: str | None) -> str | None:
    """Return a schema-recovery trigger from one execution error string."""

    if not error:
        return None

    error_lower = error.lower()
    if "invalid identifier" in error_lower:
        return f"execution_unknown_column: {_preview_error(error)}"
    if _looks_like_missing_table_error(error, error_lower):
        return f"execution_unknown_table: {_preview_error(error)}"
    return None


def _looks_like_missing_table_error(error: str, error_lower: str) -> bool:
    """Return True when an execution error points to a missing table/object."""

    return any(code in error for code in _MISSING_TABLE_ERROR_CODES) or any(
        phrase in error_lower for phrase in _MISSING_TABLE_ERROR_PHRASES
    )


def _preview_error(error: str) -> str:
    """Return a stable preview for trace payloads."""

    return error[: DEFAULT_RECOVERY_SIGNAL_POLICY.execution_error_preview_chars]
