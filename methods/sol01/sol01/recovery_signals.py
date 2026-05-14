"""Signals that decide whether schema-aware recovery should run."""

from __future__ import annotations

from sol01.infra.policy import DEFAULT_RECOVERY_SIGNAL_POLICY
from sol01.models import AttemptRecord


def schema_expansion_trigger(attempt: AttemptRecord) -> str | None:
    """Return why schema expansion is warranted for this attempt, or None."""

    for error in attempt.validation.errors:
        error_lower = error.lower()
        if "unknown table referenced" in error_lower:
            return f"validation_unknown_table: {error}"
        if "unknown column" in error_lower:
            return f"validation_unknown_column: {error}"

    for warning in attempt.validation.warnings:
        if "no selected table has it" in warning.lower():
            return f"validation_missing_column: {warning}"

    exec_error = (attempt.execution_result.error or "").lower()
    if exec_error and any(
        token in exec_error
        for token in DEFAULT_RECOVERY_SIGNAL_POLICY.execution_missing_table_substrings
    ):
        raw_error = attempt.execution_result.error or ""
        preview = raw_error[: DEFAULT_RECOVERY_SIGNAL_POLICY.execution_error_preview_chars]
        return f"execution_error: {preview}"

    critic = attempt.critic or {}
    repair_focus = (critic.get("repair_focus") or "").lower()
    if "schema_selection" in repair_focus or "missing_join" in repair_focus:
        return f"critic_repair_focus: {critic.get('repair_focus')}"
    for issue in critic.get("issues", []):
        issue_lower = issue.lower()
        if any(kw in issue_lower for kw in ("schema_selection", "missing_join", "missing table")):
            return f"critic_issue: {issue}"

    return None
