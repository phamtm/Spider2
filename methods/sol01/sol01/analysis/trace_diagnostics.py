"""Helpers for summarizing trace-level verification diagnostics."""

from __future__ import annotations

from typing import Any


def summarize_trace_diagnostics(trace: dict[str, Any]) -> str | None:
    """Return one compact diagnostic string for the latest attempt in a trace."""

    attempts = trace.get("attempts") or []
    if not isinstance(attempts, list) or not attempts:
        return _prompt_budget_diagnostic(trace)

    final_attempt = attempts[-1] if isinstance(attempts[-1], dict) else {}
    diagnostics = _attempt_diagnostics(final_attempt)
    return _join_diagnostics(
        _prompt_budget_diagnostic(trace),
        diagnostics["diagnostics"],
    )


def summarize_failed_question(trace: dict[str, Any]) -> dict[str, Any]:
    """Return a stable per-question summary of the final verification signals."""

    attempts = trace.get("attempts") or []
    final_attempt = attempts[-1] if isinstance(attempts, list) and attempts else {}
    diagnostics = _attempt_diagnostics(final_attempt if isinstance(final_attempt, dict) else {})
    prompt_budget_diagnostic = _prompt_budget_diagnostic(trace)
    if prompt_budget_diagnostic:
        diagnostics["diagnostics"] = _join_diagnostics(
            prompt_budget_diagnostic,
            diagnostics["diagnostics"],
        )
    return {
        "instance_id": trace.get("instance_id"),
        "question": trace.get("question"),
        "status": trace.get("status"),
        **diagnostics,
    }


def _join_diagnostics(*parts: str | None) -> str | None:
    """Join non-empty diagnostic fragments in the same format as attempt details."""

    values = [part for part in parts if part]
    return " | ".join(values) if values else None


def _prompt_budget_diagnostic(trace: dict[str, Any]) -> str | None:
    """Return prompt budget sizes from trace diagnostics when available."""

    budget = _prompt_budget(trace)
    if not budget:
        return None

    max_chars = budget.get("max_schema_prompt_chars")
    planning_chars = budget.get("planning_prompt_chars")
    context_chars = budget.get("sql_reference_context_chars")
    pieces: list[str] = []
    if isinstance(planning_chars, int):
        pieces.append(f"planning={planning_chars}")
    if isinstance(context_chars, int):
        pieces.append(f"context={context_chars}")
    if not pieces:
        return None
    suffix = f"/{max_chars}" if isinstance(max_chars, int) else ""
    return "prompt budget: " + ", ".join(f"{piece}{suffix}" for piece in pieces)


def _prompt_budget(trace: dict[str, Any]) -> dict[str, Any]:
    schema_selection = trace.get("schema_selection")
    if isinstance(schema_selection, dict):
        diagnostics = schema_selection.get("diagnostics")
        if isinstance(diagnostics, dict):
            budget = diagnostics.get("prompt_budget")
            if isinstance(budget, dict):
                return budget
    schema_retrieval = trace.get("schema_retrieval")
    if isinstance(schema_retrieval, dict):
        budget = schema_retrieval.get("prompt_budget")
        if isinstance(budget, dict):
            return budget
    return {}


def _attempt_diagnostics(attempt: dict[str, Any]) -> dict[str, Any]:
    """Summarize the verification checks that shaped one candidate."""

    verification_checks: list[str] = []
    failed_checks: list[str] = []
    repair_triggers: list[str] = []
    ranking_reasons: list[str] = []
    detail_parts: list[str] = []

    validation = attempt.get("validation") or {}
    execution = attempt.get("execution_result") or {}
    shape_report = attempt.get("shape_report") or {}
    filter_grounding_report = attempt.get("filter_grounding_report") or {}
    critic = attempt.get("critic") or {}
    score_breakdown = attempt.get("score_breakdown") or {}

    if isinstance(validation, dict):
        verification_checks.append("validation")
        if not validation.get("ok", True):
            failed_checks.append("validation")
            errors = [str(error) for error in validation.get("errors", []) if error]
            if errors:
                detail_parts.append(f"validation: {errors[0]}")

    if isinstance(execution, dict):
        verification_checks.append("execution")
        if not execution.get("ok", True):
            failed_checks.append("execution")
            error = execution.get("error")
            if error:
                detail_parts.append(f"execution: {error}")

    if isinstance(shape_report, dict):
        verification_checks.append("shape")
        violations = [str(item) for item in shape_report.get("violations", []) if item]
        if violations:
            failed_checks.append("shape")
            detail_parts.append(f"shape: {violations[0]}")

    if isinstance(filter_grounding_report, dict):
        verification_checks.append("filter_grounding")
        exact_filters = [
            str(item) for item in filter_grounding_report.get("exact_filters", []) if item
        ]
        value_rewrites = filter_grounding_report.get("value_rewrites") or []
        if value_rewrites:
            failed_checks.append("filter_grounding")
            rewrite = value_rewrites[0] if isinstance(value_rewrites[0], dict) else {}
            if rewrite.get("filter") and rewrite.get("rewrite"):
                detail_parts.append(
                    f"filter grounding: {rewrite['filter']} -> {rewrite['rewrite']}"
                )
        elif filter_grounding_report.get("reason"):
            detail_parts.append(f"filter grounding: {filter_grounding_report['reason']}")
        if exact_filters:
            detail_parts.append(f"filters: {', '.join(exact_filters[:2])}")

    if isinstance(critic, dict):
        verification_checks.append("critic")
        issues = [str(item) for item in critic.get("issues", []) if item]
        if critic.get("should_repair"):
            failed_checks.append("critic")
            repair_focus = critic.get("repair_focus")
            if repair_focus:
                repair_triggers.append(f"critic repair: {repair_focus}")
            else:
                repair_triggers.append("critic requested a repair")
        if issues:
            detail_parts.append(f"critic: {issues[0]}")

    if isinstance(score_breakdown, dict) and score_breakdown:
        ranking_reasons = _ranking_reasons(score_breakdown)
        if ranking_reasons:
            detail_parts.append("ranking: " + ", ".join(ranking_reasons))

    if not detail_parts:
        if failed_checks:
            detail_parts.append(f"failed checks: {', '.join(failed_checks)}")
        elif verification_checks:
            detail_parts.append(f"verification: {', '.join(dict.fromkeys(verification_checks))}")

    return {
        "verification_checks": list(dict.fromkeys(verification_checks)),
        "failed_checks": list(dict.fromkeys(failed_checks)),
        "repair_triggers": list(dict.fromkeys(repair_triggers)),
        "ranking_reasons": ranking_reasons,
        "diagnostics": " | ".join(detail_parts) if detail_parts else None,
    }


def _ranking_reasons(score_breakdown: dict[str, Any]) -> list[str]:
    """Return the strongest score contributions in descending impact order."""

    contributions: list[tuple[str, float]] = []
    for name, value in score_breakdown.items():
        if name == "confidence_tiebreaker":
            continue
        try:
            contributions.append((str(name), float(value)))
        except (TypeError, ValueError):
            continue

    contributions.sort(key=lambda item: (abs(item[1]), item[0]), reverse=True)
    return [f"{name}={value:+g}" for name, value in contributions[:3]]
