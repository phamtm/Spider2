"""Shared prompt-budget helpers for sol01 prompts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sol01.infra.policy import DEFAULT_PROMPT_BUDGET_POLICY, PromptBudgetPolicy


class PromptBudgetExceededError(ValueError):
    """Raised when a prompt cannot fit inside the configured character budget."""


def enforce_prompt_budget(prompt_name: str, prompt: str, max_chars: int) -> str:
    """Return a prompt only when it fits the configured character budget."""

    if len(prompt) > max_chars:
        raise PromptBudgetExceededError(
            f"{prompt_name} prompt is {len(prompt)} chars; configured budget is {max_chars}"
        )
    return prompt


@dataclass(frozen=True)
class PromptBudgetFit:
    """Final bounded prompt plus the effective limits used to render it."""

    prompt: str
    docs_limit: int
    evidence_limit: int
    rounds: int


def fit_prompt_budget(
    *,
    prompt_name: str,
    total_limit: int,
    docs_limit: int,
    evidence_limit: int,
    render_prompt: Callable[[int, int], str],
    strategy: PromptBudgetPolicy = DEFAULT_PROMPT_BUDGET_POLICY,
) -> PromptBudgetFit:
    """Shrink lossy context in a named order until the prompt fits."""

    prompt = render_prompt(docs_limit, evidence_limit)
    if len(prompt) <= total_limit:
        return PromptBudgetFit(
            prompt=prompt,
            docs_limit=docs_limit,
            evidence_limit=evidence_limit,
            rounds=0,
        )

    for round_index in range(1, strategy.max_shrink_rounds + 1):
        overflow = len(prompt) - total_limit
        target = _next_shrink_target(
            docs_limit=docs_limit,
            evidence_limit=evidence_limit,
            strategy=strategy,
        )
        if target is None:
            break
        if target == "evidence":
            evidence_limit = max(
                0,
                evidence_limit - overflow - strategy.shrink_overflow_buffer_chars,
            )
        else:
            docs_limit = max(
                0,
                docs_limit - overflow - strategy.shrink_overflow_buffer_chars,
            )
        prompt = render_prompt(docs_limit, evidence_limit)
        if len(prompt) <= total_limit:
            return PromptBudgetFit(
                prompt=prompt,
                docs_limit=docs_limit,
                evidence_limit=evidence_limit,
                rounds=round_index,
            )

    return PromptBudgetFit(
        prompt=enforce_prompt_budget(prompt_name, prompt, total_limit),
        docs_limit=docs_limit,
        evidence_limit=evidence_limit,
        rounds=strategy.max_shrink_rounds,
    )


def _next_shrink_target(
    *,
    docs_limit: int,
    evidence_limit: int,
    strategy: PromptBudgetPolicy,
) -> str | None:
    for target in strategy.shrink_order:
        if target == "evidence" and evidence_limit > 0:
            return target
        if target == "docs" and docs_limit > 0:
            return target
    return None
