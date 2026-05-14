"""Shared pipeline state for one solver task."""

from __future__ import annotations

import dataclasses
from typing import Any

from sol01.candidates.selection import select_winner
from sol01.infra.config import SchemaContextConfig
from sol01.infra.policy import DEFAULT_SCHEMA_CONTEXT_POLICY, DEFAULT_SOLVER_POLICY, SolverPolicy
from sol01.llm.client import LLMClient
from sol01.models import AttemptRecord, Intent, SchemaSelection, Task


@dataclasses.dataclass
class TaskRun:
    """All state for one task as it flows through the solver pipeline stages."""

    task: Task
    client: LLMClient
    schema_context_config: SchemaContextConfig
    policy: SolverPolicy = dataclasses.field(default_factory=lambda: DEFAULT_SOLVER_POLICY)
    schema_context_version: str = DEFAULT_SCHEMA_CONTEXT_POLICY.version

    # Set during plan_schema
    intent: Intent | None = None
    schema: SchemaSelection | None = None
    table_schemas: dict[str, Any] = dataclasses.field(default_factory=dict)
    sql_reference_context: str = ""
    docs_context: str | None = None
    schema_context: dict[str, Any] = dataclasses.field(default_factory=dict)

    # Accumulated across stages
    prompt_hashes: dict[str, str] = dataclasses.field(default_factory=dict)
    attempts: list[AttemptRecord] = dataclasses.field(default_factory=list)
    candidate_review_payload: dict[str, Any] | None = None
    recovery_payload: dict[str, Any] | None = None


def current_best(run: TaskRun, *, preferred_stage: str | None = None) -> AttemptRecord | None:
    """Return the current best attempt for one task run."""

    selection = select_winner(run.attempts, preferred_stage=preferred_stage)
    return selection.attempt if selection is not None else None
