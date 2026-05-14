"""Task and answer domain models."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Task(BaseModel):
    """A benchmark task that is safe to use during generation."""

    instance_id: str
    db: str
    question: str
    external_knowledge: str | None = None


class CategoryMetadata(BaseModel):
    """Category metadata attached to one Spider2-Snow task."""

    model_config = ConfigDict(extra="forbid", strict=True)

    instance_id: str
    primary_tier: int = Field(ge=1, le=12)
    tags: list[str] = Field(default_factory=list)
    difficulty_notes: str | None = None

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, tags: list[str]) -> list[str]:
        if not tags:
            raise ValueError("tags must not be empty")
        if len(set(tags)) != len(tags):
            raise ValueError("tags must be unique")
        if any(not _is_snake_case(tag) for tag in tags):
            raise ValueError("tags must be snake_case")
        return tags


class Intent(BaseModel):
    """The question rewritten into entities, metrics, filters, and assumptions."""

    summary: str
    entities: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    native_value_terms: list[str] = Field(default_factory=list)
    derived_behavioral_definitions: list[str] = Field(default_factory=list)
    time_constraints: list[str] = Field(default_factory=list)
    answer_grain: str | None = None
    requested_ordering: list[str] = Field(default_factory=list)
    output_expectation: str
    assumptions: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    unsupported_assumptions: list[str] = Field(default_factory=list)
    do_not_assume: list[str] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    """The final per-task status recorded by the coordinator."""

    instance_id: str
    status: Literal["success", "failed", "skipped"]
    sql: str | None
    csv_path: str | None
    trace_path: str


def _is_snake_case(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", value))
