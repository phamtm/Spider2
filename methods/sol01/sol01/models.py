"""Typed data contracts shared across the sol01 pipeline."""

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
    time_constraints: list[str] = Field(default_factory=list)
    output_expectation: str
    assumptions: list[str] = Field(default_factory=list)


class ColumnSchema(BaseModel):
    """Compact metadata for one database column."""

    name: str
    type: str | None = None
    description: str | None = None
    sample_values: list[str] = Field(default_factory=list)


class TableSchema(BaseModel):
    """Schema text and examples used to decide which tables matter."""

    name: str
    database_name: str | None = None
    schema_name: str | None = None
    full_name: str | None = None
    ddl: str
    columns: list[ColumnSchema] = Field(default_factory=list)
    sample_rows: list[dict[str, object]] = Field(default_factory=list)
    searchable_text: str


class SchemaSelection(BaseModel):
    """The LLM-selected table set for one task."""

    db: str
    retrieval_mode: Literal["llm_only"] = "llm_only"
    selected_tables: list[str] = Field(default_factory=list)
    expanded_tables: list[str] = Field(default_factory=list)
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    selection_prompt_chars: int = Field(default=0, ge=0)
    candidate_table_count: int = Field(default=0, ge=0)


class TableSelectionDecision(BaseModel):
    """The LLM's table shortlist for one task."""

    selected_tables: list[str] = Field(default_factory=list)
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class MetricDefinition(BaseModel):
    """A task-aware metric definition extracted from allowed documents."""

    metric_name: str
    source_file: str | None = None
    heading: str | None = None
    definition: str
    formula: str | None = None
    sql_notes: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class SQLCandidate(BaseModel):
    """One SQL answer proposed by the LLM with its reasoning context."""

    sql: str
    explanation: str
    assumptions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ValidationReport(BaseModel):
    """Static read-only SQL validation results before execution."""

    ok: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    referenced_tables: list[str] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    """SQL execution output plus the CSV path when a result is written."""

    ok: bool
    row_count: int = Field(ge=0)
    columns: list[str] = Field(default_factory=list)
    sample_rows: list[dict[str, object]] = Field(default_factory=list)
    csv_path: str | None = None
    error: str | None = None


class ConfidenceReport(BaseModel):
    """Critic output that decides whether the best result needs repair."""

    confidence: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    should_repair: bool
    repair_focus: str | None = None


class CandidateComparisonReport(BaseModel):
    """Comparator output that picks the executable candidate to inspect next."""

    baseline_stage: str | None = None
    preferred_stage: str | None = None
    compared_stages: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    """The final per-task status recorded by the coordinator."""

    instance_id: str
    status: Literal["success", "failed", "skipped"]
    sql: str | None
    csv_path: str | None
    trace_path: str


def _is_snake_case(value: str) -> bool:
    """Return True when one tag uses the lower snake_case style."""

    return bool(re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", value))
