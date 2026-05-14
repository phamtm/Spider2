"""SQL candidate, execution, and evaluation domain models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class SQLCandidate(BaseModel):
    """One SQL answer proposed by the LLM with its reasoning context."""

    sql: str
    explanation: str
    assumptions: list[str] = Field(default_factory=list)
    constraint_ledger: list[str] = Field(default_factory=list)
    unsupported_assumptions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class SQLCandidateBatch(BaseModel):
    """Several SQL candidates generated from one prompt."""

    candidates: list[SQLCandidate] = Field(default_factory=list)


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


class FilterGroundingReport(BaseModel):
    """Evidence gathered when exact string filters need value rewrites."""

    exact_filters: list[str] = Field(default_factory=list)
    probes: list[dict[str, object]] = Field(default_factory=list)
    value_rewrites: list[dict[str, object]] = Field(default_factory=list)
    zero_like_result: bool = False
    reason: str


class OutputShapeReport(BaseModel):
    """Shape analysis for one SQL candidate and its executed result."""

    expected_columns: list[str] = Field(default_factory=list)
    observed_columns: list[str] = Field(default_factory=list)
    projected_columns: list[str] = Field(default_factory=list)
    grouped_columns: list[str] = Field(default_factory=list)
    missing_columns: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)


class CandidateEvidence(BaseModel):
    """Deterministic check results for one evaluated SQL candidate."""

    executable: bool
    validation_ok: bool
    validation_errors: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    issues: list[str] = Field(default_factory=list)


class ConfidenceReport(BaseModel):
    """Repair-review output that decides whether a candidate needs another attempt."""

    confidence: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    should_repair: bool
    repair_focus: str | None = None


class AggregateGrainReport(BaseModel):
    """Heuristic grain analysis used to rank aggregate candidates."""

    inferred_grain: Literal["row_count", "distinct_entity_count", "value_count", "unknown"]
    reason: str
    distinct_reason: str | None = None
    uses_distinct: bool = False
    has_joins: bool = False
    selected_tables: list[str] = Field(default_factory=list)


class CandidateReviewReport(BaseModel):
    """Single review pass that can compare candidates and request repair."""

    baseline_stage: str | None = None
    preferred_stage: str | None = None
    compared_stages: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    should_repair: bool
    repair_focus: str | None = None


class AttemptRecord(BaseModel):
    """Typed record for one evaluated SQL candidate through the pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stage: str
    sql: str
    explanation: str
    assumptions: list[str] = Field(default_factory=list)
    constraint_ledger: list[str] = Field(default_factory=list)
    unsupported_assumptions: list[str] = Field(default_factory=list)
    candidate_confidence: float
    validation: ValidationReport
    execution_result: ExecutionResult
    filter_grounding_report: FilterGroundingReport | None = None
    shape_report: OutputShapeReport | None = None
    evidence: CandidateEvidence | None = None
    score: float
    result_profile: dict[str, Any] | None = None
    aggregate_grain: AggregateGrainReport | None = None
    elapsed_seconds: float = 0.0
    critic: dict[str, Any] | None = None
    candidate_review: dict[str, Any] | None = None
    repair_skipped_reason: str | None = None

    _dataframe: Any = PrivateAttr(default=None)
