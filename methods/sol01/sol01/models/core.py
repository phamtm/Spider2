"""Typed data contracts shared across the sol01 pipeline."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SchemaObjectKind = Literal[
    "table",
    "column",
    "column_group",
    "sample_value",
    "join_candidate",
    "family",
]
RetrievalChunkType = Literal[
    "schema_object",
    "table",
    "column",
    "column_group",
    "join_candidate",
    "sample_value",
    "table_family",
]
SelectedSchemaRole = Literal[
    "primary",
    "supporting",
    "join",
    "filter",
    "metric",
    "dimension",
    "unknown",
]

# Stable schema object id formats:
# table:<full_table_name>
# column:<full_table_name>#<column_name>
# column_group:<full_table_name>#<group_slug>:<8char_hash>
# sample_value:<full_table_name>#<column_name>:<8char_hash>
# join_candidate:<left_table>#<left_col>-><right_table>#<right_col>:<8char_hash>
# family:<db>.<schema_or_none>:<stem_slug>:<8char_hash>
SCHEMA_OBJECT_ID_PATTERNS: dict[SchemaObjectKind, re.Pattern[str]] = {
    "table": re.compile(r"^table:[^#:\s]+$"),
    "column": re.compile(r"^column:[^#:\s]+#[^#:\s]+$"),
    "column_group": re.compile(r"^column_group:[^#:\s]+#[a-z0-9][a-z0-9_-]*:[0-9a-f]{8}$"),
    "sample_value": re.compile(r"^sample_value:[^#:\s]+#[^#:\s]+:[0-9a-f]{8}$"),
    "join_candidate": re.compile(
        r"^join_candidate:[^#:\s]+#[^#:\s]+->[^#:\s]+#[^#:\s]+:[0-9a-f]{8}$"
    ),
    "family": re.compile(r"^family:[^#:\s]+\.[^#:\s]+:[a-z0-9][a-z0-9_-]*:[0-9a-f]{8}$"),
}


def schema_object_id_kind(object_id: str) -> SchemaObjectKind | None:
    """Return the schema object kind encoded in one stable object id."""

    for object_type, pattern in SCHEMA_OBJECT_ID_PATTERNS.items():
        if pattern.fullmatch(object_id):
            return object_type
    return None


def is_schema_object_id(object_id: str) -> bool:
    """Return True when a value follows one stable schema object id format."""

    return schema_object_id_kind(object_id) is not None


def validate_schema_object_id(object_id: str) -> str:
    """Validate one stable schema object id and return it unchanged."""

    if not is_schema_object_id(object_id):
        raise ValueError("schema object id does not match a stable format")
    return object_id


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


class SchemaObject(BaseModel):
    """One canonical database object that can be indexed for retrieval."""

    object_id: str
    object_type: SchemaObjectKind
    name: str
    db: str | None = None
    table_name: str | None = None
    column_name: str | None = None
    description: str | None = None
    searchable_text: str = ""
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("object_id")
    @classmethod
    def _validate_object_id(cls, object_id: str) -> str:
        return validate_schema_object_id(object_id)

    @model_validator(mode="after")
    def _object_type_matches_id(self) -> SchemaObject:
        object_type = schema_object_id_kind(self.object_id)
        if object_type != self.object_type:
            raise ValueError("object_type must match object_id prefix")
        return self


class RetrievalChunk(BaseModel):
    """Searchable text derived from one schema object or linked document."""

    chunk_id: str
    object_id: str
    text: str = ""
    chunk_type: RetrievalChunkType = "schema_object"
    parent_object_ids: list[str] = Field(default_factory=list)
    bm25_text: str = ""
    prompt_text: str = ""
    source_definition: str = ""
    inferred_usage: str = ""
    source: Literal["schema", "linked_doc", "sample", "join", "family"] = "schema"
    linked_doc_title: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("object_id")
    @classmethod
    def _validate_object_id(cls, object_id: str) -> str:
        return validate_schema_object_id(object_id)

    @field_validator("parent_object_ids")
    @classmethod
    def _validate_parent_object_ids(cls, object_ids: list[str]) -> list[str]:
        return [validate_schema_object_id(object_id) for object_id in object_ids]

    @model_validator(mode="after")
    def _default_legacy_text(self) -> RetrievalChunk:
        if not self.text:
            self.text = (
                self.prompt_text or self.source_definition or self.inferred_usage or self.bm25_text
            )
        return self


class RetrievedChunk(BaseModel):
    """One retrieval hit with its local retrieval score preserved."""

    chunk: RetrievalChunk
    rank: int = Field(ge=1)
    score: float | None = None


class RetrievedSchemaObject(BaseModel):
    """A retrieved schema object plus the chunks that support it."""

    schema_object: SchemaObject
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    rank: int = Field(ge=1)
    score: float | None = None


class SelectionConstraints(BaseModel):
    """Optional hard bounds applied when selecting retrieved schema objects."""

    required_object_ids: list[str] = Field(default_factory=list)
    excluded_object_ids: list[str] = Field(default_factory=list)
    allowed_object_types: list[SchemaObjectKind] = Field(default_factory=list)
    max_objects: int | None = Field(default=None, ge=1)
    max_tables: int | None = Field(default=None, ge=1)
    max_columns_per_table: int | None = Field(default=None, ge=1)
    include_families: bool = True

    @field_validator("required_object_ids", "excluded_object_ids")
    @classmethod
    def _validate_object_ids(cls, object_ids: list[str]) -> list[str]:
        return [validate_schema_object_id(object_id) for object_id in object_ids]


class HybridPlanningConstraints(BaseModel):
    """Question constraints inferred while selecting retrieved schema objects."""

    date_start: str | None = None
    date_end: str | None = None
    years: list[int] = Field(default_factory=list)
    suffixes: list[str] = Field(default_factory=list)
    version: str | None = None
    include_all: bool = False
    notes: list[str] = Field(default_factory=list)


class SelectedSchemaObject(BaseModel):
    """One retrieved object selected for the compact resolved schema context."""

    object_id: str
    role: SelectedSchemaRole = "unknown"
    reason: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("object_id")
    @classmethod
    def _validate_object_id(cls, object_id: str) -> str:
        return validate_schema_object_id(object_id)


class ResolvedSchemaContext(BaseModel):
    """Compact context passed forward after retrieval and selection."""

    db: str
    selected_objects: list[SelectedSchemaObject] = Field(default_factory=list)
    retrieved_objects: list[RetrievedSchemaObject] = Field(default_factory=list)
    resolved_tables: list[str] = Field(default_factory=list)
    allowed_tables: list[str] = Field(default_factory=list)
    table_schemas: dict[str, TableSchema] = Field(default_factory=dict)
    prompt_context: str = ""
    schema_prompt: str = ""
    resolution_diagnostics: dict[str, object] = Field(default_factory=dict)
    diagnostics: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sync_resolver_aliases(self) -> ResolvedSchemaContext:
        """Keep older and newer resolver field names equivalent."""

        if self.prompt_context and not self.schema_prompt:
            self.schema_prompt = self.prompt_context
        elif self.schema_prompt and not self.prompt_context:
            self.prompt_context = self.schema_prompt

        if self.resolution_diagnostics and not self.diagnostics:
            self.diagnostics = self.resolution_diagnostics
        elif self.diagnostics and not self.resolution_diagnostics:
            self.resolution_diagnostics = self.diagnostics
        return self


class SchemaSelection(BaseModel):
    """Compact schema selection passed through the current planner."""

    db: str
    selected_object_ids: list[str] = Field(default_factory=list)
    selected_tables: list[str] = Field(default_factory=list)
    expanded_tables: list[str] = Field(default_factory=list)
    allowed_tables: list[str] = Field(default_factory=list)
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    diagnostics: dict[str, object] = Field(default_factory=dict)

    @field_validator("selected_object_ids")
    @classmethod
    def _validate_selected_object_ids(cls, object_ids: list[str]) -> list[str]:
        return [validate_schema_object_id(object_id) for object_id in object_ids]


class PlanningDecision(BaseModel):
    """Combined table selection and answer-contract planning output."""

    selected_tables: list[str] = Field(default_factory=list)
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    intent: Intent


class HybridPlanningDecision(BaseModel):
    """Future planner output that combines intent with selected schema objects."""

    selected_objects: list[SelectedSchemaObject] = Field(default_factory=list)
    selected_tables: list[str] = Field(default_factory=list)
    constraints: HybridPlanningConstraints = Field(default_factory=HybridPlanningConstraints)
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    intent: Intent


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


class SchemaExpansionDecision(BaseModel):
    """LLM decision on whether to widen the table selection after a failure."""

    should_expand: bool
    additional_tables: list[str] = Field(default_factory=list)
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


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
