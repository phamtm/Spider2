"""Schema domain models: objects, context, selection, and planning."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from sol01.models.task import Intent

SchemaObjectKind = Literal[
    "table",
    "column",
    "column_group",
    "sample_value",
    "join_candidate",
    "family",
]
SchemaContextChunkType = Literal[
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
    """One canonical database object available for planner selection."""

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


class SchemaContextChunk(BaseModel):
    """Prompt text derived from one schema object or linked document."""

    chunk_id: str
    object_id: str
    text: str = ""
    chunk_type: SchemaContextChunkType = "schema_object"
    parent_object_ids: list[str] = Field(default_factory=list)
    evidence_text: str = ""
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
    def _derive_text(self) -> SchemaContextChunk:
        if not self.text:
            self.text = (
                self.prompt_text
                or self.source_definition
                or self.inferred_usage
                or self.evidence_text
            )
        return self


class SchemaContextChunkEvidence(BaseModel):
    """One schema-context chunk attached to a logical schema object."""

    chunk: SchemaContextChunk
    rank: int = Field(ge=1)
    score: float | None = None


class SchemaContextObject(BaseModel):
    """A schema object plus the chunks that explain it to the planner."""

    schema_object: SchemaObject
    chunks: list[SchemaContextChunkEvidence] = Field(default_factory=list)
    rank: int = Field(ge=1)
    score: float | None = None


class SchemaPlanningConstraints(BaseModel):
    """Question constraints inferred while selecting schema objects."""

    date_start: str | None = None
    date_end: str | None = None
    years: list[int] = Field(default_factory=list)
    suffixes: list[str] = Field(default_factory=list)
    version: str | None = None
    include_all: bool = False
    notes: list[str] = Field(default_factory=list)


class SelectedSchemaObject(BaseModel):
    """One schema object selected for the compact resolved context."""

    object_id: str
    role: SelectedSchemaRole = "unknown"
    reason: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("object_id")
    @classmethod
    def _validate_object_id(cls, object_id: str) -> str:
        return validate_schema_object_id(object_id)


class ResolvedSchemaContext(BaseModel):
    """Compact context passed forward after schema selection."""

    db: str
    selected_objects: list[SelectedSchemaObject] = Field(default_factory=list)
    schema_context_objects: list[SchemaContextObject] = Field(default_factory=list)
    resolved_tables: list[str] = Field(default_factory=list)
    table_schemas: dict[str, TableSchema] = Field(default_factory=dict)
    prompt_context: str = ""
    diagnostics: dict[str, object] = Field(default_factory=dict)


class SchemaSelection(BaseModel):
    """Compact schema selection passed through the current planner."""

    db: str
    selected_object_ids: list[str] = Field(default_factory=list)
    expanded_tables: list[str] = Field(default_factory=list)
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    diagnostics: dict[str, object] = Field(default_factory=dict)

    @field_validator("selected_object_ids")
    @classmethod
    def _validate_selected_object_ids(cls, object_ids: list[str]) -> list[str]:
        return [validate_schema_object_id(object_id) for object_id in object_ids]


class SchemaPlanningDecision(BaseModel):
    """Future planner output that combines intent with selected schema objects."""

    selected_objects: list[SelectedSchemaObject] = Field(default_factory=list)
    constraints: SchemaPlanningConstraints = Field(default_factory=SchemaPlanningConstraints)
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    intent: Intent
