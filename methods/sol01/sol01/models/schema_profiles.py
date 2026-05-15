"""Schema-profile artifact models used for generated per-database compaction."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ProfileAbstractionKind = Literal["table", "exact_family", "near_family", "wide_table"]

FORBIDDEN_SCHEMA_PROFILE_TOKENS = (
    "gold sql",
    "gold_sql",
    "answer:",
    "expected answer",
    "benchmark question",
    "instance_id",
    "sf_bq",
    "sf_local",
    "trace",
    "validation error",
    "unknown column",
)

_QUESTION_LIKE_TEXT_RE = re.compile(
    r"\b(?:what|which|show|find|given|list|return|provide)\b.{0,120}\?",
    re.IGNORECASE,
)


class TableProfile(BaseModel):
    """Generated compact profile for one physical table."""

    model_config = ConfigDict(extra="forbid", strict=True)

    profile_id: str
    abstraction_kind: Literal["table", "wide_table"]
    table_name: str
    covered_tables: list[str] = Field(min_length=1)
    grain_hint: str
    key_columns: list[str] = Field(default_factory=list)
    time_columns: list[str] = Field(default_factory=list)
    measure_columns: list[str] = Field(default_factory=list)
    dimension_columns: list[str] = Field(default_factory=list)
    repeated_column_templates: list[str] = Field(default_factory=list)
    join_anchors: list[str] = Field(default_factory=list)
    naming_rules: list[str] = Field(default_factory=list)
    compact_semantic_summary: str
    aliases: list[str] = Field(default_factory=list)
    theme_terms: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    caveats: list[str] = Field(default_factory=list)
    provenance_inputs: list[str] = Field(default_factory=list)
    source_column_count: int = Field(ge=0)
    source_sample_row_count: int = Field(ge=0)

    @field_validator("profile_id")
    @classmethod
    def _validate_profile_id(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", value):
            raise ValueError("profile_id must be lower_snake_case")
        return value

    @field_validator(
        "table_name",
        "covered_tables",
        "grain_hint",
        "key_columns",
        "time_columns",
        "measure_columns",
        "dimension_columns",
        "repeated_column_templates",
        "join_anchors",
        "naming_rules",
        "compact_semantic_summary",
        "aliases",
        "theme_terms",
        "caveats",
        "provenance_inputs",
    )
    @classmethod
    def _reject_tuning_leaks(cls, value: object) -> object:
        return _validate_profile_text(value)

    @model_validator(mode="after")
    def _validate_shape(self) -> TableProfile:
        if self.table_name not in self.covered_tables:
            raise ValueError("table_name must appear in covered_tables")
        return self


class FamilyProfile(BaseModel):
    """Generated compact profile for one repeated table family."""

    model_config = ConfigDict(extra="forbid", strict=True)

    profile_id: str
    abstraction_kind: Literal["exact_family", "near_family"]
    family_selector: str
    covered_tables: list[str] = Field(min_length=2)
    canonical_table: str
    grain_hint: str
    key_columns: list[str] = Field(default_factory=list)
    time_columns: list[str] = Field(default_factory=list)
    measure_columns: list[str] = Field(default_factory=list)
    dimension_columns: list[str] = Field(default_factory=list)
    repeated_column_templates: list[str] = Field(default_factory=list)
    join_anchors: list[str] = Field(default_factory=list)
    naming_rules: list[str] = Field(default_factory=list)
    compact_semantic_summary: str
    aliases: list[str] = Field(default_factory=list)
    theme_terms: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    caveats: list[str] = Field(default_factory=list)
    provenance_inputs: list[str] = Field(default_factory=list)
    member_count: int = Field(ge=2)

    @field_validator("profile_id")
    @classmethod
    def _validate_profile_id(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", value):
            raise ValueError("profile_id must be lower_snake_case")
        return value

    @field_validator(
        "family_selector",
        "covered_tables",
        "canonical_table",
        "grain_hint",
        "key_columns",
        "time_columns",
        "measure_columns",
        "dimension_columns",
        "repeated_column_templates",
        "join_anchors",
        "naming_rules",
        "compact_semantic_summary",
        "aliases",
        "theme_terms",
        "caveats",
        "provenance_inputs",
    )
    @classmethod
    def _reject_tuning_leaks(cls, value: object) -> object:
        return _validate_profile_text(value)

    @model_validator(mode="after")
    def _validate_shape(self) -> FamilyProfile:
        if self.canonical_table not in self.covered_tables:
            raise ValueError("canonical_table must appear in covered_tables")
        if self.member_count != len(self.covered_tables):
            raise ValueError("member_count must match covered_tables length")
        return self


class SchemaProfileCatalog(BaseModel):
    """All generated table and family profiles for one database."""

    model_config = ConfigDict(extra="forbid", strict=True)

    db: str
    source_schema_hash: str
    table_profiles: list[TableProfile] = Field(default_factory=list)
    family_profiles: list[FamilyProfile] = Field(default_factory=list)
    db_overview: str | None = None
    aliases: list[str] = Field(default_factory=list)
    theme_terms: list[str] = Field(default_factory=list)

    @field_validator("db")
    @classmethod
    def _validate_db(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("db must be non-empty")
        return normalized

    @field_validator("db_overview", "aliases", "theme_terms")
    @classmethod
    def _reject_tuning_leaks(cls, value: object) -> object:
        return _validate_profile_text(value)

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> SchemaProfileCatalog:
        profile_ids = [
            profile.profile_id for profile in [*self.table_profiles, *self.family_profiles]
        ]
        if len(set(profile_ids)) != len(profile_ids):
            raise ValueError("profile_id values must be unique within one catalog")
        return self


class SchemaProfileManifest(BaseModel):
    """Provenance manifest stored beside one per-database profile catalog."""

    model_config = ConfigDict(extra="forbid", strict=True)

    db: str
    source_schema_hash: str
    builder_version: str
    summarizer_version: str
    prompt_template_version: str
    generated_at: str
    artifact_hash: str
    table_profile_count: int = Field(ge=0)
    family_profile_count: int = Field(ge=0)


def _validate_profile_text(value: object) -> object:
    text = repr(value).casefold()
    for token in FORBIDDEN_SCHEMA_PROFILE_TOKENS:
        if token in text:
            raise ValueError(f"schema profile contains forbidden hint token: {token}")
    if isinstance(value, str) and "?" in value:
        raise ValueError("schema profile text must stay declarative, not question-shaped")
    if isinstance(value, str) and _QUESTION_LIKE_TEXT_RE.search(value):
        raise ValueError("schema profile text looks derived from a benchmark question")
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and "?" in item:
                raise ValueError("schema profile text must stay declarative, not question-shaped")
    return value
