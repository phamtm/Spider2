"""Load curated summaries for large Spider2-Snow schemas."""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sol01.infra.paths import REPO_ROOT

DEFAULT_LARGE_SCHEMA_SUMMARY_PATH = (
    REPO_ROOT / "methods" / "sol01" / "metadata" / "large_schema_summaries.json"
)

FORBIDDEN_SUMMARY_TOKENS = (
    "gold sql",
    "gold_sql",
    "answer:",
    "expected answer",
    "benchmark question",
    "instance_id",
    "sf_bq",
    "sf_local",
)


class SchemaCopy(BaseModel):
    """One database/schema location that shares a curated summary."""

    model_config = ConfigDict(extra="forbid", strict=True)

    database: str
    schema_name: str

    @field_validator("database", "schema_name")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("database and schema_name must be non-empty")
        if "." in normalized:
            raise ValueError("database and schema_name must not contain dots")
        return normalized

    def matches(self, database: str, schema_name: str) -> bool:
        """Return True when this copy covers one resolved table scope."""

        database_matches = not database or self.database.casefold() == database.casefold()
        return database_matches and self.schema_name.casefold() == schema_name.casefold()


class TableRange(BaseModel):
    """Inclusive table-suffix range parsed from a table-family regex."""

    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["yyyymmdd", "quarter"]
    group: str | None = None
    year_group: str | None = None
    quarter_group: str | None = None
    start: str
    end: str

    @model_validator(mode="after")
    def _required_groups_match_kind(self) -> TableRange:
        if self.kind == "yyyymmdd" and not self.group:
            raise ValueError("yyyymmdd ranges require group")
        if self.kind == "quarter" and (not self.year_group or not self.quarter_group):
            raise ValueError("quarter ranges require year_group and quarter_group")
        if self._range_key(self.start) > self._range_key(self.end):
            raise ValueError("range start must be before or equal to range end")
        return self

    def contains(self, match: re.Match[str]) -> bool:
        """Return True when a regex match falls inside this inclusive range."""

        if self.kind == "yyyymmdd":
            raw_value = match.group(self.group or "")
        else:
            raw_value = (
                f"{match.group(self.year_group or '')}_Q{match.group(self.quarter_group or '')}"
            )
        value = self._range_key(raw_value)
        return self._range_key(self.start) <= value <= self._range_key(self.end)

    def has_required_groups(self, pattern: re.Pattern[str]) -> bool:
        """Return True when a compiled table regex exposes this range's groups."""

        if self.kind == "yyyymmdd":
            return bool(self.group and self.group in pattern.groupindex)
        return bool(
            self.year_group
            and self.quarter_group
            and self.year_group in pattern.groupindex
            and self.quarter_group in pattern.groupindex
        )

    def _range_key(self, value: str) -> tuple[int, ...]:
        if self.kind == "yyyymmdd":
            if not re.fullmatch(r"\d{8}", value):
                raise ValueError("yyyymmdd range values must use YYYYMMDD")
            return (int(value),)
        match = re.fullmatch(r"(?P<year>\d{4})_Q(?P<quarter>[1-4])", value)
        if not match:
            raise ValueError("quarter range values must use YYYY_QN")
        return (int(match.group("year")), int(match.group("quarter")))


class TableMatchRule(BaseModel):
    """Exact-table or regex-family matching rule for one summary."""

    model_config = ConfigDict(extra="forbid", strict=True)

    table_names: list[str] = Field(default_factory=list)
    table_pattern: str | None = None
    inclusive_range: TableRange | None = None

    @model_validator(mode="after")
    def _validate_match_mode(self) -> TableMatchRule:
        has_exact = bool(self.table_names)
        has_pattern = bool(self.table_pattern)
        if has_exact == has_pattern:
            raise ValueError("match rule must define exactly one of table_names or table_pattern")
        if self.inclusive_range and not self.table_pattern:
            raise ValueError("inclusive_range requires table_pattern")
        if self.table_pattern:
            try:
                pattern = re.compile(self.table_pattern)
            except re.error as exc:
                raise ValueError(f"invalid table_pattern: {exc}") from exc
            if self.inclusive_range and not self.inclusive_range.has_required_groups(pattern):
                raise ValueError("table_pattern does not define required inclusive_range groups")
        if len(set(name.casefold() for name in self.table_names)) != len(self.table_names):
            raise ValueError("table_names must be unique case-insensitively")
        return self

    def matches(self, table_name: str) -> bool:
        """Return True when one physical table name matches this rule."""

        if self.table_names:
            return table_name.casefold() in {name.casefold() for name in self.table_names}

        assert self.table_pattern is not None
        match = re.fullmatch(self.table_pattern, table_name, flags=re.IGNORECASE)
        if match is None:
            return False
        if self.inclusive_range is None:
            return True
        return self.inclusive_range.contains(match)


class LargeSchemaSummary(BaseModel):
    """Curated database-level summary for one exact table set or table family."""

    model_config = ConfigDict(extra="forbid", strict=True)

    summary_id: str
    schema_copies: list[SchemaCopy]
    match: TableMatchRule
    purpose: str
    grain: str
    stable_columns: list[str]
    repeated_column_rules: list[str]
    inclusive_ranges: list[str] = Field(default_factory=list)
    quote_spelling_rules: list[str]
    examples: list[str]
    aliases: list[str]

    @field_validator("summary_id")
    @classmethod
    def _validate_summary_id(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", value):
            raise ValueError("summary_id must be lower_snake_case")
        return value

    @field_validator(
        "purpose",
        "grain",
        "stable_columns",
        "repeated_column_rules",
        "inclusive_ranges",
        "quote_spelling_rules",
        "examples",
        "aliases",
    )
    @classmethod
    def _reject_question_or_answer_hints(cls, value: object) -> object:
        text = json.dumps(value, sort_keys=True).casefold()
        for token in FORBIDDEN_SUMMARY_TOKENS:
            if token in text:
                raise ValueError(f"summary metadata contains forbidden hint token: {token}")
        return value

    @model_validator(mode="after")
    def _validate_summary_shape(self) -> LargeSchemaSummary:
        if not self.schema_copies:
            raise ValueError("schema_copies must not be empty")
        if not self.stable_columns:
            raise ValueError("stable_columns must not be empty")
        if not self.repeated_column_rules:
            raise ValueError("repeated_column_rules must not be empty")
        if not self.quote_spelling_rules:
            raise ValueError("quote_spelling_rules must not be empty")
        if not 3 <= len(self.examples) <= 5:
            raise ValueError("examples must contain 3 to 5 exact names")
        if not self.aliases:
            raise ValueError("aliases must not be empty")
        for example in self.examples:
            if not self.match.matches(_example_table_name(example)):
                raise ValueError(f"example {example!r} does not match summary rule")
        return self

    def matches_table(self, database: str, schema_name: str, table_name: str) -> bool:
        """Return True when this summary covers one physical table."""

        return any(copy.matches(database, schema_name) for copy in self.schema_copies) and (
            self.match.matches(table_name)
        )


class LargeSchemaSummaryRegistry(BaseModel):
    """Validated collection of curated large-schema summaries."""

    model_config = ConfigDict(extra="forbid", strict=True)

    summaries: list[LargeSchemaSummary]

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> LargeSchemaSummaryRegistry:
        summary_ids = [summary.summary_id for summary in self.summaries]
        if len(set(summary_ids)) != len(summary_ids):
            raise ValueError("summary_id values must be unique")
        return self

    def match_table_ref(self, table_ref: str) -> list[LargeSchemaSummary]:
        """Return summaries matching DB.SCHEMA.TABLE or SCHEMA.TABLE references."""

        database, schema_name, table_name = _split_table_ref(table_ref)
        return [
            summary
            for summary in self.summaries
            if summary.matches_table(database, schema_name, table_name)
        ]

    def match_table(
        self,
        *,
        database: str,
        schema_name: str,
        table_name: str,
    ) -> list[LargeSchemaSummary]:
        """Return summaries matching explicit table identity parts."""

        return [
            summary
            for summary in self.summaries
            if summary.matches_table(database, schema_name, table_name)
        ]


def load_large_schema_summary_registry(
    path: Path = DEFAULT_LARGE_SCHEMA_SUMMARY_PATH,
) -> LargeSchemaSummaryRegistry:
    """Load and validate the curated large-schema summary registry."""

    signature = _path_signature(path)
    return _load_large_schema_summary_registry(str(path.resolve()), signature)


@cache
def _load_large_schema_summary_registry(
    path: str,
    signature: tuple[int, int],
) -> LargeSchemaSummaryRegistry:
    raw_payload = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        return LargeSchemaSummaryRegistry.model_validate(raw_payload)
    except ValueError as exc:
        raise ValueError(f"invalid large schema summary registry {path}: {exc}") from exc


def _path_signature(path: Path) -> tuple[int, int]:
    try:
        stat_result = path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"large schema summary registry not found: {path}") from exc
    return stat_result.st_mtime_ns, stat_result.st_size


def _split_table_ref(table_ref: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in table_ref.split(".") if part.strip()]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return "", parts[0], parts[1]
    raise ValueError("table_ref must use DB.SCHEMA.TABLE or SCHEMA.TABLE")


def _example_table_name(example: str) -> str:
    return example.rsplit(".", maxsplit=1)[-1].strip()
