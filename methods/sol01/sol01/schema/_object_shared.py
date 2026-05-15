"""Shared schema-object types, constants, and low-level helpers."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from sol01.infra.policy import DEFAULT_SCHEMA_OBJECT_POLICY
from sol01.models import ColumnSchema, TableSchema

PrimitiveType = Literal["string", "numeric", "boolean", "temporal", "semi_structured", "unknown"]

DEFAULT_MAX_SAMPLE_VALUES_PER_COLUMN = DEFAULT_SCHEMA_OBJECT_POLICY.max_sample_values_per_column
DEFAULT_MAX_JOIN_CANDIDATES = DEFAULT_SCHEMA_OBJECT_POLICY.max_join_candidates
DEFAULT_MAX_JOIN_CANDIDATES_PER_COLUMN = DEFAULT_SCHEMA_OBJECT_POLICY.max_join_candidates_per_column
DEFAULT_MAX_FAMILY_VARIANT_COLUMNS = DEFAULT_SCHEMA_OBJECT_POLICY.max_family_variant_columns

_KEY_TOKENS = {"id", "ids", "identifier", "uuid", "guid", "key", "pk", "fk"}
_TIME_TOKENS = {
    "date",
    "datetime",
    "time",
    "timestamp",
    "created",
    "updated",
    "modified",
    "year",
    "month",
    "day",
}
_MEASURE_NAME_TOKENS = {
    "amount",
    "balance",
    "cost",
    "count",
    "duration",
    "freight",
    "lat",
    "latitude",
    "lng",
    "longitude",
    "measure",
    "price",
    "quantity",
    "rate",
    "score",
    "total",
    "value",
    "weight",
}
_CATEGORICAL_NAME_TOKENS = {
    "category",
    "class",
    "code",
    "enum",
    "flag",
    "level",
    "region",
    "segment",
    "state",
    "status",
    "tier",
    "type",
}
_TEXT_LIKE_NAME_TOKENS = {
    "address",
    "bio",
    "body",
    "comment",
    "description",
    "email",
    "html",
    "json",
    "message",
    "name",
    "notes",
    "payload",
    "text",
    "url",
}
_NUMERIC_TYPES = {
    "bigint",
    "decimal",
    "double",
    "float",
    "int",
    "integer",
    "number",
    "numeric",
    "real",
    "smallint",
}
_STRING_TYPES = {"char", "character", "nchar", "nvarchar", "string", "text", "varchar"}
_BOOLEAN_TYPES = {"bool", "boolean"}
_TEMPORAL_TYPES = {"date", "datetime", "time", "timestamp"}
_SEMI_STRUCTURED_TYPES = {"array", "binary", "blob", "json", "object", "variant"}

_DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:[ tT]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?$"
)
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_HEX_HASH_RE = re.compile(r"^[0-9a-fA-F]{24,}$")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_VERSION_SUFFIX_RE = re.compile(r"^(?P<stem>.+?)(?:[_-](?:v|ver|version)|v)(?P<value>\d+)$", re.I)
_YYYY_MM_DD_SUFFIX_RE = re.compile(
    r"^(?P<stem>.+?)[_-](?P<year>\d{4})[_-](?P<month>\d{2})[_-](?P<day>\d{2})$"
)
_YYYYMMDD_SUFFIX_RE = re.compile(
    r"^(?P<stem>.+?)[_-]?(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})$"
)
_YYYY_SUFFIX_RE = re.compile(r"^(?P<stem>.+?)[_-](?P<value>\d{4})$")
_INTEGER_SUFFIX_RE = re.compile(r"^(?P<stem>.+?)[_-](?P<value>\d+)$")


@dataclass(frozen=True)
class _ColumnRef:
    """One column and its resolved table context."""

    table_key: str
    table: TableSchema
    column: ColumnSchema
    table_full_name: str
    database_name: str | None
    schema_name: str | None


@dataclass(frozen=True)
class _TableFamilyMember:
    """One physical table considered for table-family detection."""

    table_key: str
    table: TableSchema
    table_full_name: str
    family_stem: str
    suffix_dimension: dict[str, object] | None
    signature: tuple[tuple[str, str], ...]
    column_names: frozenset[str]


@dataclass(frozen=True)
class _SampleValueDecision:
    """Decision details for one column's sample-value indexing policy."""

    include: bool
    values: list[object]
    sample_size: int
    distinct_count: int
    reason: str


def _is_key_like(column: ColumnSchema) -> bool:
    tokens = set(_tokens(column.name))
    normalized_name = _normalize_identifier(column.name)
    if tokens & _KEY_TOKENS:
        return True
    if normalized_name.endswith("_id"):
        return True
    description = column.description or ""
    return bool(re.search(r"\b(unique identifier|primary key|foreign key)\b", description, re.I))


def _is_time_like(column: ColumnSchema) -> bool:
    if _primitive_type(column.type) == "temporal":
        return True
    tokens = set(_tokens(column.name))
    if tokens & _TIME_TOKENS:
        return True
    description = column.description or ""
    return bool(re.search(r"\b(date|time|timestamp|datetime)\b", description, re.I))


def _is_numeric_measure_candidate(column: ColumnSchema) -> bool:
    if _primitive_type(column.type) != "numeric":
        return False
    tokens = set(_tokens(column.name))
    return bool(tokens & _MEASURE_NAME_TOKENS) or not _is_key_like(column)


def _primitive_type(raw_type: str | None) -> PrimitiveType:
    """Map database-specific type text to a coarse schema primitive."""

    if not raw_type:
        return "unknown"
    tokens = set(_tokens(raw_type))
    if tokens & _TEMPORAL_TYPES:
        return "temporal"
    if tokens & _NUMERIC_TYPES:
        return "numeric"
    if tokens & _BOOLEAN_TYPES:
        return "boolean"
    if tokens & _SEMI_STRUCTURED_TYPES:
        return "semi_structured"
    if tokens & _STRING_TYPES:
        return "string"
    return "unknown"


def _is_indexable_sample_value(value: object) -> bool:
    """Return True only for short human-readable categorical labels or codes."""

    if isinstance(value, bool):
        return True
    if isinstance(value, int | float):
        return False
    text = str(value).strip()
    if not text or len(text) > 80:
        return False
    if text.startswith(("{", "[", "<")):
        return False
    if "\n" in text or "\r" in text:
        return False
    if _UUID_RE.fullmatch(text) or _HEX_HASH_RE.fullmatch(text):
        return False
    if _EMAIL_RE.fullmatch(text) or _URL_RE.match(text):
        return False
    if _DATE_RE.fullmatch(text):
        return False
    if _looks_numeric(text):
        return False
    return True


def _stable_distinct(values: Iterable[object]) -> list[object]:
    """Return unique values sorted by their normalized text representation."""

    seen: dict[str, object] = {}
    for value in values:
        key = str(value).strip().casefold()
        if key:
            seen.setdefault(key, value)
    return [seen[key] for key in sorted(seen)]


def _table_full_name(table_key: str, table: TableSchema) -> str:
    return table.full_name or table_key or table.name


def _has_categorical_name(column_name: str) -> bool:
    return bool(set(_tokens(column_name)) & _CATEGORICAL_NAME_TOKENS)


def _has_text_like_name(column_name: str) -> bool:
    return bool(set(_tokens(column_name)) & _TEXT_LIKE_NAME_TOKENS)


def _looks_numeric(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _normalize_identifier(value: str) -> str:
    return "_".join(_tokens(value))


def _tokens(value: str) -> list[str]:
    """Split identifier or type text into lowercase alphanumeric tokens."""

    return [match.group(0).lower() for match in _TOKEN_RE.finditer(value)]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "_", value.lower()).strip("_")
    return slug or "group"


def _stable_hash(*parts: object) -> str:
    """Return the fixed-width hash used in schema object ids."""

    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def _first_non_empty(values: Iterable[str | None]) -> str | None:
    for value in values:
        if value:
            return value
    return None


def _join_text(parts: Iterable[object | None]) -> str:
    """Join non-empty pieces into searchable text."""

    return " ".join(str(part).strip() for part in parts if str(part or "").strip())


def _valid_month_day(month_text: str, day_text: str) -> bool:
    month = int(month_text)
    day = int(day_text)
    return 1 <= month <= 12 and 1 <= day <= 31
