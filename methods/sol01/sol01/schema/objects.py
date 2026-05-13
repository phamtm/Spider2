"""Build canonical schema objects from table metadata."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from sol01.infra.config import DEFAULT_FAMILY_SIMILARITY_THRESHOLD
from sol01.models import ColumnSchema, SchemaObject, TableSchema

PrimitiveType = Literal["string", "numeric", "boolean", "temporal", "semi_structured", "unknown"]

DEFAULT_MAX_SAMPLE_VALUES_PER_COLUMN = 20
DEFAULT_MAX_JOIN_CANDIDATES = 2_000
DEFAULT_MAX_JOIN_CANDIDATES_PER_COLUMN = 100
DEFAULT_MAX_FAMILY_VARIANT_COLUMNS = 20

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


def build_schema_objects(
    db_index: Mapping[str, TableSchema],
    *,
    max_sample_values_per_column: int = DEFAULT_MAX_SAMPLE_VALUES_PER_COLUMN,
    max_join_candidates: int = DEFAULT_MAX_JOIN_CANDIDATES,
    max_join_candidates_per_column: int = DEFAULT_MAX_JOIN_CANDIDATES_PER_COLUMN,
    family_similarity_threshold: float = DEFAULT_FAMILY_SIMILARITY_THRESHOLD,
) -> list[SchemaObject]:
    """Build deterministic schema objects from a database table index."""

    from sol01.schema.object_families import table_family_objects  # noqa: PLC0415
    from sol01.schema.object_joins import join_candidate_objects  # noqa: PLC0415
    from sol01.schema.object_sample_values import sample_value_objects  # noqa: PLC0415

    column_refs = _column_refs(db_index)
    objects: list[SchemaObject] = []

    for table_key in sorted(db_index):
        table = db_index[table_key]
        table_full_name = _table_full_name(table_key, table)
        objects.append(_table_object(table_key, table, table_full_name))
        objects.extend(_column_object(ref) for ref in column_refs if ref.table_key == table_key)
        objects.extend(_column_group_objects(table_key, table, table_full_name))
        objects.extend(
            sample_value_objects(
                table_key,
                table,
                table_full_name,
                max_sample_values_per_column=max_sample_values_per_column,
            )
        )

    objects.extend(
        table_family_objects(
            db_index,
            family_similarity_threshold=family_similarity_threshold,
        )
    )
    objects.extend(
        join_candidate_objects(
            column_refs,
            max_join_candidates=max_join_candidates,
            max_join_candidates_per_column=max_join_candidates_per_column,
        )
    )
    return objects


def _column_refs(db_index: Mapping[str, TableSchema]) -> list[_ColumnRef]:
    """Flatten table columns into sorted references for deterministic processing."""

    refs: list[_ColumnRef] = []
    for table_key in sorted(db_index):
        table = db_index[table_key]
        for column in sorted(table.columns, key=lambda item: item.name):
            refs.append(
                _ColumnRef(
                    table_key=table_key,
                    table=table,
                    column=column,
                    table_full_name=_table_full_name(table_key, table),
                    database_name=table.database_name,
                    schema_name=table.schema_name,
                )
            )
    return refs


def _table_object(table_key: str, table: TableSchema, table_full_name: str) -> SchemaObject:
    """Create the table-level object with table provenance and family hooks."""

    columns = [column.name for column in table.columns]
    description_parts = [column.description for column in table.columns if column.description]
    searchable_parts = [
        table_full_name,
        table.name,
        table.searchable_text,
        " ".join(columns),
        " ".join(description_parts),
    ]
    return SchemaObject(
        object_id=f"table:{table_full_name}",
        object_type="table",
        name=table_full_name,
        db=table.database_name,
        table_name=table_full_name,
        description=_first_non_empty(description_parts),
        searchable_text=_join_text(searchable_parts),
        metadata={
            "source": "load_db_index",
            "source_table_key": table_key,
            "database_name": table.database_name,
            "schema_name": table.schema_name,
            "full_name": table_full_name,
            "short_name": table.name,
            "ddl": table.ddl,
            "columns": [column.model_dump(mode="json") for column in table.columns],
            "sample_row_count": len(table.sample_rows),
            "normalized_table_name": _normalize_identifier(table.name),
            "name_tokens": _tokens(table.name),
            "family_hooks": {
                "database_name": table.database_name,
                "schema_name": table.schema_name,
                "normalized_table_name": _normalize_identifier(table.name),
                "name_tokens": _tokens(table.name),
            },
            "provenance": {"source": "load_db_index", "table_key": table_key},
        },
    )


def _column_object(ref: _ColumnRef) -> SchemaObject:
    """Create the column-level object with type and source metadata preserved."""

    primitive_type = _primitive_type(ref.column.type)
    searchable_parts = [
        ref.table_full_name,
        ref.column.name,
        ref.column.type,
        ref.column.description,
    ]
    return SchemaObject(
        object_id=f"column:{ref.table_full_name}#{ref.column.name}",
        object_type="column",
        name=f"{ref.table_full_name}.{ref.column.name}",
        db=ref.database_name,
        table_name=ref.table_full_name,
        column_name=ref.column.name,
        description=ref.column.description,
        searchable_text=_join_text(searchable_parts),
        metadata={
            "source": "load_db_index",
            "source_table_key": ref.table_key,
            "database_name": ref.database_name,
            "schema_name": ref.schema_name,
            "table_full_name": ref.table_full_name,
            "column_name": ref.column.name,
            "column_type": ref.column.type,
            "primitive_type": primitive_type,
            "description": ref.column.description,
            "sample_values": list(ref.column.sample_values),
            "is_key_like": _is_key_like(ref.column),
            "is_time_like": _is_time_like(ref.column),
            "normalized_column_name": _normalize_identifier(ref.column.name),
            "provenance": {"source": "load_db_index", "table_key": ref.table_key},
        },
    )


def _column_group_objects(
    table_key: str,
    table: TableSchema,
    table_full_name: str,
) -> list[SchemaObject]:
    """Create deterministic structural column groups for one table."""

    groups: list[tuple[str, str, list[ColumnSchema]]] = []
    key_columns = [column for column in table.columns if _is_key_like(column)]
    time_columns = [column for column in table.columns if _is_time_like(column)]
    measure_columns = [
        column
        for column in table.columns
        if _is_numeric_measure_candidate(column)
        and not _is_key_like(column)
        and not _is_time_like(column)
    ]
    if key_columns:
        groups.append(("key_like", "Key-like columns", key_columns))
    if time_columns:
        groups.append(("time_like", "Time-like columns", time_columns))
    if measure_columns:
        groups.append(("numeric_measures", "Numeric measure candidates", measure_columns))
    groups.extend(_repeated_prefix_groups(table.columns))

    objects: list[SchemaObject] = []
    for group_slug, group_name, columns in groups:
        column_names = sorted({column.name for column in columns})
        if not column_names:
            continue
        group_hash = _stable_hash(table_full_name, group_slug, *column_names)
        object_id = f"column_group:{table_full_name}#{group_slug}:{group_hash}"
        objects.append(
            SchemaObject(
                object_id=object_id,
                object_type="column_group",
                name=f"{table_full_name}.{group_name}",
                db=table.database_name,
                table_name=table_full_name,
                searchable_text=_join_text([table_full_name, group_name, " ".join(column_names)]),
                metadata={
                    "source": "load_db_index",
                    "source_table_key": table_key,
                    "database_name": table.database_name,
                    "schema_name": table.schema_name,
                    "table_full_name": table_full_name,
                    "group_type": group_slug,
                    "columns": column_names,
                    "provenance": {"source": "load_db_index", "table_key": table_key},
                },
            )
        )
    return objects


def _repeated_prefix_groups(
    columns: Iterable[ColumnSchema],
) -> list[tuple[str, str, list[ColumnSchema]]]:
    """Find repeated-prefix groups using column-name structure only."""

    candidates: dict[str, list[ColumnSchema]] = defaultdict(list)
    for column in columns:
        tokens = _tokens(column.name)
        for prefix_length in range(1, min(len(tokens), 3)):
            prefix = "_".join(tokens[:prefix_length])
            if prefix:
                candidates[prefix].append(column)

    groups: list[tuple[str, str, list[ColumnSchema]]] = []
    used_signatures: set[tuple[str, ...]] = set()
    for prefix in sorted(candidates):
        members = sorted(
            {column.name: column for column in candidates[prefix]}.values(),
            key=lambda column: column.name,
        )
        if len(members) < 3:
            continue
        signature = tuple(column.name for column in members)
        if signature in used_signatures:
            continue
        used_signatures.add(signature)
        slug = _slug(f"prefix_{prefix}")
        groups.append((slug, f"Repeated prefix {prefix}", members))
    return groups


def _is_key_like(column: ColumnSchema) -> bool:
    """Return True when a column structurally looks like an identifier or key."""

    tokens = set(_tokens(column.name))
    normalized_name = _normalize_identifier(column.name)
    if tokens & _KEY_TOKENS:
        return True
    if normalized_name.endswith("_id"):
        return True
    description = column.description or ""
    return bool(re.search(r"\b(unique identifier|primary key|foreign key)\b", description, re.I))


def _is_time_like(column: ColumnSchema) -> bool:
    """Return True when a column structurally looks temporal."""

    if _primitive_type(column.type) == "temporal":
        return True
    tokens = set(_tokens(column.name))
    if tokens & _TIME_TOKENS:
        return True
    description = column.description or ""
    return bool(re.search(r"\b(date|time|timestamp|datetime)\b", description, re.I))


def _is_numeric_measure_candidate(column: ColumnSchema) -> bool:
    """Return True for numeric columns that look like measures."""

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
    """Resolve the full table name used in stable object ids."""

    return table.full_name or table_key or table.name


def _has_categorical_name(column_name: str) -> bool:
    """Return True when the column name has generic categorical tokens."""

    return bool(set(_tokens(column_name)) & _CATEGORICAL_NAME_TOKENS)


def _has_text_like_name(column_name: str) -> bool:
    """Return True for column names that usually hold free-form text or PII."""

    return bool(set(_tokens(column_name)) & _TEXT_LIKE_NAME_TOKENS)


def _looks_numeric(text: str) -> bool:
    """Return True when text is a plain numeric value."""

    try:
        float(text)
    except ValueError:
        return False
    return True


def _normalize_identifier(value: str) -> str:
    """Normalize an identifier for structural comparison."""

    return "_".join(_tokens(value))


def _tokens(value: str) -> list[str]:
    """Split identifier or type text into lowercase alphanumeric tokens."""

    return [match.group(0).lower() for match in _TOKEN_RE.finditer(value)]


def _slug(value: str) -> str:
    """Return a stable object-id-safe slug."""

    slug = re.sub(r"[^a-z0-9_-]+", "_", value.lower()).strip("_")
    return slug or "group"


def _stable_hash(*parts: object) -> str:
    """Return the fixed-width hash used in schema object ids."""

    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def _first_non_empty(values: Iterable[str | None]) -> str | None:
    """Return the first non-empty string in an iterable."""

    for value in values:
        if value:
            return value
    return None


def _join_text(parts: Iterable[object | None]) -> str:
    """Join non-empty pieces into searchable text."""

    return " ".join(str(part).strip() for part in parts if str(part or "").strip())
