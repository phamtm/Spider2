"""Build canonical schema objects from table metadata."""

from __future__ import annotations

import hashlib
import itertools
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


def build_schema_objects(
    db_index: Mapping[str, TableSchema],
    *,
    max_sample_values_per_column: int = DEFAULT_MAX_SAMPLE_VALUES_PER_COLUMN,
    max_join_candidates: int = DEFAULT_MAX_JOIN_CANDIDATES,
    max_join_candidates_per_column: int = DEFAULT_MAX_JOIN_CANDIDATES_PER_COLUMN,
    family_similarity_threshold: float = DEFAULT_FAMILY_SIMILARITY_THRESHOLD,
) -> list[SchemaObject]:
    """Build deterministic schema objects from a database table index."""

    column_refs = _column_refs(db_index)
    objects: list[SchemaObject] = []

    for table_key in sorted(db_index):
        table = db_index[table_key]
        table_full_name = _table_full_name(table_key, table)
        objects.append(_table_object(table_key, table, table_full_name))
        objects.extend(_column_object(ref) for ref in column_refs if ref.table_key == table_key)
        objects.extend(_column_group_objects(table_key, table, table_full_name))
        objects.extend(
            _sample_value_objects(
                table_key,
                table,
                table_full_name,
                max_sample_values_per_column=max_sample_values_per_column,
            )
        )

    objects.extend(
        _table_family_objects(
            db_index,
            family_similarity_threshold=family_similarity_threshold,
        )
    )
    objects.extend(
        _join_candidate_objects(
            column_refs,
            max_join_candidates=max_join_candidates,
            max_join_candidates_per_column=max_join_candidates_per_column,
        )
    )
    return objects


def _table_family_objects(
    db_index: Mapping[str, TableSchema],
    *,
    family_similarity_threshold: float,
) -> list[SchemaObject]:
    """Create logical table-family objects from repeated physical tables."""

    members = [
        _table_family_member(table_key, db_index[table_key]) for table_key in sorted(db_index)
    ]
    by_exact_key: dict[
        tuple[str | None, str | None, str, str, tuple[tuple[str, str], ...]],
        list[_TableFamilyMember],
    ] = defaultdict(list)
    by_stem_key: dict[tuple[str | None, str | None, str, str], list[_TableFamilyMember]] = (
        defaultdict(list)
    )

    for member in members:
        dimension_kind = _suffix_dimension_kind(member)
        if dimension_kind is None:
            continue
        scope_key = (
            member.table.database_name,
            member.table.schema_name,
            member.family_stem,
            dimension_kind,
        )
        by_exact_key[(*scope_key, member.signature)].append(member)
        by_stem_key[scope_key].append(member)

    objects: list[SchemaObject] = []
    emitted_member_sets: set[frozenset[str]] = set()

    for exact_members in by_exact_key.values():
        if len(exact_members) < 2:
            continue
        member_set = frozenset(member.table_full_name for member in exact_members)
        emitted_member_sets.add(member_set)
        objects.append(_table_family_object("exact", exact_members, family_similarity_threshold))

    for near_members in by_stem_key.values():
        if len(near_members) < 3:
            continue
        member_set = frozenset(member.table_full_name for member in near_members)
        if member_set in emitted_member_sets:
            continue
        if len({member.signature for member in near_members}) < 2:
            continue
        if not _is_near_family(near_members, family_similarity_threshold):
            continue
        objects.append(_table_family_object("near", near_members, family_similarity_threshold))

    return sorted(objects, key=lambda obj: obj.object_id)


def _table_family_member(table_key: str, table: TableSchema) -> _TableFamilyMember:
    """Return family-detection metadata for one physical table."""

    table_full_name = _table_full_name(table_key, table)
    suffix_dimension = _suffix_dimension(table.name)
    family_stem = (
        str(suffix_dimension["normalized_stem"])
        if suffix_dimension is not None
        else _normalize_identifier(table.name)
    )
    return _TableFamilyMember(
        table_key=table_key,
        table=table,
        table_full_name=table_full_name,
        family_stem=family_stem,
        suffix_dimension=suffix_dimension,
        signature=_column_signature(table.columns),
        column_names=frozenset(_normalize_identifier(column.name) for column in table.columns),
    )


def _table_family_object(
    family_kind: Literal["exact", "near"],
    raw_members: list[_TableFamilyMember],
    family_similarity_threshold: float,
) -> SchemaObject:
    """Build one table-family schema object with deterministic metadata."""

    members = sorted(raw_members, key=_canonical_member_sort_key)
    canonical = members[0]
    scope = _family_scope(canonical.table.database_name, canonical.table.schema_name)
    stem_slug = _slug(f"{family_kind}_{canonical.family_stem}")
    family_hash = _stable_hash(
        family_kind, scope, canonical.family_stem, *sorted(m.table_full_name for m in members)
    )
    member_refs = [member.table_full_name for member in members]
    common_columns = _common_column_names(members)
    variant_columns = _variant_columns(members, common_columns)
    suffix_dimensions = _family_suffix_dimensions(members)
    min_jaccard = _min_column_jaccard(members)
    caveats = []
    if family_kind == "near":
        caveats.append(
            "Near family: structural column-name similarity is not guaranteed logical sameness."
        )
        caveats.append(
            f"Minimum pairwise column-name Jaccard is {min_jaccard:.3f}; "
            f"threshold is {family_similarity_threshold:.3f}."
        )

    searchable_parts = [
        "table family",
        family_kind,
        scope,
        canonical.family_stem,
        canonical.table_full_name,
        " ".join(member_refs),
        " ".join(common_columns),
        " ".join(column["column_name"] for column in variant_columns["columns"]),
        " ".join(str(dimension["kind"]) for dimension in suffix_dimensions),
    ]

    return SchemaObject(
        object_id=f"family:{scope}:{stem_slug}:{family_hash}",
        object_type="family",
        name=f"{scope}.{canonical.family_stem} table family",
        db=canonical.table.database_name,
        table_name=canonical.table_full_name,
        description=f"{family_kind.title()} table family with {len(members)} physical members.",
        searchable_text=_join_text(searchable_parts),
        metadata={
            "source": "load_db_index",
            "schema_object_subtype": "table_family",
            "family_kind": family_kind,
            "database_name": canonical.table.database_name,
            "schema_name": canonical.table.schema_name,
            "normalized_stem": canonical.family_stem,
            "canonical_member": canonical.table_full_name,
            "member_table_refs": member_refs,
            "members": [_family_member_metadata(member) for member in members],
            "common_columns": common_columns,
            "variant_columns": variant_columns,
            "suffix_dimensions": suffix_dimensions,
            "partition_dimensions": suffix_dimensions,
            "member_count": len(members),
            "min_column_name_jaccard": min_jaccard,
            "family_similarity_threshold": family_similarity_threshold,
            "caveats": caveats,
            "provenance": {
                "source": "load_db_index",
                "detection": "same ordered column signature"
                if family_kind == "exact"
                else "same normalized stem and column-name Jaccard threshold",
                "source_table_keys": [member.table_key for member in members],
            },
        },
    )


def _suffix_dimension_kind(member: _TableFamilyMember) -> str | None:
    """Return the recognized suffix dimension kind for one family member."""

    if member.suffix_dimension is None:
        return None
    return str(member.suffix_dimension["kind"])


def _column_signature(columns: Iterable[ColumnSchema]) -> tuple[tuple[str, str], ...]:
    """Return the ordered structural signature used for exact families."""

    return tuple(
        (_normalize_identifier(column.name), " ".join(_tokens(column.type or "")))
        for column in columns
    )


def _is_near_family(
    members: list[_TableFamilyMember],
    family_similarity_threshold: float,
) -> bool:
    """Return True when every member has compatible column-name similarity."""

    return _min_column_jaccard(members) >= family_similarity_threshold


def _min_column_jaccard(members: list[_TableFamilyMember]) -> float:
    """Return the minimum pairwise column-name Jaccard similarity."""

    if len(members) < 2:
        return 1.0
    return min(
        _jaccard(left.column_names, right.column_names)
        for left, right in itertools.combinations(members, 2)
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Return Jaccard similarity for two normalized column-name sets."""

    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _canonical_member_sort_key(member: _TableFamilyMember) -> tuple[int, str, str]:
    """Sort physical members using shortest table name, then lexical order."""

    table_name = member.table.name or member.table_full_name
    return len(table_name), table_name, member.table_full_name


def _family_scope(database_name: str | None, schema_name: str | None) -> str:
    """Return the scope segment used in family object ids."""

    return f"{database_name or 'none'}.{schema_name or 'none'}"


def _common_column_names(members: list[_TableFamilyMember]) -> list[str]:
    """Return common columns using the canonical member's column order and spelling."""

    common_normalized = set.intersection(*(set(member.column_names) for member in members))
    canonical = sorted(members, key=_canonical_member_sort_key)[0]
    return [
        column.name
        for column in canonical.table.columns
        if _normalize_identifier(column.name) in common_normalized
    ]


def _variant_columns(
    members: list[_TableFamilyMember],
    common_columns: list[str],
) -> dict[str, object]:
    """Return capped non-common columns with table membership evidence."""

    common_normalized = {_normalize_identifier(column_name) for column_name in common_columns}
    by_column: dict[str, dict[str, object]] = {}
    for member in members:
        for column in member.table.columns:
            normalized_name = _normalize_identifier(column.name)
            if normalized_name in common_normalized:
                continue
            entry = by_column.setdefault(
                normalized_name,
                {"column_name": column.name, "member_table_refs": []},
            )
            table_refs = entry["member_table_refs"]
            if isinstance(table_refs, list):
                table_refs.append(member.table_full_name)

    columns = [
        {
            "column_name": str(entry["column_name"]),
            "member_table_refs": sorted(str(table_ref) for table_ref in entry["member_table_refs"]),
        }
        for _, entry in sorted(by_column.items())
    ]
    return {
        "columns": columns[:DEFAULT_MAX_FAMILY_VARIANT_COLUMNS],
        "total_count": len(columns),
        "truncated": len(columns) > DEFAULT_MAX_FAMILY_VARIANT_COLUMNS,
    }


def _family_suffix_dimensions(members: list[_TableFamilyMember]) -> list[dict[str, object]]:
    """Aggregate suffix dimensions across table-family members."""

    by_kind: dict[str, list[dict[str, object]]] = defaultdict(list)
    for member in members:
        if member.suffix_dimension is None:
            continue
        by_kind[str(member.suffix_dimension["kind"])].append(member.suffix_dimension)

    dimensions: list[dict[str, object]] = []
    for kind in sorted(by_kind):
        raw_values = sorted({str(dimension["raw_value"]) for dimension in by_kind[kind]})
        normalized_values = sorted({str(dimension["value"]) for dimension in by_kind[kind]})
        dimensions.append(
            {
                "kind": kind,
                "raw_values": raw_values,
                "values": normalized_values,
            }
        )
    return dimensions


def _family_member_metadata(member: _TableFamilyMember) -> dict[str, object]:
    """Return metadata for one physical table in a family."""

    metadata: dict[str, object] = {
        "source_table_key": member.table_key,
        "table_full_name": member.table_full_name,
        "short_name": member.table.name,
    }
    if member.suffix_dimension is not None:
        metadata["suffix_dimension"] = member.suffix_dimension
    return metadata


def _suffix_dimension(table_name: str) -> dict[str, object] | None:
    """Detect one generic partition or suffix dimension at the end of a table name."""

    for detector in (
        _yyyy_mm_dd_suffix_dimension,
        _yyyymmdd_suffix_dimension,
        _version_suffix_dimension,
        _yyyy_suffix_dimension,
        _integer_suffix_dimension,
    ):
        dimension = detector(table_name)
        if dimension is not None:
            return dimension
    return None


def _yyyy_mm_dd_suffix_dimension(table_name: str) -> dict[str, object] | None:
    """Detect a trailing YYYY_MM_DD or YYYY-MM-DD suffix."""

    match = _YYYY_MM_DD_SUFFIX_RE.fullmatch(table_name)
    if not match or not _valid_month_day(match.group("month"), match.group("day")):
        return None
    value = f"{match.group('year')}-{match.group('month')}-{match.group('day')}"
    raw_value = table_name[len(match.group("stem")) + 1 :]
    return _suffix_dimension_payload(
        table_name,
        match.group("stem"),
        "YYYY_MM_DD",
        raw_value,
        value,
    )


def _yyyymmdd_suffix_dimension(table_name: str) -> dict[str, object] | None:
    """Detect a trailing YYYYMMDD suffix."""

    match = _YYYYMMDD_SUFFIX_RE.fullmatch(table_name)
    if not match or not _valid_month_day(match.group("month"), match.group("day")):
        return None
    value = f"{match.group('year')}-{match.group('month')}-{match.group('day')}"
    raw_value = f"{match.group('year')}{match.group('month')}{match.group('day')}"
    return _suffix_dimension_payload(table_name, match.group("stem"), "YYYYMMDD", raw_value, value)


def _version_suffix_dimension(table_name: str) -> dict[str, object] | None:
    """Detect trailing version suffixes such as v2, _v2, _ver2, or _version2."""

    match = _VERSION_SUFFIX_RE.fullmatch(table_name)
    if not match:
        return None
    raw_value = table_name[len(match.group("stem")) :]
    return _suffix_dimension_payload(
        table_name,
        match.group("stem"),
        "version",
        raw_value,
        int(match.group("value")),
    )


def _yyyy_suffix_dimension(table_name: str) -> dict[str, object] | None:
    """Detect a trailing four-digit year suffix."""

    match = _YYYY_SUFFIX_RE.fullmatch(table_name)
    if not match:
        return None
    year = int(match.group("value"))
    if not 1900 <= year <= 2099:
        return None
    return _suffix_dimension_payload(
        table_name,
        match.group("stem"),
        "YYYY",
        match.group("value"),
        year,
    )


def _integer_suffix_dimension(table_name: str) -> dict[str, object] | None:
    """Detect a trailing integer suffix after date and year checks fail."""

    match = _INTEGER_SUFFIX_RE.fullmatch(table_name)
    if not match:
        return None
    return _suffix_dimension_payload(
        table_name,
        match.group("stem"),
        "integer",
        match.group("value"),
        int(match.group("value")),
    )


def _suffix_dimension_payload(
    table_name: str,
    raw_stem: str,
    kind: str,
    raw_value: object,
    value: object,
) -> dict[str, object] | None:
    """Build one suffix dimension payload when the stem remains meaningful."""

    normalized_stem = _normalize_identifier(raw_stem)
    if not normalized_stem:
        return None
    return {
        "kind": kind,
        "raw_stem": raw_stem,
        "normalized_stem": normalized_stem,
        "raw_value": raw_value,
        "value": value,
        "source_table_name": table_name,
    }


def _valid_month_day(month_text: str, day_text: str) -> bool:
    """Return True for a plausible month/day pair without dataset semantics."""

    month = int(month_text)
    day = int(day_text)
    return 1 <= month <= 12 and 1 <= day <= 31


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


def _join_candidate_objects(
    column_refs: list[_ColumnRef],
    *,
    max_join_candidates: int,
    max_join_candidates_per_column: int,
) -> list[SchemaObject]:
    """Infer same-name compatible-type join evidence across tables."""

    grouped_refs: dict[
        tuple[str | None, str | None, str, PrimitiveType],
        list[_ColumnRef],
    ] = defaultdict(list)
    for ref in column_refs:
        primitive_type = _primitive_type(ref.column.type)
        if primitive_type in {"semi_structured", "unknown"}:
            continue
        key = (
            ref.database_name,
            ref.schema_name,
            _normalize_identifier(ref.column.name),
            primitive_type,
        )
        grouped_refs[key].append(ref)

    objects: list[SchemaObject] = []
    for (database_name, schema_name, normalized_name, primitive_type), refs in sorted(
        grouped_refs.items()
    ):
        unique_refs = sorted(
            refs,
            key=lambda ref: (ref.table_full_name, ref.column.name),
        )
        if len({ref.table_full_name for ref in unique_refs}) < 2:
            continue

        emitted_for_group = 0
        for left, right in itertools.combinations(unique_refs, 2):
            if left.table_full_name == right.table_full_name:
                continue
            object_id = _join_candidate_id(left, right)
            objects.append(
                SchemaObject(
                    object_id=object_id,
                    object_type="join_candidate",
                    name=(
                        f"{left.table_full_name}.{left.column.name} -> "
                        f"{right.table_full_name}.{right.column.name}"
                    ),
                    db=database_name,
                    searchable_text=_join_text(
                        [
                            "join candidate",
                            normalized_name,
                            left.table_full_name,
                            left.column.name,
                            right.table_full_name,
                            right.column.name,
                        ]
                    ),
                    metadata={
                        "source": "load_db_index",
                        "evidence_type": "same_normalized_column_name_and_compatible_type",
                        "is_inferred_evidence": True,
                        "is_foreign_key": False,
                        "database_name": database_name,
                        "schema_name": schema_name,
                        "normalized_column_name": normalized_name,
                        "primitive_type": primitive_type,
                        "left": _join_side_metadata(left),
                        "right": _join_side_metadata(right),
                        "provenance": {"source": "load_db_index"},
                    },
                )
            )
            emitted_for_group += 1
            if emitted_for_group >= max_join_candidates_per_column:
                break
            if len(objects) >= max_join_candidates:
                return objects
    return objects


def _sample_value_objects(
    table_key: str,
    table: TableSchema,
    table_full_name: str,
    *,
    max_sample_values_per_column: int,
) -> list[SchemaObject]:
    """Create sample-value objects only for bounded categorical evidence."""

    objects: list[SchemaObject] = []
    for column in table.columns:
        decision = _sample_value_decision(column, table.sample_rows)
        if not decision.include:
            continue

        for value in decision.values[:max_sample_values_per_column]:
            value_text = str(value)
            value_hash = _stable_hash(table_full_name, column.name, value_text)
            objects.append(
                SchemaObject(
                    object_id=f"sample_value:{table_full_name}#{column.name}:{value_hash}",
                    object_type="sample_value",
                    name=value_text,
                    db=table.database_name,
                    table_name=table_full_name,
                    column_name=column.name,
                    searchable_text=_join_text([value_text, column.name, table_full_name]),
                    metadata={
                        "source": "load_db_index",
                        "source_table_key": table_key,
                        "database_name": table.database_name,
                        "schema_name": table.schema_name,
                        "table_full_name": table_full_name,
                        "column_name": column.name,
                        "column_type": column.type,
                        "value": value,
                        "sample_size": decision.sample_size,
                        "distinct_count": decision.distinct_count,
                        "inclusion_reason": decision.reason,
                        "provenance": {
                            "source": "load_db_index",
                            "table_key": table_key,
                            "sample_size": decision.sample_size,
                            "distinct_count": decision.distinct_count,
                        },
                    },
                )
            )
    return objects


@dataclass(frozen=True)
class _SampleValueDecision:
    """Decision details for one column's sample-value indexing policy."""

    include: bool
    values: list[object]
    sample_size: int
    distinct_count: int
    reason: str


def _sample_value_decision(
    column: ColumnSchema,
    sample_rows: list[dict[str, object]],
) -> _SampleValueDecision:
    """Return the conservative sample-value indexing decision for one column."""

    raw_values = [
        row[column.name]
        for row in sample_rows
        if column.name in row and row[column.name] not in (None, "")
    ]
    distinct_values = _stable_distinct(raw_values)
    sample_size = len(raw_values)
    distinct_count = len(distinct_values)
    primitive_type = _primitive_type(column.type)

    if sample_size < 2 or not distinct_values:
        return _exclude_sample_values(sample_size, distinct_count, "insufficient_sample")
    if _is_key_like(column):
        return _exclude_sample_values(sample_size, distinct_count, "key_like_column")
    if _is_time_like(column):
        return _exclude_sample_values(sample_size, distinct_count, "time_like_column")
    if primitive_type in {"numeric", "temporal", "semi_structured"}:
        return _exclude_sample_values(sample_size, distinct_count, f"{primitive_type}_column")
    if _has_text_like_name(column.name):
        return _exclude_sample_values(sample_size, distinct_count, "raw_text_like_column")
    if any(not _is_indexable_sample_value(value) for value in distinct_values):
        return _exclude_sample_values(sample_size, distinct_count, "unsafe_value_shape")
    if distinct_count > DEFAULT_MAX_SAMPLE_VALUES_PER_COLUMN:
        return _exclude_sample_values(sample_size, distinct_count, "too_many_distinct_values")
    if distinct_count == sample_size and not _has_categorical_name(column.name):
        return _exclude_sample_values(sample_size, distinct_count, "unknown_cardinality")

    reason = (
        "repeated_bounded_sample" if distinct_count < sample_size else "categorical_column_name"
    )
    return _SampleValueDecision(
        include=True,
        values=distinct_values,
        sample_size=sample_size,
        distinct_count=distinct_count,
        reason=reason,
    )


def _exclude_sample_values(
    sample_size: int,
    distinct_count: int,
    reason: str,
) -> _SampleValueDecision:
    """Return a negative sample-value decision with diagnostic counts."""

    return _SampleValueDecision(
        include=False,
        values=[],
        sample_size=sample_size,
        distinct_count=distinct_count,
        reason=reason,
    )


def _join_side_metadata(ref: _ColumnRef) -> dict[str, object]:
    """Return compact metadata for one side of an inferred join candidate."""

    return {
        "source_table_key": ref.table_key,
        "table_full_name": ref.table_full_name,
        "column_name": ref.column.name,
        "column_type": ref.column.type,
        "primitive_type": _primitive_type(ref.column.type),
    }


def _join_candidate_id(left: _ColumnRef, right: _ColumnRef) -> str:
    """Create a stable join candidate id from the ordered pair."""

    join_hash = _stable_hash(
        left.table_full_name,
        left.column.name,
        right.table_full_name,
        right.column.name,
        "join_candidate",
    )
    return (
        f"join_candidate:{left.table_full_name}#{left.column.name}->"
        f"{right.table_full_name}#{right.column.name}:{join_hash}"
    )


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
