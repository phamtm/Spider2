"""Build table-family schema objects."""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from sol01.models import ColumnSchema, SchemaObject, TableSchema
from sol01.schema import objects as common

PrimitiveType = common.PrimitiveType
DEFAULT_MAX_FAMILY_VARIANT_COLUMNS = common.DEFAULT_MAX_FAMILY_VARIANT_COLUMNS
_INTEGER_SUFFIX_RE = common._INTEGER_SUFFIX_RE
_VERSION_SUFFIX_RE = common._VERSION_SUFFIX_RE
_YYYYMMDD_SUFFIX_RE = common._YYYYMMDD_SUFFIX_RE
_YYYY_MM_DD_SUFFIX_RE = common._YYYY_MM_DD_SUFFIX_RE
_YYYY_SUFFIX_RE = common._YYYY_SUFFIX_RE
_join_text = common._join_text
_normalize_identifier = common._normalize_identifier
_primitive_type = common._primitive_type
_slug = common._slug
_stable_hash = common._stable_hash
_table_full_name = common._table_full_name
_tokens = common._tokens


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


def table_family_objects(
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
