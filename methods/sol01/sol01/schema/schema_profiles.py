"""Build, load, and match generated per-database schema-profile artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

from sol01.infra.fs_cache import (
    path_signature,
    read_json,
    safe_path_segment,
    stable_hash,
    write_json,
)
from sol01.infra.paths import REPO_ROOT
from sol01.infra.policy import DEFAULT_SCHEMA_CONTEXT_POLICY
from sol01.models import (
    FamilyProfile,
    SchemaObject,
    SchemaProfileCatalog,
    SchemaProfileManifest,
    TableProfile,
    TableSchema,
)
from sol01.schema._object_families import table_family_objects
from sol01.schema._object_shared import (
    _has_categorical_name,
    _has_text_like_name,
    _is_key_like,
    _is_numeric_measure_candidate,
    _is_time_like,
    _normalize_identifier,
    _primitive_type,
    _table_full_name,
    _tokens,
)
from sol01.schema.index import (
    SNOW_METADATA_ROOT,
    _read_table_metadata,
    _schema_dirs,
    _short_table_name,
    _table_identity,
    build_db_index,
)

DEFAULT_SCHEMA_PROFILE_ROOT = REPO_ROOT / "methods" / "sol01" / "metadata" / "schema_profiles"
SCHEMA_PROFILE_BUILDER_VERSION = "schema-profile-builder-v1"
SCHEMA_PROFILE_SUMMARIZER_VERSION = "schema-profile-deterministic-v1"
SCHEMA_PROFILE_TEMPLATE_VERSION = "schema-profile-template-v1"

_STOP_TOKENS = {
    "all",
    "and",
    "at",
    "by",
    "code",
    "count",
    "data",
    "date",
    "day",
    "for",
    "from",
    "id",
    "ids",
    "in",
    "key",
    "name",
    "of",
    "on",
    "or",
    "table",
    "the",
    "time",
    "to",
    "total",
    "type",
    "value",
    "year",
}


@dataclass(frozen=True)
class SchemaProfileBuildResult:
    """One build result for a per-database schema-profile catalog."""

    db: str
    catalog: SchemaProfileCatalog
    manifest: SchemaProfileManifest
    catalog_path: Path
    manifest_path: Path
    changed: bool


def load_schema_profile_catalog(
    db: str,
    *,
    profile_root: Path = DEFAULT_SCHEMA_PROFILE_ROOT,
) -> SchemaProfileCatalog | None:
    """Load one generated schema-profile catalog, or return None when absent."""

    catalog_path = schema_profile_catalog_path(db, profile_root=profile_root)
    manifest_path = schema_profile_manifest_path(db, profile_root=profile_root)
    catalog_signature = path_signature(catalog_path)
    manifest_signature = path_signature(manifest_path)
    if catalog_signature is None or manifest_signature is None:
        return None
    return _load_schema_profile_catalog(
        str(catalog_path.resolve()),
        catalog_signature,
        str(manifest_path.resolve()),
        manifest_signature,
    )


def load_schema_profile_manifest(
    db: str,
    *,
    profile_root: Path = DEFAULT_SCHEMA_PROFILE_ROOT,
) -> SchemaProfileManifest | None:
    """Load one generated schema-profile manifest, or return None when absent."""

    manifest_path = schema_profile_manifest_path(db, profile_root=profile_root)
    signature = path_signature(manifest_path)
    if signature is None:
        return None
    return _load_schema_profile_manifest(str(manifest_path.resolve()), signature)


def schema_profile_catalog_hash(
    db: str,
    *,
    profile_root: Path = DEFAULT_SCHEMA_PROFILE_ROOT,
) -> str | None:
    """Return the stable artifact hash for one database catalog when it exists."""

    manifest = load_schema_profile_manifest(db, profile_root=profile_root)
    if manifest is None:
        return None
    return manifest.artifact_hash


def schema_profile_catalog_path(
    db: str,
    *,
    profile_root: Path = DEFAULT_SCHEMA_PROFILE_ROOT,
) -> Path:
    return profile_root / safe_path_segment(db) / "catalog.json"


def schema_profile_manifest_path(
    db: str,
    *,
    profile_root: Path = DEFAULT_SCHEMA_PROFILE_ROOT,
) -> Path:
    return profile_root / safe_path_segment(db) / "manifest.json"


def build_schema_profiles(
    dbs: Iterable[str],
    *,
    metadata_root: Path = SNOW_METADATA_ROOT,
    profile_root: Path = DEFAULT_SCHEMA_PROFILE_ROOT,
    family_similarity_threshold: float = DEFAULT_SCHEMA_CONTEXT_POLICY.family_similarity_threshold,
    force: bool = False,
) -> list[SchemaProfileBuildResult]:
    """Build one per-database profile catalog for each requested database."""

    unique_dbs = sorted({db.strip() for db in dbs if db.strip()})
    return [
        build_schema_profile(
            db,
            metadata_root=metadata_root,
            profile_root=profile_root,
            family_similarity_threshold=family_similarity_threshold,
            force=force,
        )
        for db in unique_dbs
    ]


def build_schema_profile(
    db: str,
    *,
    metadata_root: Path = SNOW_METADATA_ROOT,
    profile_root: Path = DEFAULT_SCHEMA_PROFILE_ROOT,
    family_similarity_threshold: float = DEFAULT_SCHEMA_CONTEXT_POLICY.family_similarity_threshold,
    force: bool = False,
) -> SchemaProfileBuildResult:
    """Build or refresh one generated schema-profile catalog from raw metadata only."""

    db_index = build_db_index(db, metadata_root=metadata_root)
    source_schema_hash = stable_hash(
        {
            table_name: db_index[table_name].model_dump(mode="json")
            for table_name in sorted(db_index)
        }
    )
    existing_catalog = load_schema_profile_catalog(db, profile_root=profile_root)
    existing_manifest = load_schema_profile_manifest(db, profile_root=profile_root)
    if (
        not force
        and existing_catalog is not None
        and existing_manifest is not None
        and existing_manifest.source_schema_hash == source_schema_hash
        and existing_manifest.builder_version == SCHEMA_PROFILE_BUILDER_VERSION
        and existing_manifest.summarizer_version == SCHEMA_PROFILE_SUMMARIZER_VERSION
        and existing_manifest.prompt_template_version == SCHEMA_PROFILE_TEMPLATE_VERSION
    ):
        return SchemaProfileBuildResult(
            db=db,
            catalog=existing_catalog,
            manifest=existing_manifest,
            catalog_path=schema_profile_catalog_path(db, profile_root=profile_root),
            manifest_path=schema_profile_manifest_path(db, profile_root=profile_root),
            changed=False,
        )

    metadata_paths = _metadata_paths_for_db(db, metadata_root=metadata_root)
    shared_columns = _shared_column_presence(db_index)
    column_templates = _column_templates_by_table(db_index)
    table_profiles = [
        _table_profile(
            table_key=table_key,
            table=db_index[table_key],
            shared_columns=shared_columns,
            column_templates=column_templates.get(table_key, []),
            metadata_path=metadata_paths.get(table_key),
        )
        for table_key in sorted(db_index)
    ]
    family_profiles = [
        _family_profile(
            schema_object=family_object,
            tables_by_name=db_index,
            shared_columns=shared_columns,
            metadata_paths=metadata_paths,
        )
        for family_object in table_family_objects(
            db_index,
            family_similarity_threshold=family_similarity_threshold,
        )
    ]
    table_profiles = sorted(table_profiles, key=lambda profile: profile.profile_id)
    family_profiles = sorted(family_profiles, key=lambda profile: profile.profile_id)
    catalog = SchemaProfileCatalog(
        db=db,
        source_schema_hash=source_schema_hash,
        table_profiles=table_profiles,
        family_profiles=family_profiles,
        db_overview=_db_overview(db, table_profiles, family_profiles),
        aliases=_catalog_aliases(db, table_profiles, family_profiles),
        theme_terms=_catalog_theme_terms(table_profiles, family_profiles),
    )
    artifact_hash = stable_hash(catalog.model_dump(mode="json"))
    manifest = SchemaProfileManifest(
        db=db,
        source_schema_hash=source_schema_hash,
        builder_version=SCHEMA_PROFILE_BUILDER_VERSION,
        summarizer_version=SCHEMA_PROFILE_SUMMARIZER_VERSION,
        prompt_template_version=SCHEMA_PROFILE_TEMPLATE_VERSION,
        generated_at=datetime.now(UTC).isoformat(),
        artifact_hash=artifact_hash,
        table_profile_count=len(table_profiles),
        family_profile_count=len(family_profiles),
    )
    catalog_path = schema_profile_catalog_path(db, profile_root=profile_root)
    manifest_path = schema_profile_manifest_path(db, profile_root=profile_root)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(catalog_path, catalog.model_dump(mode="json"))
    write_json(manifest_path, manifest.model_dump(mode="json"))
    return SchemaProfileBuildResult(
        db=db,
        catalog=catalog,
        manifest=manifest,
        catalog_path=catalog_path,
        manifest_path=manifest_path,
        changed=True,
    )


def schema_profiles_for_object(
    schema_object: SchemaObject,
    *,
    catalog: SchemaProfileCatalog | None,
) -> list[TableProfile | FamilyProfile]:
    """Return generated profiles that cover one schema object."""

    if catalog is None:
        return []
    if schema_object.object_type == "table":
        table_name = schema_object.table_name or schema_object.name
        matches = [
            profile for profile in catalog.table_profiles if table_name in profile.covered_tables
        ]
        matches.extend(
            profile for profile in catalog.family_profiles if table_name in profile.covered_tables
        )
        return _stable_profile_matches(matches)
    if schema_object.object_type == "family":
        member_refs = schema_object.metadata.get("member_table_refs")
        if not isinstance(member_refs, list):
            return []
        members = {str(item).strip() for item in member_refs if str(item).strip()}
        matches = [
            profile for profile in catalog.family_profiles if set(profile.covered_tables) == members
        ]
        if matches:
            return _stable_profile_matches(matches)
        overlaps = [
            profile
            for profile in catalog.family_profiles
            if members and members.issubset(set(profile.covered_tables))
        ]
        return _stable_profile_matches(overlaps)
    return []


def compact_table_keys_for_profiles(catalog: SchemaProfileCatalog | None) -> set[str]:
    """Return tables whose planner objects should stay compact at cache build time."""

    if catalog is None:
        return set()
    compact_tables = {
        table_name
        for profile in catalog.table_profiles
        if profile.abstraction_kind == "wide_table"
        for table_name in profile.covered_tables
    }
    compact_tables.update(
        table_name for profile in catalog.family_profiles for table_name in profile.covered_tables
    )
    return compact_tables


def profile_by_id(
    catalog: SchemaProfileCatalog | None,
) -> dict[str, TableProfile | FamilyProfile]:
    """Return one dictionary keyed by profile id for convenient lookups."""

    if catalog is None:
        return {}
    return {
        profile.profile_id: profile
        for profile in [*catalog.table_profiles, *catalog.family_profiles]
    }


@cache
def _load_schema_profile_catalog(
    catalog_path: str,
    catalog_signature: tuple[int, int],
    manifest_path: str,
    manifest_signature: tuple[int, int],
) -> SchemaProfileCatalog:
    del manifest_path, manifest_signature
    return SchemaProfileCatalog.model_validate(read_json(Path(catalog_path)))


@cache
def _load_schema_profile_manifest(
    manifest_path: str,
    signature: tuple[int, int],
) -> SchemaProfileManifest:
    del signature
    return SchemaProfileManifest.model_validate(read_json(Path(manifest_path)))


def _shared_column_presence(
    db_index: Mapping[str, TableSchema],
) -> dict[str, list[str]]:
    shared: dict[str, list[str]] = defaultdict(list)
    for table_key in sorted(db_index):
        table = db_index[table_key]
        table_full_name = _table_full_name(table_key, table)
        seen_in_table: set[str] = set()
        for column in table.columns:
            normalized = _normalize_identifier(column.name)
            if normalized and normalized not in seen_in_table:
                shared[normalized].append(table_full_name)
                seen_in_table.add(normalized)
    return {key: value for key, value in shared.items() if len(value) > 1}


def _column_templates_by_table(
    db_index: Mapping[str, TableSchema],
) -> dict[str, list[str]]:
    templates: dict[str, list[str]] = {}
    for table_key in sorted(db_index):
        counts: Counter[str] = Counter()
        for column in db_index[table_key].columns:
            tokens = _tokens(column.name)
            for prefix_length in range(1, min(len(tokens), 3)):
                counts["_".join(tokens[:prefix_length])] += 1
        templates[table_key] = [
            f"{prefix}_* ({count} columns)"
            for prefix, count in sorted(counts.items())
            if prefix and count >= 3
        ][:8]
    return templates


def _table_profile(
    *,
    table_key: str,
    table: TableSchema,
    shared_columns: Mapping[str, Sequence[str]],
    column_templates: Sequence[str],
    metadata_path: Path | None,
) -> TableProfile:
    table_full_name = _table_full_name(table_key, table)
    key_columns, time_columns, measure_columns, dimension_columns = _column_groups(table)
    wide_table = len(table.columns) >= 40 or len(column_templates) >= 3
    aliases = _table_aliases(table)
    theme_terms = _table_theme_terms(table)
    join_anchors = _join_anchor_lines(table, shared_columns)
    naming_rules = _naming_rules(table)
    caveats = _table_caveats(table)
    summary = _table_summary(
        table=table,
        key_columns=key_columns,
        time_columns=time_columns,
        measure_columns=measure_columns,
        dimension_columns=dimension_columns,
        column_templates=column_templates,
    )
    profile_id = _profile_id(table_full_name, suffix="table")
    return TableProfile(
        profile_id=profile_id,
        abstraction_kind="wide_table" if wide_table else "table",
        table_name=table_full_name,
        covered_tables=[table_full_name],
        grain_hint=_grain_hint(table, key_columns=key_columns, time_columns=time_columns),
        key_columns=key_columns,
        time_columns=time_columns,
        measure_columns=measure_columns,
        dimension_columns=dimension_columns,
        repeated_column_templates=list(column_templates),
        join_anchors=join_anchors,
        naming_rules=naming_rules,
        compact_semantic_summary=summary,
        aliases=aliases,
        theme_terms=theme_terms,
        confidence=0.85 if wide_table else 0.75,
        caveats=caveats,
        provenance_inputs=[str(metadata_path)] if metadata_path is not None else [table_full_name],
        source_column_count=len(table.columns),
        source_sample_row_count=len(table.sample_rows),
    )


def _family_profile(
    *,
    schema_object: SchemaObject,
    tables_by_name: Mapping[str, TableSchema],
    shared_columns: Mapping[str, Sequence[str]],
    metadata_paths: Mapping[str, Path],
) -> FamilyProfile:
    covered_tables = [
        str(item).strip()
        for item in schema_object.metadata.get("member_table_refs", [])
        if str(item).strip()
    ]
    canonical_table = str(schema_object.metadata.get("canonical_member") or covered_tables[0])
    canonical_schema = tables_by_name.get(canonical_table) or next(iter(tables_by_name.values()))
    key_columns, time_columns, measure_columns, dimension_columns = _column_groups(canonical_schema)
    family_kind = str(schema_object.metadata.get("family_kind") or "exact")
    abstraction_kind = "exact_family" if family_kind == "exact" else "near_family"
    normalized_stem = str(schema_object.metadata.get("normalized_stem") or canonical_schema.name)
    suffix_dimensions = schema_object.metadata.get("suffix_dimensions")
    naming_rules = _family_naming_rules(canonical_schema, suffix_dimensions)
    join_anchors = _join_anchor_lines(canonical_schema, shared_columns)
    common_columns = [
        str(item).strip()
        for item in schema_object.metadata.get("common_columns", [])
        if str(item).strip()
    ]
    repeated_column_templates = (
        [f"common columns: {', '.join(common_columns[:8])}"] if common_columns else []
    )
    repeated_column_templates.extend(_suffix_dimension_lines(suffix_dimensions))
    repeated_column_templates = repeated_column_templates[:8]
    aliases = _family_aliases(canonical_schema, normalized_stem)
    theme_terms = _table_theme_terms(canonical_schema)
    caveats = [
        str(item).strip() for item in schema_object.metadata.get("caveats", []) if str(item).strip()
    ]
    summary = _family_summary(
        canonical_schema=canonical_schema,
        abstraction_kind=abstraction_kind,
        member_count=len(covered_tables),
        suffix_dimensions=suffix_dimensions,
        common_columns=common_columns,
    )
    provenance_inputs = [
        str(metadata_paths[table_name])
        for table_name in covered_tables
        if table_name in metadata_paths
    ] or covered_tables
    return FamilyProfile(
        profile_id=_profile_id(canonical_table, suffix="family"),
        abstraction_kind=abstraction_kind,
        family_selector=(
            f"stem={normalized_stem};kind={abstraction_kind};canonical={canonical_table}"
        ),
        covered_tables=covered_tables,
        canonical_table=canonical_table,
        grain_hint=_grain_hint(
            canonical_schema,
            key_columns=key_columns,
            time_columns=time_columns,
        ),
        key_columns=key_columns,
        time_columns=time_columns,
        measure_columns=measure_columns,
        dimension_columns=dimension_columns,
        repeated_column_templates=repeated_column_templates,
        join_anchors=join_anchors,
        naming_rules=naming_rules,
        compact_semantic_summary=summary,
        aliases=aliases,
        theme_terms=theme_terms,
        confidence=0.92 if abstraction_kind == "exact_family" else 0.8,
        caveats=caveats,
        provenance_inputs=provenance_inputs,
        member_count=len(covered_tables),
    )


def _column_groups(table: TableSchema) -> tuple[list[str], list[str], list[str], list[str]]:
    key_columns: list[str] = []
    time_columns: list[str] = []
    measure_columns: list[str] = []
    dimension_columns: list[str] = []
    for column in table.columns:
        if _is_key_like(column):
            key_columns.append(column.name)
        if _is_time_like(column):
            time_columns.append(column.name)
        if _is_numeric_measure_candidate(column) and column.name not in measure_columns:
            measure_columns.append(column.name)
        if (
            _has_categorical_name(column.name)
            or _has_text_like_name(column.name)
            or _primitive_type(column.type) == "string"
        ) and column.name not in dimension_columns:
            dimension_columns.append(column.name)
    return key_columns[:8], time_columns[:8], measure_columns[:12], dimension_columns[:12]


def _join_anchor_lines(
    table: TableSchema,
    shared_columns: Mapping[str, Sequence[str]],
) -> list[str]:
    lines: list[str] = []
    for column in table.columns:
        normalized = _normalize_identifier(column.name)
        if not normalized or normalized not in shared_columns:
            continue
        peers = [
            name for name in shared_columns[normalized] if name != (table.full_name or table.name)
        ]
        if not peers:
            continue
        lines.append(f"{column.name} appears across {min(len(peers) + 1, 6)} related tables")
        if len(lines) == 6:
            break
    return lines


def _naming_rules(table: TableSchema) -> list[str]:
    lines = [
        f"Use exact table name {table.full_name or table.name}.",
    ]
    lower_case = [column.name for column in table.columns if column.name != column.name.upper()]
    if lower_case:
        lines.append(
            "Preserve exact snake_case column spellings such as " + ", ".join(lower_case[:6]) + "."
        )
    else:
        lines.append("Preserve exact uppercase spellings for identifiers and codes.")
    return lines[:4]


def _family_naming_rules(table: TableSchema, suffix_dimensions: object) -> list[str]:
    lines = _naming_rules(table)
    for line in _suffix_dimension_lines(suffix_dimensions)[:2]:
        lines.append(f"Physical member naming follows {line}.")
    return lines[:5]


def _table_caveats(table: TableSchema) -> list[str]:
    caveats: list[str] = []
    if any(_primitive_type(column.type) == "semi_structured" for column in table.columns):
        caveats.append("Contains semi-structured columns that often need exact field validation.")
    if len(table.columns) >= 80:
        caveats.append("Wide table: prefer exact column evidence over name guessing.")
    return caveats


def _table_summary(
    *,
    table: TableSchema,
    key_columns: Sequence[str],
    time_columns: Sequence[str],
    measure_columns: Sequence[str],
    dimension_columns: Sequence[str],
    column_templates: Sequence[str],
) -> str:
    themes = ", ".join(_table_theme_terms(table)[:4]) or "mixed schema"
    parts = [
        (
            f"{table.full_name or table.name} is a "
            f"{'wide ' if len(table.columns) >= 40 else ''}table covering {themes}."
        ),
        f"It exposes {len(table.columns)} exact columns.",
    ]
    if key_columns:
        parts.append(f"Typical lookup starts from {', '.join(key_columns[:4])}.")
    if time_columns:
        parts.append(f"Time fields include {', '.join(time_columns[:3])}.")
    if measure_columns:
        parts.append(f"Measure columns include {', '.join(measure_columns[:4])}.")
    if dimension_columns and not key_columns:
        parts.append(f"Dimension-style columns include {', '.join(dimension_columns[:4])}.")
    if column_templates:
        parts.append(f"Repeated column templates include {', '.join(column_templates[:3])}.")
    return " ".join(parts)


def _family_summary(
    *,
    canonical_schema: TableSchema,
    abstraction_kind: str,
    member_count: int,
    suffix_dimensions: object,
    common_columns: Sequence[str],
) -> str:
    suffix_bits = ", ".join(_suffix_dimension_lines(suffix_dimensions)[:2])
    parts = [
        f"{abstraction_kind.replace('_', ' ')} profile spanning {member_count} physical tables.",
        f"Canonical member is {canonical_schema.full_name or canonical_schema.name}.",
    ]
    if suffix_bits:
        parts.append(f"Member names vary by {suffix_bits}.")
    if common_columns:
        parts.append(f"Stable shared columns include {', '.join(common_columns[:6])}.")
    return " ".join(parts)


def _grain_hint(
    table: TableSchema,
    *,
    key_columns: Sequence[str],
    time_columns: Sequence[str],
) -> str:
    parts = ["One row per logical record"]
    if key_columns:
        parts.append(f"keyed by {', '.join(key_columns[:3])}")
    if time_columns:
        parts.append(f"with time anchored by {', '.join(time_columns[:2])}")
    return "; ".join(parts) + "."


def _table_aliases(table: TableSchema) -> list[str]:
    tokens = [token for token in _tokens(table.name) if token and not token.isdigit()]
    aliases = [" ".join(tokens[:3]).strip()] if tokens else []
    if table.schema_name:
        aliases.append(f"{table.schema_name.lower()} {table.name.lower()}")
    return [alias for alias in aliases if alias][:4]


def _family_aliases(table: TableSchema, normalized_stem: str) -> list[str]:
    aliases = [" ".join(_tokens(normalized_stem.replace("_", " ")))[:80].strip()]
    if table.schema_name:
        aliases.append(f"{table.schema_name.lower()} {normalized_stem.replace('_', ' ')}")
    return [alias for alias in aliases if alias][:4]


def _table_theme_terms(table: TableSchema) -> list[str]:
    counter: Counter[str] = Counter()
    for token in _tokens(table.name):
        if token not in _STOP_TOKENS and not token.isdigit():
            counter[token] += 3
    for column in table.columns:
        for token in _tokens(column.name):
            if token not in _STOP_TOKENS and not token.isdigit():
                counter[token] += 1
        if column.description:
            for token in _tokens(column.description):
                if token not in _STOP_TOKENS and not token.isdigit():
                    counter[token] += 1
    return [token for token, _ in counter.most_common(8)]


def _suffix_dimension_lines(raw_dimensions: object) -> list[str]:
    if not isinstance(raw_dimensions, list):
        return []
    lines: list[str] = []
    for raw_dimension in raw_dimensions:
        if not isinstance(raw_dimension, dict):
            continue
        kind = str(raw_dimension.get("kind") or "").strip()
        values = [
            str(item).strip() for item in raw_dimension.get("values", []) if str(item).strip()
        ]
        if kind and values:
            lines.append(f"{kind} suffix values {', '.join(values[:6])}")
    return lines


def _db_overview(
    db: str,
    table_profiles: Sequence[TableProfile],
    family_profiles: Sequence[FamilyProfile],
) -> str:
    wide_count = sum(1 for profile in table_profiles if profile.abstraction_kind == "wide_table")
    themes = ", ".join(_catalog_theme_terms(table_profiles, family_profiles)[:6]) or "mixed themes"
    return (
        f"{db} profile catalog covers {len(table_profiles)} tables and {len(family_profiles)} "
        f"families. Wide-schema abstractions: {wide_count}. Main themes: {themes}."
    )


def _catalog_aliases(
    db: str,
    table_profiles: Sequence[TableProfile],
    family_profiles: Sequence[FamilyProfile],
) -> list[str]:
    aliases = [db.lower()]
    aliases.extend(alias for profile in family_profiles[:4] for alias in profile.aliases[:1])
    aliases.extend(alias for profile in table_profiles[:4] for alias in profile.aliases[:1])
    seen: set[str] = set()
    unique: list[str] = []
    for alias in aliases:
        normalized = alias.strip().casefold()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(alias.strip())
    return unique[:10]


def _catalog_theme_terms(
    table_profiles: Sequence[TableProfile],
    family_profiles: Sequence[FamilyProfile],
) -> list[str]:
    counter: Counter[str] = Counter()
    for profile in [*table_profiles, *family_profiles]:
        counter.update(profile.theme_terms[:6])
    return [token for token, _ in counter.most_common(10)]


def _profile_id(table_name: str, *, suffix: str) -> str:
    return "_".join(_tokens(f"{table_name}_{suffix}"))[:96]


def _metadata_paths_for_db(
    db: str,
    *,
    metadata_root: Path,
) -> dict[str, Path]:
    db_dir = metadata_root / db
    paths: dict[str, Path] = {}
    for schema_dir in _schema_dirs(db_dir):
        schema_name = None if schema_dir == db_dir else schema_dir.name
        for metadata_path in sorted(schema_dir.glob("*.json")):
            metadata = _read_table_metadata(metadata_path)
            table_name = _short_table_name(metadata, metadata_path)
            table_identity = _table_identity(
                metadata,
                db=db,
                schema=schema_name,
                table_name=table_name,
            )
            paths[table_identity] = metadata_path
    return paths


def _stable_profile_matches(
    matches: Sequence[TableProfile | FamilyProfile],
) -> list[TableProfile | FamilyProfile]:
    return sorted(
        {profile.profile_id: profile for profile in matches}.values(),
        key=lambda profile: (
            0 if profile.abstraction_kind in {"exact_family", "wide_table"} else 1,
            profile.profile_id,
        ),
    )
