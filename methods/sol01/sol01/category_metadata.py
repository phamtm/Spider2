"""Load and validate Spider2-Snow category batch metadata."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from sol01.logging import get_logger
from sol01.models import CategoryMetadata

REPO_ROOT = Path(__file__).resolve().parents[3]
SPIDER2_SNOW_PATH = REPO_ROOT / "spider2-snow" / "spider2-snow.jsonl"
CATEGORY_BATCHES_DIR = REPO_ROOT / "methods" / "sol01" / "metadata" / "category_batches"
logger = get_logger(__name__)

_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
KNOWN_CATEGORY_TAGS = frozenset(
    {
        "activity_span",
        "address_profile",
        "ads",
        "ads_targeting",
        "age_group",
        "aggregation",
        "annotations",
        "annual",
        "anova",
        "anti_join",
        "area_weighting",
        "array_split",
        "array_unnest",
        "balance",
        "basket_analysis",
        "binary_filter",
        "bitcoin",
        "blockchain",
        "categorization",
        "chi_square",
        "citation",
        "claims",
        "classification",
        "clinical",
        "coalesce",
        "code_frequency",
        "cohort",
        "comparison",
        "concept_hierarchy",
        "conversion_rate",
        "coordinate",
        "copy_number",
        "correlation",
        "count",
        "country",
        "cpc",
        "cross_join",
        "crosswalk",
        "cumulative",
        "dataset_build",
        "date_diff",
        "date_filter",
        "date_range",
        "dedupe",
        "dependency",
        "depth_filter",
        "device_filter",
        "distance",
        "distance_bucket",
        "distinct",
        "distinct_count",
        "distinct_lists",
        "document_lookup",
        "double_entry",
        "drug_metadata",
        "embedding",
        "event_classification",
        "event_log",
        "event_parsing",
        "event_sequence",
        "external_knowledge",
        "family_aggregation",
        "fare_metrics",
        "feature_engineering",
        "fees",
        "file_extension",
        "file_filter",
        "filter",
        "forecasting",
        "formatting",
        "formula",
        "forum",
        "frequency",
        "full_join",
        "gap_analysis",
        "gene_expression",
        "genomics",
        "genotype",
        "geo",
        "geometry",
        "geospatial",
        "github_events",
        "graph",
        "group_by",
        "group_detection",
        "hardy_weinberg",
        "historical",
        "imaging",
        "intersection",
        "interval_overlap",
        "join",
        "json",
        "kruskal_wallis",
        "lag",
        "language_filter",
        "language_list",
        "language_mix",
        "length",
        "license_filter",
        "line_item_match",
        "lineage",
        "max",
        "median",
        "metadata",
        "moving_average",
        "multi_country",
        "multi_dataset",
        "multi_join",
        "multi_metric",
        "multi_scale",
        "multi_step",
        "multi_year",
        "mutation",
        "nearest_neighbor",
        "nested_aggregation",
        "nested_ranking",
        "normalization",
        "ordering",
        "pairwise",
        "path",
        "path_sequence",
        "pathway_enrichment",
        "percent_change",
        "percentage",
        "pivot",
        "product_analysis",
        "proportion",
        "publication_metadata",
        "quantile",
        "quarter",
        "radius",
        "range",
        "ranking",
        "ratio",
        "recursive",
        "regex",
        "regression",
        "repository_filter",
        "retention",
        "sample_type",
        "segmentation",
        "self_join",
        "session_aggregation",
        "session_boundary",
        "session_window",
        "set_difference",
        "share",
        "similarity",
        "sort",
        "sorting",
        "spatial",
        "spatial_adjustment",
        "spatial_join",
        "state_filter",
        "string_agg",
        "string_cleaning",
        "string_match",
        "string_normalization",
        "subquery",
        "t_statistic",
        "t_test",
        "temporal",
        "text_extract",
        "text_join",
        "text_match",
        "text_parse",
        "text_processing",
        "text_search",
        "threshold",
        "tie_break",
        "time_bucket",
        "time_series",
        "timezone",
        "token_supply",
        "token_transfer",
        "top_k",
        "train_test_split",
        "transformation",
        "union",
        "url_cleaning",
        "url_lookup",
        "user_average",
        "user_segment",
        "uuid",
        "validation",
        "variance",
        "variant_analysis",
        "weighted_average",
        "window",
        "z_score",
    }
)


class CategoryMetadataValidationError(ValueError):
    """Raised when one or more category metadata rows fail validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


def load_category_metadata(
    *,
    dataset_path: Path = SPIDER2_SNOW_PATH,
    batch_dir: Path = CATEGORY_BATCHES_DIR,
    allowed_tags: Iterable[str] = KNOWN_CATEGORY_TAGS,
) -> list[CategoryMetadata]:
    """Load, validate, and order Spider2-Snow category batch rows."""

    dataset_order = _dataset_order(dataset_path)
    allowed_lookup = set(allowed_tags)
    records, errors = _read_batch_records(
        batch_dir,
        dataset_order=dataset_order,
        allowed_tags=allowed_lookup,
    )
    if errors:
        raise CategoryMetadataValidationError(errors)

    ordered_records = sorted(records, key=lambda record: dataset_order[record.instance_id])
    logger.info(
        "category metadata loaded",
        batch_dir=str(batch_dir),
        dataset_path=str(dataset_path),
        record_count=len(ordered_records),
    )
    return ordered_records


def load_category_metadata_map(
    *,
    dataset_path: Path = SPIDER2_SNOW_PATH,
    batch_dir: Path = CATEGORY_BATCHES_DIR,
    allowed_tags: Iterable[str] = KNOWN_CATEGORY_TAGS,
) -> dict[str, CategoryMetadata]:
    """Load category metadata into a lookup keyed by instance_id."""

    return {
        record.instance_id: record
        for record in load_category_metadata(
            dataset_path=dataset_path,
            batch_dir=batch_dir,
            allowed_tags=allowed_tags,
        )
    }


def _dataset_order(dataset_path: Path) -> dict[str, int]:
    """Map each Spider2-Snow instance id to its dataset order."""

    order: dict[str, int] = {}
    with dataset_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            instance_id = str(record["instance_id"])
            order[instance_id] = index
    return order


def _read_batch_records(
    batch_dir: Path,
    *,
    dataset_order: dict[str, int],
    allowed_tags: set[str],
) -> tuple[list[CategoryMetadata], list[str]]:
    """Read every batch file and collect validation errors in one pass."""

    if not batch_dir.exists():
        return [], [f"Category batch directory does not exist: {batch_dir}"]

    batch_paths = sorted(batch_dir.glob("batch_*.jsonl"))
    if not batch_paths:
        return [], [f"No category batch files found in: {batch_dir}"]

    records: list[CategoryMetadata] = []
    errors: list[str] = []
    seen_instance_ids: set[str] = set()

    for batch_path in batch_paths:
        for line_number, line in enumerate(batch_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{batch_path.name}:{line_number}: invalid JSON: {exc.msg}")
                continue

            if not isinstance(payload, dict):
                errors.append(f"{batch_path.name}:{line_number}: row must be a JSON object")
                continue

            instance_id = payload.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id.strip():
                errors.append(
                    f"{batch_path.name}:{line_number}: instance_id must be a non-empty string"
                )
                continue

            if instance_id not in dataset_order:
                errors.append(
                    f"{batch_path.name}:{line_number}: unknown instance_id {instance_id}"
                )
                continue

            if instance_id in seen_instance_ids:
                errors.append(
                    f"{batch_path.name}:{line_number}: duplicate metadata row for {instance_id}"
                )
                continue

            primary_tier = payload.get("primary_tier")
            if (
                not isinstance(primary_tier, int)
                or isinstance(primary_tier, bool)
                or not 1 <= primary_tier <= 12
            ):
                errors.append(
                    f"{batch_path.name}:{line_number}: invalid primary_tier for {instance_id}"
                )
                continue

            tags = payload.get("tags")
            if not isinstance(tags, list) or not tags:
                errors.append(f"{batch_path.name}:{line_number}: tags must be a non-empty list")
                continue

            if len({tag for tag in tags if isinstance(tag, str)}) != len(tags):
                errors.append(f"{batch_path.name}:{line_number}: tags must be unique")
                continue

            bad_tag = next(
                (
                    tag
                    for tag in tags
                    if not isinstance(tag, str)
                    or not _SNAKE_CASE_RE.fullmatch(tag)
                    or tag not in allowed_tags
                ),
                None,
            )
            if bad_tag is not None:
                if not isinstance(bad_tag, str) or not _SNAKE_CASE_RE.fullmatch(bad_tag):
                    errors.append(
                        f"{batch_path.name}:{line_number}: invalid tag format for {instance_id}"
                    )
                else:
                    errors.append(
                        f"{batch_path.name}:{line_number}: unknown tag {bad_tag} for {instance_id}"
                    )
                continue

            difficulty_notes = payload.get("difficulty_notes")
            if difficulty_notes is not None and not isinstance(difficulty_notes, str):
                errors.append(
                    f"{batch_path.name}:{line_number}: difficulty_notes must be a string or null"
                )
                continue

            seen_instance_ids.add(instance_id)
            records.append(
                CategoryMetadata(
                    instance_id=instance_id,
                    primary_tier=primary_tier,
                    tags=list(tags),
                    difficulty_notes=difficulty_notes,
                )
            )

    return records, errors
