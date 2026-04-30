"""Load and filter Spider2-snow tasks."""

from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Sequence
from pathlib import Path

from sol01.category_metadata import (  # noqa: F401
    CATEGORY_BATCHES_DIR,
    KNOWN_CATEGORY_TAGS,
    CategoryMetadata,
    CategoryMetadataValidationError,
    load_category_metadata,
    load_category_metadata_map,
)
from sol01.logging import get_logger
from sol01.models import Task

REPO_ROOT = Path(__file__).resolve().parents[3]
SPIDER2_SNOW_PATH = REPO_ROOT / "spider2-snow" / "spider2-snow.jsonl"
ALL_TASK_SELECTOR = "all"
logger = get_logger(__name__)
_TIER_SELECTOR_RE = re.compile(r"^\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$")
_TAG_SELECTOR_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


def load_tasks(
    *,
    dataset_path: Path = SPIDER2_SNOW_PATH,
    instance_id: str | None = None,
    db: str | None = None,
    question_contains: str | None = None,
    limit: int | None = None,
) -> list[Task]:
    """Return Spider2-snow tasks after applying the requested filters."""

    tasks = _read_tasks(dataset_path)

    if instance_id is not None:
        tasks = [task for task in tasks if task.instance_id == instance_id]

    if db is not None:
        tasks = [task for task in tasks if task.db == db]

    if question_contains:
        needle = question_contains.casefold()
        tasks = [task for task in tasks if needle in task.question.casefold()]

    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        tasks = tasks[:limit]

    logger.info(
        "tasks loaded",
        dataset_path=str(dataset_path),
        task_count=len(tasks),
        instance_id=instance_id,
        db=db,
        question_contains=question_contains,
        limit=limit,
    )
    return tasks


def select_tasks(
    selectors: Sequence[str] | str,
    *,
    dataset_path: Path = SPIDER2_SNOW_PATH,
    batch_dir: Path = CATEGORY_BATCHES_DIR,
) -> list[Task]:
    """Resolve exact IDs, globs, and category filters against dataset order."""

    if isinstance(selectors, str):
        selectors = [selectors]
    normalized_selectors = [selector.strip() for selector in selectors]
    if not normalized_selectors:
        raise ValueError("selectors must not be empty")
    if ALL_TASK_SELECTOR in normalized_selectors:
        if len(normalized_selectors) != 1:
            raise ValueError("all selector cannot be combined with other selectors")
        tasks = _read_tasks(dataset_path)
        logger.info(
            "tasks selected",
            dataset_path=str(dataset_path),
            selector=ALL_TASK_SELECTOR,
            task_count=len(tasks),
        )
        return tasks

    task_selectors, primary_tiers, tags = _split_selector_filters(normalized_selectors)
    for selector in task_selectors:
        _validate_selector(selector)

    tasks = _read_tasks(dataset_path)
    metadata_map = None
    if primary_tiers or tags:
        metadata_map = load_category_metadata_map(dataset_path=dataset_path, batch_dir=batch_dir)
    missing_metadata_ids: list[str] = []
    selected: list[Task] = []
    seen: set[str] = set()
    for task in tasks:
        if task_selectors and not any(
            fnmatch.fnmatchcase(task.instance_id, selector) for selector in task_selectors
        ):
            continue
        if metadata_map is not None:
            metadata = metadata_map.get(task.instance_id)
            if metadata is None:
                missing_metadata_ids.append(task.instance_id)
                continue
            if primary_tiers and metadata.primary_tier not in primary_tiers:
                continue
            if tags and any(tag not in metadata.tags for tag in tags):
                continue
        if task.instance_id in seen:
            continue
        seen.add(task.instance_id)
        selected.append(task)

    if missing_metadata_ids:
        logger.warning(
            "skipping tasks without category metadata",
            missing_count=len(missing_metadata_ids),
            sample_instance_ids=missing_metadata_ids[:5],
        )

    if not selected:
        raise ValueError(f"No tasks matched selectors: {', '.join(normalized_selectors)}")

    logger.info(
        "tasks selected",
        dataset_path=str(dataset_path),
        selectors=normalized_selectors,
        task_count=len(selected),
    )
    return selected


def _read_tasks(dataset_path: Path) -> list[Task]:
    """Read the JSONL dataset into typed task objects."""

    with dataset_path.open(encoding="utf-8") as handle:
        return [_task_from_record(json.loads(line)) for line in handle if line.strip()]


def _task_from_record(record: dict[str, object]) -> Task:
    """Adapt a Spider2-snow JSONL record to the compact task contract."""

    external_knowledge = record.get("external_knowledge")
    return Task(
        instance_id=str(record["instance_id"]),
        db=str(record["db_id"]),
        question=str(record["instruction"]),
        external_knowledge=str(external_knowledge) if external_knowledge is not None else None,
    )


def _validate_selector(selector: str) -> None:
    """Reject selector forms we do not want to support in run mode."""

    if not selector:
        raise ValueError("selector must not be empty")
    if selector == "*":
        raise ValueError("bare '*' is not allowed; use the all selector instead")
    if selector.startswith(("tier:", "tier=", "tag:", "tag=")):
        raise ValueError(f"category selector must be handled separately: {selector}")
    if "/" in selector or ".." in selector:
        raise ValueError(f"path-like selector is not allowed: {selector}")


def _split_selector_filters(selectors: Sequence[str]) -> tuple[list[str], set[int], list[str]]:
    """Split task selectors from tier and tag filters."""

    task_selectors: list[str] = []
    primary_tiers: list[int] = []
    tags: list[str] = []
    for selector in selectors:
        if selector.startswith(("tier:", "tier=")):
            primary_tiers.extend(_parse_tier_selector(selector))
            continue
        if selector.startswith(("tag:", "tag=")):
            tags.extend(_parse_tag_selector(selector))
            continue
        task_selectors.append(selector)
    return task_selectors, set(primary_tiers), list(dict.fromkeys(tags))


def _parse_tier_selector(selector: str) -> list[int]:
    """Return all tiers described by one selector token."""

    _, value = selector.split(":", 1) if ":" in selector else selector.split("=", 1)
    value = value.strip()
    if not value:
        raise ValueError(f"tier selector must not be empty: {selector}")

    tiers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"tier selector must not contain empty entries: {selector}")
        if part.count("-") > 1:
            raise ValueError(f"invalid tier range: {part}")
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError as exc:
                raise ValueError(f"invalid tier range: {part}") from exc
            if start > end:
                raise ValueError(f"tier range must be ascending: {part}")
            tiers.extend(range(start, end + 1))
            continue
        try:
            tiers.append(int(part))
        except ValueError as exc:
            raise ValueError(f"invalid tier value: {part}") from exc

    for tier in tiers:
        if tier < 1 or tier > 12:
            raise ValueError(f"tier must be between 1 and 12: {tier}")
    return tiers


def _parse_tag_selector(selector: str) -> list[str]:
    """Return all tags described by one selector token."""

    _, value = selector.split(":", 1) if ":" in selector else selector.split("=", 1)
    value = value.strip()
    if not value:
        raise ValueError(f"tag selector must not be empty: {selector}")

    tags: list[str] = []
    for part in value.split(","):
        tag = part.strip()
        if not tag:
            raise ValueError(f"tag selector must not contain empty entries: {selector}")
        if not _TAG_SELECTOR_RE.fullmatch(tag):
            raise ValueError(f"invalid tag selector: {tag}")
        if tag not in KNOWN_CATEGORY_TAGS:
            raise ValueError(f"unknown tag selector: {tag}")
        tags.append(tag)
    return tags
