"""Load and filter Spider2-snow tasks."""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from collections.abc import Sequence

from sol01.logging import get_logger
from sol01.models import Task

REPO_ROOT = Path(__file__).resolve().parents[3]
SPIDER2_SNOW_PATH = REPO_ROOT / "spider2-snow" / "spider2-snow.jsonl"
ALL_TASK_SELECTOR = "all"
logger = get_logger(__name__)


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
) -> list[Task]:
    """Resolve exact instance IDs or shell-style globs against dataset order."""

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

    for selector in normalized_selectors:
        _validate_selector(selector)

    tasks = _read_tasks(dataset_path)
    selected: list[Task] = []
    seen: set[str] = set()
    for task in tasks:
        if not any(fnmatch.fnmatchcase(task.instance_id, selector) for selector in normalized_selectors):
            continue
        if task.instance_id in seen:
            continue
        seen.add(task.instance_id)
        selected.append(task)

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
    if "/" in selector or ".." in selector:
        raise ValueError(f"path-like selector is not allowed: {selector}")
