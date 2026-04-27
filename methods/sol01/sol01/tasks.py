"""Load and filter Spider2-Lite tasks for the local SQLite subset."""

from __future__ import annotations

import json
from pathlib import Path

from sol01.logging import get_logger
from sol01.models import Task

REPO_ROOT = Path(__file__).resolve().parents[3]
SPIDER2_LITE_PATH = REPO_ROOT / "spider2-lite" / "spider2-lite.jsonl"
logger = get_logger(__name__)


def load_tasks(
    *,
    dataset_path: Path = SPIDER2_LITE_PATH,
    instance_id: str | None = None,
    db: str | None = None,
    question_contains: str | None = None,
    limit: int | None = None,
) -> list[Task]:
    """Return local Spider2-Lite tasks after applying the requested filters."""

    tasks = [task for task in _read_tasks(dataset_path) if task.instance_id.startswith("local")]

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


def _read_tasks(dataset_path: Path) -> list[Task]:
    """Read the JSONL dataset into typed task objects."""

    with dataset_path.open(encoding="utf-8") as handle:
        return [Task.model_validate(json.loads(line)) for line in handle if line.strip()]
