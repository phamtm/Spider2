import json
from pathlib import Path

import pytest

from sol01.loading.tasks import ALL_TASK_SELECTOR, SPIDER2_SNOW_PATH, load_tasks, select_tasks


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _setup_category_dataset(
    tmp_path: Path,
    instance_ids: list[str],
    batch_rows: list[dict[str, object]],
) -> tuple[Path, Path]:
    dataset = tmp_path / "spider2-snow.jsonl"
    _write_jsonl(
        dataset,
        [
            {"instance_id": iid, "instruction": "q", "db_id": "DB", "external_knowledge": None}
            for iid in instance_ids
        ],
    )
    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    _write_jsonl(batch_dir / "batch_01.jsonl", batch_rows)
    return dataset, batch_dir


def test_default_dataset_path_exists():
    assert SPIDER2_SNOW_PATH.exists()


def test_snow_task_count_is_547():
    tasks = load_tasks()

    assert len(tasks) == 547
    assert all(task.instance_id.startswith("sf") for task in tasks)


def test_can_select_sf_local003_by_instance_id():
    tasks = select_tasks(["sf_local003"])

    assert len(tasks) == 1
    assert tasks[0].instance_id == "sf_local003"
    assert tasks[0].db == "E_COMMERCE"
    assert tasks[0].question.startswith("According to the RFM definition document")
    assert tasks[0].external_knowledge == "RFM.md"


def test_exact_task_selector_matches_one_task():
    tasks = select_tasks(["sf_local003"])

    assert [task.instance_id for task in tasks] == ["sf_local003"]


def test_multiple_glob_selectors_dedupe_and_preserve_dataset_order():
    tasks = select_tasks(["sf_local003", "sf_local00[34]"])

    assert [task.instance_id for task in tasks] == [
        "sf_local003",
        "sf_local004",
    ]


def test_all_selector_returns_full_dataset():
    tasks = select_tasks([ALL_TASK_SELECTOR])

    assert len(tasks) == 547
    assert tasks[0].instance_id.startswith("sf")


def test_db_question_and_limit_filters_can_be_combined():
    tasks = select_tasks(
        None,
        db="E_COMMERCE",
        question_contains="average payment per order",
        limit=1,
    )

    assert len(tasks) == 1
    assert tasks[0].instance_id == "sf_local004"


def test_question_filter_is_case_insensitive():
    tasks = select_tasks(None, question_contains="AVERAGE SINGLE CAREER SPAN")

    assert any(task.instance_id == "sf_local007" for task in tasks)


def test_negative_limit_is_rejected():
    with pytest.raises(ValueError, match="limit"):
        select_tasks(None, limit=-1)


@pytest.mark.parametrize("selectors", [["*"], ["sf_local003/extra"], ["sf_local003..bad"]])
def test_invalid_selectors_are_rejected(selectors):
    with pytest.raises(ValueError, match="not allowed"):
        select_tasks(selectors)


def test_zero_match_selectors_are_rejected():
    with pytest.raises(ValueError, match="No tasks matched"):
        select_tasks(["sf_does_not_exist*"])


FOUR_TASK_IDS = ["sf_a", "sf_b", "sf_c", "sf_d"]
FOUR_TASK_BATCH = [
    {"instance_id": "sf_a", "primary_tier": 1, "tags": ["aggregation"]},
    {"instance_id": "sf_b", "primary_tier": 3, "tags": ["aggregation", "temporal"]},
    {"instance_id": "sf_c", "primary_tier": 5, "tags": ["comparison"]},
    {"instance_id": "sf_d", "primary_tier": 4, "tags": ["aggregation", "comparison"]},
]


def test_category_selectors_can_filter_by_tier_and_tags(tmp_path: Path):
    dataset, batch_dir = _setup_category_dataset(tmp_path, FOUR_TASK_IDS, FOUR_TASK_BATCH)

    assert [
        task.instance_id
        for task in select_tasks(["tier:3"], dataset_path=dataset, batch_dir=batch_dir)
    ] == ["sf_b"]
    assert [
        task.instance_id
        for task in select_tasks(["tier:3-4"], dataset_path=dataset, batch_dir=batch_dir)
    ] == ["sf_b", "sf_d"]
    assert [
        task.instance_id
        for task in select_tasks(["tag:aggregation"], dataset_path=dataset, batch_dir=batch_dir)
    ] == ["sf_a", "sf_b", "sf_d"]
    assert [
        task.instance_id
        for task in select_tasks(
            ["tag:aggregation", "tag:temporal"],
            dataset_path=dataset,
            batch_dir=batch_dir,
        )
    ] == ["sf_b"]
    assert [
        task.instance_id
        for task in select_tasks(
            ["sf_*", "tier:4", "tag:comparison"],
            dataset_path=dataset,
            batch_dir=batch_dir,
        )
    ] == ["sf_d"]


def test_category_selectors_skip_tasks_missing_metadata(tmp_path: Path):
    dataset, batch_dir = _setup_category_dataset(
        tmp_path,
        ["sf_a", "sf_b", "sf_c"],
        [
            {"instance_id": "sf_a", "primary_tier": 1, "tags": ["aggregation"]},
            {"instance_id": "sf_c", "primary_tier": 3, "tags": ["temporal"]},
        ],
    )

    tasks = select_tasks(
        ["sf_a", "sf_b", "tag:aggregation"],
        dataset_path=dataset,
        batch_dir=batch_dir,
    )

    assert [task.instance_id for task in tasks] == ["sf_a"]


def test_category_selectors_reject_bad_ranges_and_tags(tmp_path: Path):
    dataset, batch_dir = _setup_category_dataset(
        tmp_path,
        ["sf_a"],
        [{"instance_id": "sf_a", "primary_tier": 1, "tags": ["aggregation"]}],
    )

    with pytest.raises(ValueError, match="ascending"):
        select_tasks(["tier:4-2"], dataset_path=dataset, batch_dir=batch_dir)
    with pytest.raises(ValueError, match="unknown tag selector"):
        select_tasks(["tag:not_a_tag"], dataset_path=dataset, batch_dir=batch_dir)
