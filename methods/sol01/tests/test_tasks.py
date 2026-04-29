import pytest

from sol01.tasks import ALL_TASK_SELECTOR, SPIDER2_SNOW_PATH, load_tasks, select_tasks


def test_default_dataset_path_exists():
    assert SPIDER2_SNOW_PATH.exists()


def test_snow_task_count_is_547():
    tasks = load_tasks()

    assert len(tasks) == 547
    assert all(task.instance_id.startswith("sf") for task in tasks)


def test_can_select_sf_local003_by_instance_id():
    tasks = load_tasks(instance_id="sf_local003")

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
    tasks = load_tasks(
        db="E_COMMERCE",
        question_contains="average payment per order",
        limit=1,
    )

    assert len(tasks) == 1
    assert tasks[0].instance_id == "sf_local004"


def test_question_filter_is_case_insensitive():
    tasks = load_tasks(question_contains="AVERAGE SINGLE CAREER SPAN")

    assert any(task.instance_id == "sf_local007" for task in tasks)


def test_negative_limit_is_rejected():
    with pytest.raises(ValueError, match="limit"):
        load_tasks(limit=-1)


@pytest.mark.parametrize("selectors", [["*"], ["sf_local003/extra"], ["sf_local003..bad"]])
def test_invalid_selectors_are_rejected(selectors):
    with pytest.raises(ValueError, match="not allowed"):
        select_tasks(selectors)


def test_zero_match_selectors_are_rejected():
    with pytest.raises(ValueError, match="No tasks matched"):
        select_tasks(["sf_does_not_exist*"])
