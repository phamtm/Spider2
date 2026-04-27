import pytest

from sol01.tasks import SPIDER2_LITE_PATH, load_tasks


def test_default_dataset_path_exists():
    assert SPIDER2_LITE_PATH.exists()


def test_local_task_count_is_135():
    tasks = load_tasks()

    assert len(tasks) == 135
    assert all(task.instance_id.startswith("local") for task in tasks)


def test_can_select_local003_by_instance_id():
    tasks = load_tasks(instance_id="local003")

    assert len(tasks) == 1
    assert tasks[0].instance_id == "local003"
    assert tasks[0].db == "E_commerce"
    assert tasks[0].external_knowledge == "RFM.md"


def test_db_question_and_limit_filters_can_be_combined():
    tasks = load_tasks(
        db="E_commerce",
        question_contains="average payment per order",
        limit=1,
    )

    assert len(tasks) == 1
    assert tasks[0].instance_id == "local004"


def test_question_filter_is_case_insensitive():
    tasks = load_tasks(question_contains="AVERAGE SINGLE CAREER SPAN")

    assert any(task.instance_id == "local007" for task in tasks)


def test_negative_limit_is_rejected():
    with pytest.raises(ValueError, match="limit"):
        load_tasks(limit=-1)
