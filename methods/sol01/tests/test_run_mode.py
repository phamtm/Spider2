"""Tests for the persisted local run mode."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sol01 import run_mode
from sol01.config import RuntimeConfig
from sol01.models import FinalAnswer, Task


def test_run_persisted_mode_exact_selector_updates_registry_and_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(run_mode, "_utc_now", lambda: "260429.1200")
    captured: dict[str, object] = {}

    def fake_run_tasks(
        tasks: list[Task],
        *,
        run_id: str,
        config: RuntimeConfig,
        outputs_root: Path | None = None,
        force: bool = False,
        skip_failed: bool = False,
    ) -> list[FinalAnswer]:
        captured["task_ids"] = [task.instance_id for task in tasks]
        captured["run_id"] = run_id
        captured["outputs_root"] = outputs_root
        return [
            FinalAnswer(
                instance_id=task.instance_id,
                status="success",
                sql="SELECT 1",
                csv_path=str(tmp_path / f"{task.instance_id}.csv"),
                trace_path=str(tmp_path / f"{task.instance_id}.json"),
            )
            for task in tasks
        ]

    def fake_eval(
        run_id: str,
        *,
        expected_instance_ids: list[str],
        outputs_root: Path | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        captured["eval_expected"] = expected_instance_ids
        return {
            "attempted_tasks": 1,
            "correct_tasks": 1,
            "missing_csv_count": 0,
            "per_instance": [
                {
                    "instance_id": "sf_local003",
                    "score": 1,
                    "passed": True,
                    "csv_present": True,
                    "failure_reason": None,
                }
            ],
            "instance_scores": {"sf_local003": 1},
            "result_dir": str(tmp_path / run_id / "eval" / "scored_csv"),
            "returncode": 0,
        }

    def fake_registry(records, *, outputs_root: Path) -> dict[str, object]:
        records = list(records)
        captured["registry_records"] = records
        return {"task_results": [{"instance_id": record.instance_id} for record in records]}

    monkeypatch.setattr(run_mode, "run_tasks", fake_run_tasks)
    monkeypatch.setattr(run_mode, "run_persisted_eval", fake_eval)
    monkeypatch.setattr(run_mode, "record_registry_batch", fake_registry)

    result = run_mode.run_persisted_mode(
        ["sf_local003"],
        outputs_root=tmp_path,
        config=RuntimeConfig(api_key="test-key"),
    )

    assert captured["task_ids"] == ["sf_local003"]
    assert captured["eval_expected"] == ["sf_local003"]
    assert result["run_id"] == "run-sf-local003-260429.1200"
    assert result["task_count"] == 1
    assert (tmp_path / result["run_id"]).exists()
    assert (tmp_path / result["run_id"] / "logs" / "stdout.txt").exists()
    assert (tmp_path / result["run_id"] / "logs" / "stderr.txt").exists()
    assert (tmp_path / result["run_id"] / "logs" / "run.jsonl").exists()
    output = capsys.readouterr().out
    assert "Run ID: run-sf-local003-260429.1200" in output
    assert "Exec time:" in output
    run_events = [
        json.loads(line)
        for line in (tmp_path / result["run_id"] / "logs" / "run.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [event["event"] for event in run_events] == [
        "run_started",
        "solver_complete",
        "eval_complete",
        "registry_updated",
        "run_complete",
    ]
    records = captured["registry_records"]
    assert len(records) == 1
    assert records[0].instance_id == "sf_local003"
    assert records[0].solver_status == "success"
    assert records[0].score == 1


def test_run_persisted_mode_multiple_globs_dedupe_and_preserve_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(run_mode, "_utc_now", lambda: "260429.1200")
    captured: dict[str, object] = {}

    def fake_run_tasks(
        tasks: list[Task],
        *,
        run_id: str,
        config: RuntimeConfig,
        outputs_root: Path | None = None,
        force: bool = False,
        skip_failed: bool = False,
    ) -> list[FinalAnswer]:
        captured["task_ids"] = [task.instance_id for task in tasks]
        return [
            FinalAnswer(
                instance_id=task.instance_id,
                status="success",
                sql="SELECT 1",
                csv_path=str(tmp_path / f"{task.instance_id}.csv"),
                trace_path=str(tmp_path / f"{task.instance_id}.json"),
            )
            for task in tasks
        ]

    def fake_eval(
        run_id: str,
        *,
        expected_instance_ids: list[str],
        outputs_root: Path | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        captured["eval_expected"] = expected_instance_ids
        return {
            "attempted_tasks": 2,
            "correct_tasks": 2,
            "missing_csv_count": 0,
            "per_instance": [
                {
                    "instance_id": instance_id,
                    "score": 1,
                    "passed": True,
                    "csv_present": True,
                    "failure_reason": None,
                }
                for instance_id in expected_instance_ids
            ],
            "instance_scores": {instance_id: 1 for instance_id in expected_instance_ids},
            "result_dir": str(tmp_path / run_id / "eval" / "scored_csv"),
            "returncode": 0,
        }

    monkeypatch.setattr(run_mode, "run_tasks", fake_run_tasks)
    monkeypatch.setattr(run_mode, "run_persisted_eval", fake_eval)
    monkeypatch.setattr(
        run_mode,
        "record_registry_batch",
        lambda records, *, outputs_root: {"task_results": list(records)},
    )

    result = run_mode.run_persisted_mode(
        ["sf_local003", "sf_local00[34]"],
        outputs_root=tmp_path,
        config=RuntimeConfig(api_key="test-key"),
    )

    assert captured["task_ids"] == ["sf_local003", "sf_local004"]
    assert captured["eval_expected"] == ["sf_local003", "sf_local004"]
    assert result["run_id"].startswith("run-pat-")
    assert result["run_id"].endswith("260429.1200")


def test_run_persisted_mode_all_mode_resolves_full_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(run_mode, "_utc_now", lambda: "260429.1200")
    captured: dict[str, object] = {}

    def fake_run_tasks(
        tasks: list[Task],
        *,
        run_id: str,
        config: RuntimeConfig,
        outputs_root: Path | None = None,
        force: bool = False,
        skip_failed: bool = False,
    ) -> list[FinalAnswer]:
        captured["task_count"] = len(tasks)
        return [
            FinalAnswer(
                instance_id=task.instance_id,
                status="success",
                sql="SELECT 1",
                csv_path=str(tmp_path / f"{task.instance_id}.csv"),
                trace_path=str(tmp_path / f"{task.instance_id}.json"),
            )
            for task in tasks
        ]

    def fake_eval(
        run_id: str,
        *,
        expected_instance_ids: list[str],
        outputs_root: Path | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        return {
            "attempted_tasks": len(expected_instance_ids),
            "correct_tasks": len(expected_instance_ids),
            "missing_csv_count": 0,
            "per_instance": [
                {
                    "instance_id": instance_id,
                    "score": 1,
                    "passed": True,
                    "csv_present": True,
                    "failure_reason": None,
                }
                for instance_id in expected_instance_ids
            ],
            "instance_scores": {instance_id: 1 for instance_id in expected_instance_ids},
            "result_dir": str(tmp_path / run_id / "eval" / "scored_csv"),
            "returncode": 0,
        }

    monkeypatch.setattr(run_mode, "run_tasks", fake_run_tasks)
    monkeypatch.setattr(run_mode, "run_persisted_eval", fake_eval)
    monkeypatch.setattr(
        run_mode,
        "record_registry_batch",
        lambda records, *, outputs_root: {"task_results": list(records)},
    )

    result = run_mode.run_persisted_mode(
        all_mode=True,
        outputs_root=tmp_path,
        config=RuntimeConfig(api_key="test-key"),
    )

    assert captured["task_count"] == 547
    assert result["run_id"].startswith("run-all-260429.1200")


def test_run_mode_main_forwards_tier_and_tag_filters(monkeypatch: pytest.MonkeyPatch):
    called: dict[str, object] = {}

    def fake_run_persisted_mode(selectors=None, **kwargs):
        called["selectors"] = selectors
        called.update(kwargs)
        return {}

    monkeypatch.setattr(run_mode, "run_persisted_mode", fake_run_persisted_mode)

    exit_code = run_mode.main(
        ["sf_local*", "--tier", "3-5", "--tag", "aggregation", "--tag", "temporal"]
    )

    assert exit_code == 0
    assert called == {
        "selectors": ["sf_local*"],
        "tiers": ["3-5"],
        "tags": ["aggregation", "temporal"],
        "all_mode": False,
    }


def test_run_mode_main_forwards_explicit_concurrency(monkeypatch: pytest.MonkeyPatch):
    called: dict[str, object] = {}

    def fake_run_persisted_mode(selectors=None, **kwargs):
        called["selectors"] = selectors
        called.update(kwargs)
        return {}

    monkeypatch.setattr(run_mode, "run_persisted_mode", fake_run_persisted_mode)

    exit_code = run_mode.main(["sf_local*", "--concurrency", "6"])

    assert exit_code == 0
    assert called["concurrency"] == 6


def test_run_mode_main_rejects_all_with_category_filters(capsys: pytest.CaptureFixture[str]):
    exit_code = run_mode.main(["--all", "--tier", "3"])

    assert exit_code == 2
    assert "must stand alone" in capsys.readouterr().err


def test_run_mode_main_help_mentions_selector_rules(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as excinfo:
        run_mode.main(["--help"])

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "Task selectors are ORed" in output
    assert "repeated --tag values are ANDed" in output
    assert "must be used by itself." in output.lower()


def test_mode_label_is_order_insensitive_for_equivalent_selectors():
    assert run_mode._mode_label(["sf_*", "tier:3"], all_mode=False) == run_mode._mode_label(
        ["tier:3", "sf_*"],
        all_mode=False,
    )


def test_run_persisted_mode_rejects_bare_star(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="all selector"):
        run_mode.run_persisted_mode(
            ["*"],
            outputs_root=tmp_path,
            config=RuntimeConfig(api_key="test-key"),
        )


def test_run_persisted_mode_marks_eval_failed_rows_in_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(run_mode, "_utc_now", lambda: "260429.1200")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        run_mode,
        "run_tasks",
        lambda tasks, *, run_id, config, outputs_root=None, force=False, skip_failed=False: [
            FinalAnswer(
                instance_id=task.instance_id,
                status="success",
                sql="SELECT 1",
                csv_path=str(tmp_path / f"{task.instance_id}.csv"),
                trace_path=str(tmp_path / f"{task.instance_id}.json"),
            )
            for task in tasks
        ],
    )
    monkeypatch.setattr(
        run_mode,
        "run_persisted_eval",
        lambda run_id, *, expected_instance_ids, outputs_root=None, **kwargs: {
            "attempted_tasks": 1,
            "correct_tasks": 0,
            "missing_csv_count": 0,
            "per_instance": [
                {
                    "instance_id": "sf_local003",
                    "score": None,
                    "passed": False,
                    "csv_present": True,
                    "failure_reason": "eval_failed",
                }
            ],
            "instance_scores": {},
            "result_dir": str(tmp_path / run_id / "eval" / "scored_csv"),
            "returncode": 0,
        },
    )

    def fake_registry(records, *, outputs_root: Path) -> dict[str, object]:
        records = list(records)
        captured["records"] = records
        return {"task_results": [{"instance_id": record.instance_id} for record in records]}

    monkeypatch.setattr(run_mode, "record_registry_batch", fake_registry)

    run_mode.run_persisted_mode(
        ["sf_local003"],
        outputs_root=tmp_path,
        config=RuntimeConfig(api_key="test-key"),
    )

    records = captured["records"]
    assert records[0].eval_status == "failed"
    assert records[0].eval_error == "eval_failed"


def test_run_persisted_mode_rejects_run_id_collisions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(run_mode, "_utc_now", lambda: "260429.1200")
    monkeypatch.setattr(
        run_mode,
        "run_tasks",
        lambda tasks, *, run_id, config, outputs_root=None, force=False, skip_failed=False: [
            FinalAnswer(
                instance_id=task.instance_id,
                status="success",
                sql="SELECT 1",
                csv_path=str(tmp_path / f"{task.instance_id}.csv"),
                trace_path=str(tmp_path / f"{task.instance_id}.json"),
            )
            for task in tasks
        ],
    )
    monkeypatch.setattr(
        run_mode,
        "run_persisted_eval",
        lambda run_id, *, expected_instance_ids, outputs_root=None, **kwargs: {
            "attempted_tasks": len(expected_instance_ids),
            "correct_tasks": len(expected_instance_ids),
            "missing_csv_count": 0,
            "per_instance": [
                {
                    "instance_id": instance_id,
                    "score": 1,
                    "passed": True,
                    "csv_present": True,
                    "failure_reason": None,
                }
                for instance_id in expected_instance_ids
            ],
            "instance_scores": {instance_id: 1 for instance_id in expected_instance_ids},
            "result_dir": str(tmp_path / "unused"),
            "returncode": 0,
        },
    )
    monkeypatch.setattr(
        run_mode,
        "record_registry_batch",
        lambda records, *, outputs_root: {"task_results": list(records)},
    )

    run_mode.run_persisted_mode(
        ["sf_local003"],
        outputs_root=tmp_path,
        config=RuntimeConfig(api_key="test-key"),
    )

    with pytest.raises(ValueError, match="collision"):
        run_mode.run_persisted_mode(
            ["sf_local003"],
            outputs_root=tmp_path,
            config=RuntimeConfig(api_key="test-key"),
        )
