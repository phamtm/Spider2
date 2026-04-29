"""Tests for the local sol01 run registry."""

from __future__ import annotations

import json
from pathlib import Path

from sol01.registry import (
    RegistryTaskRecord,
    ensure_registry_paths,
    load_latest,
    record_registry_batch,
)


def _record(
    *,
    run_id: str,
    instance_id: str,
    timestamp: str,
    score: float | None,
    run_path: str,
    csv_path: str | None = None,
    trace_path: str | None = None,
    eval_path: str | None = None,
    solver_status: str | None = None,
    eval_status: str | None = None,
    eval_error: str | None = None,
) -> RegistryTaskRecord:
    return RegistryTaskRecord(
        run_id=run_id,
        instance_id=instance_id,
        db="TEST_DB",
        timestamp=timestamp,
        score=score,
        run_path=run_path,
        csv_path=csv_path,
        trace_path=trace_path,
        eval_path=eval_path,
        solver_status=solver_status,
        eval_status=eval_status,
        eval_error=eval_error,
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_registry_writes_first_batch_and_appends_history(tmp_path: Path):
    records = [
        _record(
            run_id="run-a",
            instance_id="sf_local001",
            timestamp="2026-04-29T00:00:00+00:00",
            score=1.0,
            run_path=str(tmp_path / "run-a"),
            csv_path=str(tmp_path / "run-a" / "csv" / "sf_local001.csv"),
            trace_path=str(tmp_path / "run-a" / "traces" / "sf_local001.json"),
            solver_status="success",
        ),
        _record(
            run_id="run-a",
            instance_id="sf_local002",
            timestamp="2026-04-29T00:00:00+00:00",
            score=None,
            run_path=str(tmp_path / "run-a"),
            trace_path=str(tmp_path / "run-a" / "traces" / "sf_local002.json"),
            solver_status="success",
            eval_status="failed",
            eval_error="evaluator crashed",
        ),
    ]

    latest = record_registry_batch(records, outputs_root=tmp_path)
    paths = ensure_registry_paths(outputs_root=tmp_path)

    assert paths.root == tmp_path / "registry"
    assert paths.runs_path.exists()
    assert paths.task_results_path.exists()
    assert paths.latest_path.exists()
    assert _read_jsonl(paths.runs_path)[0]["task_count"] == 2
    assert len(_read_jsonl(paths.task_results_path)) == 2
    assert latest["task_count"] == 2
    statuses = {row["instance_id"]: row["status"] for row in latest["task_results"]}
    assert statuses == {
        "sf_local001": "pass",
        "sf_local002": "eval_failed",
    }


def test_registry_updates_latest_on_pass_to_fail(tmp_path: Path):
    first = _record(
        run_id="run-b",
        instance_id="sf_local010",
        timestamp="2026-04-29T01:00:00+00:00",
        score=1.0,
        run_path=str(tmp_path / "run-b"),
        csv_path=str(tmp_path / "run-b" / "csv" / "sf_local010.csv"),
        trace_path=str(tmp_path / "run-b" / "traces" / "sf_local010.json"),
        solver_status="success",
    )
    second = _record(
        run_id="run-c",
        instance_id="sf_local010",
        timestamp="2026-04-29T02:00:00+00:00",
        score=0.0,
        run_path=str(tmp_path / "run-c"),
        csv_path=str(tmp_path / "run-c" / "csv" / "sf_local010.csv"),
        trace_path=str(tmp_path / "run-c" / "traces" / "sf_local010.json"),
        solver_status="success",
    )

    record_registry_batch([first], outputs_root=tmp_path)
    latest = record_registry_batch([second], outputs_root=tmp_path)
    paths = ensure_registry_paths(outputs_root=tmp_path)

    assert len(_read_jsonl(paths.task_results_path)) == 2
    assert len(_read_jsonl(paths.runs_path)) == 2
    assert latest["task_results"][0]["status"] == "official_fail"
    assert latest["task_results"][0]["run_id"] == "run-c"


def test_registry_updates_latest_on_fail_to_pass(tmp_path: Path):
    first = _record(
        run_id="run-d",
        instance_id="sf_local020",
        timestamp="2026-04-29T03:00:00+00:00",
        score=None,
        run_path=str(tmp_path / "run-d"),
        trace_path=str(tmp_path / "run-d" / "traces" / "sf_local020.json"),
        solver_status="failed",
    )
    second = _record(
        run_id="run-e",
        instance_id="sf_local020",
        timestamp="2026-04-29T04:00:00+00:00",
        score=1.0,
        run_path=str(tmp_path / "run-e"),
        csv_path=str(tmp_path / "run-e" / "csv" / "sf_local020.csv"),
        trace_path=str(tmp_path / "run-e" / "traces" / "sf_local020.json"),
        solver_status="success",
    )

    record_registry_batch([first], outputs_root=tmp_path)
    latest = record_registry_batch([second], outputs_root=tmp_path)

    assert latest["task_results"][0]["status"] == "pass"
    assert latest["task_results"][0]["run_id"] == "run-e"


def test_registry_rebuilds_latest_from_history_when_missing(tmp_path: Path):
    first = _record(
        run_id="run-f",
        instance_id="sf_local030",
        timestamp="2026-04-29T05:00:00+00:00",
        score=0.0,
        run_path=str(tmp_path / "run-f"),
        csv_path=str(tmp_path / "run-f" / "csv" / "sf_local030.csv"),
        trace_path=str(tmp_path / "run-f" / "traces" / "sf_local030.json"),
        solver_status="success",
    )
    second = _record(
        run_id="run-g",
        instance_id="sf_local030",
        timestamp="2026-04-29T06:00:00+00:00",
        score=1.0,
        run_path=str(tmp_path / "run-g"),
        csv_path=str(tmp_path / "run-g" / "csv" / "sf_local030.csv"),
        trace_path=str(tmp_path / "run-g" / "traces" / "sf_local030.json"),
        solver_status="success",
    )

    record_registry_batch([first], outputs_root=tmp_path)
    record_registry_batch([second], outputs_root=tmp_path)

    latest_path = tmp_path / "registry" / "latest.json"
    latest_path.unlink()

    rebuilt = load_latest(outputs_root=tmp_path)

    assert latest_path.exists()
    assert rebuilt["task_results"][0]["status"] == "pass"
    assert rebuilt["task_results"][0]["run_id"] == "run-g"
