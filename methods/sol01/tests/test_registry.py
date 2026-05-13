"""Tests for the local sol01 run registry."""

from __future__ import annotations

import json
from pathlib import Path

from sol01.output.registry import (
    RegistryTaskRecord,
    ensure_registry_paths,
    load_latest,
    record_registry_batch,
    resolve_llm_call_log_path,
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
    extra_artifacts: dict[str, str | None] | None = None,
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
        extra_artifacts=extra_artifacts or {},
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


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
            extra_artifacts={
                "llm_call_log_path": str(tmp_path / "run-a" / "llm_calls" / "sf_local001.jsonl")
            },
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
    assert latest["task_results"][0]["extra_artifacts"]["llm_call_log_path"] == (
        str(tmp_path / "run-a" / "llm_calls" / "sf_local001.jsonl")
    )


def test_registry_updates_latest_on_status_change(tmp_path: Path):
    first_pass = _record(
        run_id="run-b",
        instance_id="sf_local010",
        timestamp="2026-04-29T01:00:00+00:00",
        score=1.0,
        run_path=str(tmp_path / "run-b"),
        csv_path=str(tmp_path / "run-b" / "csv" / "sf_local010.csv"),
        trace_path=str(tmp_path / "run-b" / "traces" / "sf_local010.json"),
        solver_status="success",
    )
    second_fail = _record(
        run_id="run-c",
        instance_id="sf_local010",
        timestamp="2026-04-29T02:00:00+00:00",
        score=0.0,
        run_path=str(tmp_path / "run-c"),
        csv_path=str(tmp_path / "run-c" / "csv" / "sf_local010.csv"),
        trace_path=str(tmp_path / "run-c" / "traces" / "sf_local010.json"),
        solver_status="success",
    )

    record_registry_batch([first_pass], outputs_root=tmp_path)
    latest = record_registry_batch([second_fail], outputs_root=tmp_path)
    by_id = {r["instance_id"]: r for r in latest["task_results"]}
    assert by_id["sf_local010"]["status"] == "official_fail"
    assert by_id["sf_local010"]["run_id"] == "run-c"

    first_fail = _record(
        run_id="run-d",
        instance_id="sf_local020",
        timestamp="2026-04-29T03:00:00+00:00",
        score=None,
        run_path=str(tmp_path / "run-d"),
        trace_path=str(tmp_path / "run-d" / "traces" / "sf_local020.json"),
        solver_status="failed",
    )
    second_pass = _record(
        run_id="run-e",
        instance_id="sf_local020",
        timestamp="2026-04-29T04:00:00+00:00",
        score=1.0,
        run_path=str(tmp_path / "run-e"),
        csv_path=str(tmp_path / "run-e" / "csv" / "sf_local020.csv"),
        trace_path=str(tmp_path / "run-e" / "traces" / "sf_local020.json"),
        solver_status="success",
    )

    record_registry_batch([first_fail], outputs_root=tmp_path)
    latest = record_registry_batch([second_pass], outputs_root=tmp_path)
    by_id = {r["instance_id"]: r for r in latest["task_results"]}
    assert by_id["sf_local020"]["status"] == "pass"
    assert by_id["sf_local020"]["run_id"] == "run-e"


def test_registry_rebuilds_latest_from_history(tmp_path: Path):
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


def test_resolve_llm_call_log_path_prefers_registry_extra_artifacts(tmp_path: Path):
    row = {
        "run_id": "run-x",
        "instance_id": "sf_local099",
        "trace_path": str(tmp_path / "run-x" / "traces" / "sf_local099.json"),
        "extra_artifacts": {
            "llm_call_log_path": str(tmp_path / "run-x" / "llm_calls" / "sf_local099.jsonl")
        },
    }

    assert resolve_llm_call_log_path(row, outputs_root=tmp_path) == (
        tmp_path / "run-x" / "llm_calls" / "sf_local099.jsonl"
    )


def test_resolve_llm_call_log_path_falls_back_to_trace_field(tmp_path: Path):
    trace_path = tmp_path / "run-y" / "traces" / "sf_local100.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        trace_path,
        {
            "instance_id": "sf_local100",
            "llm_call_log_path": str(tmp_path / "run-y" / "llm_calls" / "sf_local100.jsonl"),
        },
    )
    row = {
        "run_id": "run-y",
        "instance_id": "sf_local100",
        "trace_path": str(trace_path),
        "extra_artifacts": {},
    }

    assert resolve_llm_call_log_path(row, outputs_root=tmp_path) == (
        tmp_path / "run-y" / "llm_calls" / "sf_local100.jsonl"
    )


def test_resolve_llm_call_log_path_derives_older_run_path(tmp_path: Path):
    row = {
        "run_id": "run-z",
        "instance_id": "sf_local101",
        "trace_path": str(tmp_path / "run-z" / "traces" / "sf_local101.json"),
        "extra_artifacts": {},
    }

    assert resolve_llm_call_log_path(row, outputs_root=tmp_path) == (
        tmp_path / "run-z" / "llm_calls" / "sf_local101.jsonl"
    )
