"""Tests for the local sol01 run registry."""

from __future__ import annotations

import json
from pathlib import Path

from sol01.registry import (
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


def test_registry_preserves_eval_failed_for_missing_scores_with_csv(tmp_path: Path):
    record = _record(
        run_id="run-parse-failed",
        instance_id="sf_local011",
        timestamp="2026-04-29T02:30:00+00:00",
        score=None,
        run_path=str(tmp_path / "run-parse-failed"),
        csv_path=str(tmp_path / "run-parse-failed" / "csv" / "sf_local011.csv"),
        trace_path=str(tmp_path / "run-parse-failed" / "traces" / "sf_local011.json"),
        solver_status="success",
        eval_status="failed",
        eval_error="eval_failed",
    )

    latest = record_registry_batch([record], outputs_root=tmp_path)

    assert latest["task_results"][0]["status"] == "eval_failed"


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


def test_registry_rebuilds_stale_latest_and_skips_invalid_history_rows(tmp_path: Path):
    first = _record(
        run_id="run-h",
        instance_id="sf_local040",
        timestamp="2026-04-29T07:00:00+00:00",
        score=0.0,
        run_path=str(tmp_path / "run-h"),
        csv_path=str(tmp_path / "run-h" / "csv" / "sf_local040.csv"),
        solver_status="success",
    )
    second = _record(
        run_id="run-i",
        instance_id="sf_local041",
        timestamp="2026-04-29T08:00:00+00:00",
        score=1.0,
        run_path=str(tmp_path / "run-i"),
        csv_path=str(tmp_path / "run-i" / "csv" / "sf_local041.csv"),
        solver_status="success",
    )

    record_registry_batch([first], outputs_root=tmp_path)
    paths = ensure_registry_paths(outputs_root=tmp_path)
    stale_latest = load_latest(outputs_root=tmp_path)
    second_row = record_registry_batch([second], outputs_root=tmp_path)["task_results"][1]
    paths.latest_path.write_text(json.dumps(stale_latest, indent=2) + "\n", encoding="utf-8")
    with paths.task_results_path.open("a", encoding="utf-8") as handle:
        handle.write("{broken json\n")
    paths.task_results_path.touch()

    rebuilt = load_latest(outputs_root=tmp_path)

    assert {row["instance_id"] for row in rebuilt["task_results"]} == {
        "sf_local040",
        "sf_local041",
    }
    assert second_row in rebuilt["task_results"]


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
