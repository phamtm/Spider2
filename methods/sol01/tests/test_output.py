"""Tests for output layout creation and resume decisions."""

from __future__ import annotations

import json

from sol01.output import (
    csv_path_for,
    ensure_ask_paths,
    ensure_run_paths,
    should_skip_task,
    write_manifest,
    write_sql,
    write_trace,
)


def test_ensure_run_paths_and_manifest_creation(tmp_path):
    run_paths = ensure_run_paths("smoke-local003", outputs_root=tmp_path)

    manifest_path = write_manifest(
        run_paths,
        manifest={
            "task_ids": ["local003"],
            "model": "deepseek/deepseek-v4-pro",
        },
    )
    sql_path = write_sql(run_paths, instance_id="local003", sql="SELECT 1")
    trace_path = write_trace(
        run_paths,
        instance_id="local003",
        trace={"status": "success", "attempts": []},
    )

    assert run_paths.root == tmp_path / "smoke-local003"
    assert run_paths.sql_dir.exists()
    assert run_paths.csv_dir.exists()
    assert run_paths.traces_dir.exists()
    assert run_paths.eval_dir.exists()
    assert run_paths.analysis_dir.exists()
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["task_ids"] == ["local003"]
    assert sql_path.read_text(encoding="utf-8") == "SELECT 1\n"
    assert json.loads(trace_path.read_text(encoding="utf-8"))["status"] == "success"


def test_successful_task_is_skipped_when_trace_and_csv_exist(tmp_path):
    run_paths = ensure_run_paths("resume-run", outputs_root=tmp_path)
    write_trace(run_paths, instance_id="local003", trace={"status": "success"})
    csv_path_for(run_paths, instance_id="local003").write_text("answer\n1\n", encoding="utf-8")

    assert should_skip_task(run_paths, instance_id="local003") is True


def test_successful_task_reruns_when_csv_is_missing(tmp_path):
    run_paths = ensure_run_paths("resume-run", outputs_root=tmp_path)
    write_trace(run_paths, instance_id="local003", trace={"status": "success"})

    assert should_skip_task(run_paths, instance_id="local003") is False


def test_failed_trace_reruns_by_default_and_skip_failed_can_skip(tmp_path):
    run_paths = ensure_run_paths("resume-run", outputs_root=tmp_path)
    write_trace(run_paths, instance_id="local004", trace={"status": "failed"})

    assert should_skip_task(run_paths, instance_id="local004") is False
    assert should_skip_task(run_paths, instance_id="local004", skip_failed=True) is True


def test_skipped_trace_stays_skipped_on_resume(tmp_path):
    run_paths = ensure_run_paths("resume-run", outputs_root=tmp_path)
    write_trace(run_paths, instance_id="local005", trace={"status": "skipped"})

    assert should_skip_task(run_paths, instance_id="local005") is True


def test_ensure_ask_paths_creates_default_layout(tmp_path):
    ask_paths = ensure_ask_paths(timestamp="20260427T120000Z", outputs_root=tmp_path)

    assert ask_paths.root == tmp_path / "ask" / "20260427T120000Z"
    assert ask_paths.root.exists()
    assert ask_paths.sql_path.name == "sql.sql"
    assert ask_paths.csv_path.name == "result.csv"
    assert ask_paths.trace_path.name == "trace.json"


def test_ensure_ask_paths_avoids_collisions_without_explicit_timestamp(tmp_path):
    first = ensure_ask_paths(outputs_root=tmp_path)
    second = ensure_ask_paths(outputs_root=tmp_path)

    assert first.root != second.root
    assert first.root.exists()
    assert second.root.exists()
