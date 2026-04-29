"""Tests for the official evaluation wrapper."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from sol01.eval_runner import (
    BENCHMARK_TOTAL,
    EVALUATE_SCRIPT,
    GOLD_DIR,
    build_eval_command,
    parse_eval_stdout,
    run_persisted_eval,
    run_official_eval,
)
from sol01.output import ensure_run_paths


def test_build_eval_command_uses_active_python_and_exec_result_mode(tmp_path: Path):
    run_paths = ensure_run_paths("smoke-local003", outputs_root=tmp_path)

    command = build_eval_command(run_paths)

    assert command == [
        sys.executable,
        str(EVALUATE_SCRIPT),
        "--result_dir",
        str(run_paths.csv_dir),
        "--mode",
        "exec_result",
        "--gold_dir",
        str(GOLD_DIR),
    ]


def test_run_official_eval_writes_stdout_and_summary(tmp_path: Path):
    run_paths = ensure_run_paths("smoke-local003", outputs_root=tmp_path)
    run_paths.manifest_path.write_text(
        '{"task_ids": ["sf_local003", "sf_local004", "sf_local007"]}\n',
        encoding="utf-8",
    )
    (run_paths.csv_dir / "sf_local003.csv").write_text("answer\n1\n", encoding="utf-8")

    def fake_runner(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        assert command[0] == sys.executable
        assert command[1] == str(EVALUATE_SCRIPT)
        assert cwd.name.startswith("sol01-snow-eval-")
        assert (cwd / "snowflake_credential.json").exists()
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=(
                "{'sf_local003': 1, 'sf_local004': 0}\n"
                "Final score: 0.5, Correct examples: 2, Total examples: 4\n"
                "Real score: 0.003656307129798903, Correct examples: 2, Total examples: 547\n"
            ),
            stderr="",
        )

    summary = run_official_eval(
        "smoke-local003",
        outputs_root=tmp_path,
        runner=fake_runner,
    )

    assert summary["run_id"] == "smoke-local003"
    assert summary["correct_tasks"] == 2
    assert summary["attempted_tasks"] == 4
    assert summary["instance_scores"] == {"sf_local003": 1, "sf_local004": 0}
    assert summary["per_instance"] == [
        {"csv_present": True, "instance_id": "sf_local003", "passed": True, "score": 1},
        {"csv_present": False, "instance_id": "sf_local004", "passed": False, "score": 0},
        {"csv_present": False, "instance_id": "sf_local007", "passed": False, "score": None},
    ]
    assert summary["missing_csv_count"] == 2
    assert summary["missing_instance_ids"] == ["sf_local004", "sf_local007"]
    assert (run_paths.eval_dir / "official_stdout.txt").exists()
    assert (run_paths.eval_dir / "official_stderr.txt").exists()
    assert (run_paths.eval_dir / "summary.json").exists()


def test_parse_eval_stdout_returns_local_and_full_scores():
    stdout = (
        "{'sf_local003': 1, 'sf_local004': 0}\n"
        "Final score: 0.5, Correct examples: 2, Total examples: 4\n"
        "Real score: 0.003656307129798903, Correct examples: 2, Total examples: 547\n"
    )

    summary = parse_eval_stdout(stdout)

    assert summary == {
        "attempted_tasks": 4,
        "correct_tasks": 2,
        "attempted_score": 0.5,
        "benchmark_total": BENCHMARK_TOTAL,
        "benchmark_score": 0.003656307129798903,
        "instance_scores": {"sf_local003": 1, "sf_local004": 0},
    }


def test_run_official_eval_persists_output_on_failure(tmp_path: Path):
    run_paths = ensure_run_paths("broken-run", outputs_root=tmp_path)
    run_paths.manifest_path.write_text('{"task_ids": ["sf_local003"]}\n', encoding="utf-8")

    def failing_runner(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=command,
            returncode=2,
            stdout="partial stdout\n",
            stderr="failure details\n",
        )

    try:
        run_official_eval("broken-run", outputs_root=tmp_path, runner=failing_runner)
    except subprocess.CalledProcessError as exc:
        assert exc.returncode == 2
    else:
        raise AssertionError("Expected CalledProcessError")

    assert (run_paths.eval_dir / "official_stdout.txt").read_text(
        encoding="utf-8"
    ) == "partial stdout\n"
    assert (run_paths.eval_dir / "official_stderr.txt").read_text(
        encoding="utf-8"
    ) == "failure details\n"
    summary = (run_paths.eval_dir / "summary.json").read_text(encoding="utf-8")
    assert '"returncode": 2' in summary


def test_run_official_eval_with_artifact_tag_keeps_shared_summary_name_free(tmp_path: Path):
    run_paths = ensure_run_paths("filtered-run", outputs_root=tmp_path)
    run_paths.manifest_path.write_text('{"task_ids": ["sf_local003"]}\n', encoding="utf-8")

    def fake_runner(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="Final score: 1.0, Correct examples: 1, Total examples: 1\n",
            stderr="",
        )

    run_official_eval(
        "filtered-run",
        outputs_root=tmp_path,
        expected_instance_ids=["sf_local003"],
        artifact_tag="filtered-sf-local003",
        runner=fake_runner,
    )

    assert not (run_paths.eval_dir / "summary.json").exists()
    tagged_summary = run_paths.eval_dir / "summary.filtered-sf-local003.json"
    assert tagged_summary.exists()
    payload = tagged_summary.read_text(encoding="utf-8")
    assert '"result_dir"' not in payload


def test_run_official_eval_uses_the_scored_result_dir_for_bookkeeping(tmp_path: Path):
    ensure_run_paths("filtered-run", outputs_root=tmp_path)
    scored_dir = tmp_path / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)
    (scored_dir / "sf_local003.csv").write_text("answer\n1\n", encoding="utf-8")

    def fake_runner(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=(
                "{'sf_local003': 1}\nFinal score: 1.0, Correct examples: 1, Total examples: 1\n"
            ),
            stderr="",
        )

    summary = run_official_eval(
        "filtered-run",
        outputs_root=tmp_path,
        expected_instance_ids=["sf_local003", "sf_local004"],
        result_dir=scored_dir,
        runner=fake_runner,
    )

    assert summary["result_dir"] == str(scored_dir)
    assert summary["missing_csv_count"] == 1
    assert summary["missing_instance_ids"] == ["sf_local004"]
    assert summary["per_instance"] == [
        {"csv_present": True, "instance_id": "sf_local003", "passed": True, "score": 1},
        {"csv_present": False, "instance_id": "sf_local004", "passed": False, "score": None},
    ]


def test_run_persisted_eval_copies_scored_csvs_and_writes_per_instance_records(
    tmp_path: Path,
):
    run_paths = ensure_run_paths("persisted-run", outputs_root=tmp_path)
    run_paths.manifest_path.write_text(
        '{"task_ids": ["sf_local003", "sf_local004"]}\n',
        encoding="utf-8",
    )
    (run_paths.csv_dir / "sf_local003.csv").write_text("answer\n1\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_runner(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["cwd"] = cwd
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=(
                "{'sf_local003': 1}\n"
                "Final score: 1.0, Correct examples: 1, Total examples: 1\n"
                "Real score: 0.001, Correct examples: 1, Total examples: 547\n"
            ),
            stderr="",
        )

    summary = run_persisted_eval(
        "persisted-run",
        outputs_root=tmp_path,
        expected_instance_ids=["sf_local003", "sf_local004"],
        runner=fake_runner,
    )

    assert summary["result_dir"] == str(run_paths.eval_scored_csv_dir)
    assert captured["cwd"].name.startswith("sol01-snow-eval-")
    assert str(run_paths.eval_scored_csv_dir) in captured["command"]
    assert (run_paths.eval_scored_csv_dir / "sf_local003.csv").exists()
    assert not (run_paths.eval_scored_csv_dir / "sf_local004.csv").exists()
    per_instance_rows = [
        json.loads(line)
        for line in (run_paths.eval_dir / "per_instance.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert per_instance_rows == [
        {
            "csv_present": True,
            "failure_reason": None,
            "instance_id": "sf_local003",
            "passed": True,
            "score": 1,
        },
        {
            "csv_present": False,
            "failure_reason": "missing_csv",
            "instance_id": "sf_local004",
            "passed": False,
            "score": None,
        },
    ]
    assert summary["per_instance"] == per_instance_rows


def test_run_persisted_eval_handles_no_csv_without_crashing(tmp_path: Path):
    run_paths = ensure_run_paths("empty-persisted-run", outputs_root=tmp_path)
    run_paths.manifest_path.write_text(
        '{"task_ids": ["sf_local003", "sf_local004"]}\n',
        encoding="utf-8",
    )

    summary = run_persisted_eval(
        "empty-persisted-run",
        outputs_root=tmp_path,
        expected_instance_ids=["sf_local003", "sf_local004"],
    )

    assert summary["result_dir"] == str(run_paths.eval_scored_csv_dir)
    assert summary["missing_csv_count"] == 2
    assert summary["missing_instance_ids"] == ["sf_local003", "sf_local004"]
    assert summary["per_instance"] == [
        {
            "csv_present": False,
            "failure_reason": "missing_csv",
            "instance_id": "sf_local003",
            "passed": False,
            "score": None,
        },
        {
            "csv_present": False,
            "failure_reason": "missing_csv",
            "instance_id": "sf_local004",
            "passed": False,
            "score": None,
        },
    ]
    assert (run_paths.eval_dir / "summary.json").exists()
    assert (run_paths.eval_dir / "official_stdout.txt").read_text(encoding="utf-8") == ""
    assert (run_paths.eval_dir / "official_stderr.txt").read_text(encoding="utf-8") == ""
    assert (run_paths.eval_dir / "per_instance.jsonl").exists()


def test_run_official_eval_stages_local_credentials_without_mutating_suite(
    tmp_path: Path,
):
    run_paths = ensure_run_paths("credential-run", outputs_root=tmp_path)
    credential_path = tmp_path / "snowflake_credential.json"
    credential_path.write_text(
        """
        {
          "username": "alice",
          "password": "pat-token",
          "account": "RSRSBDK-YDB67606",
          "role": "PARTICIPANT",
          "warehouse": "COMPUTE_WH_PARTICIPANT",
          "session_parameters": {"QUERY_TAG": "sol01"}
        }
        """,
        encoding="utf-8",
    )
    suite_credential = EVALUATE_SCRIPT.parent / "snowflake_credential.json"
    before = suite_credential.read_text(encoding="utf-8") if suite_credential.exists() else None
    captured: dict[str, str] = {}

    def fake_runner(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        captured["credential"] = (cwd / "snowflake_credential.json").read_text(encoding="utf-8")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="Final score: 0.0, Correct examples: 0, Total examples: 0\n",
            stderr="",
        )

    run_official_eval(
        "credential-run",
        outputs_root=tmp_path,
        expected_instance_ids=[],
        credential_path=credential_path,
        runner=fake_runner,
    )

    assert '"user": "alice"' in captured["credential"]
    assert '"password": "pat-token"' in captured["credential"]
    assert '"QUERY_TAG": "sol01"' in captured["credential"]
    assert run_paths.eval_dir.exists()
    after = suite_credential.read_text(encoding="utf-8") if suite_credential.exists() else None
    assert after == before
