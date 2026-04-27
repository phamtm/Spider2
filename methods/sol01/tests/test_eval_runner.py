"""Tests for the official evaluation wrapper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from sol01.eval_runner import (
    EVALUATE_SCRIPT,
    GOLD_DIR,
    build_eval_command,
    parse_eval_stdout,
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
        '{"task_ids": ["local003", "local004", "local007"]}\n',
        encoding="utf-8",
    )
    (run_paths.csv_dir / "local003.csv").write_text("answer\n1\n", encoding="utf-8")

    def fake_runner(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        assert command[0] == sys.executable
        assert command[1] == str(EVALUATE_SCRIPT)
        assert cwd.name == "evaluation_suite"
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=(
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
    assert summary["correct_local_tasks"] == 2
    assert summary["attempted_local_tasks"] == 4
    assert summary["missing_csv_count"] == 2
    assert summary["missing_instance_ids"] == ["local004", "local007"]
    assert (run_paths.eval_dir / "official_stdout.txt").exists()
    assert (run_paths.eval_dir / "official_stderr.txt").exists()
    assert (run_paths.eval_dir / "summary.json").exists()


def test_parse_eval_stdout_returns_local_and_full_scores():
    stdout = (
        "{'local003': 1, 'local004': 0}\n"
        "Final score: 0.5, Correct examples: 2, Total examples: 4\n"
        "Real score: 0.003656307129798903, Correct examples: 2, Total examples: 547\n"
    )

    summary = parse_eval_stdout(stdout)

    assert summary == {
        "attempted_local_tasks": 4,
        "correct_local_tasks": 2,
        "local_subset_total": 135,
        "local_subset_score": 2 / 135,
        "full_benchmark_total": 547,
        "full_benchmark_equivalent_score": 0.003656307129798903,
    }


def test_run_official_eval_persists_output_on_failure(tmp_path: Path):
    run_paths = ensure_run_paths("broken-run", outputs_root=tmp_path)
    run_paths.manifest_path.write_text('{"task_ids": ["local003"]}\n', encoding="utf-8")

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
