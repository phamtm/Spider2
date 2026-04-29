"""Run the official Spider2-snow evaluator and save a summary."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from sol01.logging import get_logger
from sol01.output import RunPaths, ensure_run_paths
from sol01.tasks import REPO_ROOT, load_tasks

EVALUATION_SUITE_DIR = REPO_ROOT / "spider2-snow" / "evaluation_suite"
EVALUATE_SCRIPT = EVALUATION_SUITE_DIR / "evaluate.py"
GOLD_DIR = EVALUATION_SUITE_DIR / "gold"
LOCAL_SUBSET_TOTAL = 135
FULL_BENCHMARK_TOTAL = 547
FINAL_SCORE_RE = re.compile(
    r"Final score:\s*([0-9.]+),\s*Correct examples:\s*(\d+),\s*Total examples:\s*(\d+)"
)
REAL_SCORE_RE = re.compile(
    r"Real score:\s*([0-9.]+),\s*Correct examples:\s*(\d+),\s*Total examples:\s*(\d+)"
)
Runner = Any
logger = get_logger(__name__)


def run_official_eval(
    run_id: str,
    *,
    python_executable: str | None = None,
    outputs_root: Path | None = None,
    expected_instance_ids: list[str] | None = None,
    artifact_tag: str | None = None,
    result_dir: Path | None = None,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Run the official evaluator in exec_result mode and write a summary file."""

    run_paths = ensure_run_paths(
        run_id,
        outputs_root=outputs_root or REPO_ROOT / "methods" / "sol01" / "outputs",
    )
    scored_result_dir = result_dir or run_paths.csv_dir
    logger.info(
        "evaluation start",
        run_id=run_id,
        result_dir=str(scored_result_dir),
        expected_instance_count=len(expected_instance_ids) if expected_instance_ids else None,
        artifact_tag=artifact_tag,
    )
    command = build_eval_command(
        run_paths,
        python_executable=python_executable,
        result_dir=scored_result_dir,
    )
    completed = (runner or _run_subprocess)(command, cwd=EVALUATION_SUITE_DIR)

    suffix = f".{artifact_tag}" if artifact_tag else ""
    stdout_path = run_paths.eval_dir / f"official_stdout{suffix}.txt"
    stderr_path = run_paths.eval_dir / f"official_stderr{suffix}.txt"
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")

    # Some injected test runners may return None instead of an empty string.
    summary = parse_eval_stdout(completed.stdout or "")
    summary["run_id"] = run_id
    summary["stdout_path"] = str(stdout_path)
    summary["stderr_path"] = str(stderr_path)
    if artifact_tag is None:
        summary["result_dir"] = str(scored_result_dir)
    expected_ids = (
        set(expected_instance_ids)
        if expected_instance_ids is not None
        else _expected_instance_ids(run_paths)
    )
    summary["per_instance"] = _per_instance_summary(
        expected_ids=expected_ids,
        instance_scores=summary.get("instance_scores", {}),
        result_dir=scored_result_dir,
    )
    summary["missing_csv_count"] = _missing_csv_count(
        scored_result_dir,
        expected_ids=expected_ids,
    )
    summary["missing_instance_ids"] = _missing_instance_ids(
        scored_result_dir,
        expected_ids=expected_ids,
    )
    summary["returncode"] = completed.returncode

    summary_path = run_paths.eval_dir / f"summary{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(
        "evaluation complete",
        run_id=run_id,
        returncode=completed.returncode,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        summary_path=str(summary_path),
        missing_csv_count=summary["missing_csv_count"],
        correct_local_tasks=summary["correct_local_tasks"],
        attempted_local_tasks=summary["attempted_local_tasks"],
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            returncode=completed.returncode,
            cmd=command,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return summary


def build_eval_command(
    run_paths: RunPaths,
    *,
    python_executable: str | None = None,
    result_dir: Path | None = None,
) -> list[str]:
    """Build the official evaluator subprocess command."""

    return [
        python_executable or sys.executable,
        str(EVALUATE_SCRIPT),
        "--result_dir",
        str(result_dir or run_paths.csv_dir),
        "--mode",
        "exec_result",
        "--gold_dir",
        str(GOLD_DIR),
    ]


def parse_eval_stdout(stdout: str) -> dict[str, Any]:
    """Parse the evaluator stdout into stable local summary fields."""

    final_match = FINAL_SCORE_RE.search(stdout)
    real_match = REAL_SCORE_RE.search(stdout)

    summary: dict[str, Any] = {
        "attempted_local_tasks": 0,
        "correct_local_tasks": 0,
        "local_subset_total": LOCAL_SUBSET_TOTAL,
        "local_subset_score": 0.0,
        "full_benchmark_total": FULL_BENCHMARK_TOTAL,
        "full_benchmark_equivalent_score": 0.0,
        "instance_scores": _parse_instance_scores(stdout),
    }

    if final_match:
        summary["attempted_local_tasks"] = int(final_match.group(3))
        summary["correct_local_tasks"] = int(final_match.group(2))
        summary["local_subset_score"] = int(final_match.group(2)) / LOCAL_SUBSET_TOTAL

    if real_match:
        summary["full_benchmark_equivalent_score"] = float(real_match.group(1))
    elif final_match:
        summary["full_benchmark_equivalent_score"] = (
            int(final_match.group(2)) / FULL_BENCHMARK_TOTAL
        )

    return summary


def _parse_instance_scores(stdout: str) -> dict[str, int]:
    """Read the evaluator's per-instance score map from stdout when present."""

    for line in stdout.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            continue
        try:
            payload = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        if not all(isinstance(key, str) for key in payload):
            continue
        if not all(isinstance(value, int) for value in payload.values()):
            continue
        return dict(sorted(payload.items()))
    return {}


def _per_instance_summary(
    *,
    expected_ids: set[str],
    instance_scores: dict[str, int],
    result_dir: Path,
) -> list[dict[str, Any]]:
    """Build stable per-task eval records for CLI output and saved summaries."""

    rows: list[dict[str, Any]] = []
    for instance_id in sorted(expected_ids):
        score = instance_scores.get(instance_id)
        rows.append(
            {
                "instance_id": instance_id,
                "score": score,
                "passed": score == 1,
                "csv_present": (result_dir / f"{instance_id}.csv").exists(),
            }
        )
    return rows


def _missing_instance_ids(result_dir: Path, *, expected_ids: set[str]) -> list[str]:
    """List local task IDs that do not have a CSV in the run output."""

    present_ids = {path.stem for path in result_dir.glob("*.csv")}
    return sorted(expected_ids - present_ids)


def _missing_csv_count(result_dir: Path, *, expected_ids: set[str]) -> int:
    """Count how many local tasks still do not have a CSV output."""

    return len(_missing_instance_ids(result_dir, expected_ids=expected_ids))


def _expected_instance_ids(run_paths: RunPaths) -> set[str]:
    """Use manifest task IDs when present, otherwise fall back to all local tasks."""

    if run_paths.manifest_path.exists():
        manifest = json.loads(run_paths.manifest_path.read_text(encoding="utf-8"))
        task_ids = manifest.get("task_ids")
        if isinstance(task_ids, list) and all(isinstance(item, str) for item in task_ids):
            return set(task_ids)
    return {task.instance_id for task in load_tasks()}


def _run_subprocess(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the evaluator subprocess and return its captured stdout."""

    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
