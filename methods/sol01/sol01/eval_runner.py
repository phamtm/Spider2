"""Run the official Spider2-snow evaluator and save a summary."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from shutil import copy2
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from sol01.logging import get_logger
from sol01.output import (
    RunPaths,
    ensure_run_paths,
    per_instance_eval_path_for,
    scored_csv_dir_for,
)
from sol01.snowflake_runner import DEFAULT_CREDENTIAL_PATH
from sol01.tasks import REPO_ROOT, load_tasks

EVALUATION_SUITE_DIR = REPO_ROOT / "spider2-snow" / "evaluation_suite"
EVALUATE_SCRIPT = EVALUATION_SUITE_DIR / "evaluate.py"
GOLD_DIR = EVALUATION_SUITE_DIR / "gold"
BENCHMARK_TOTAL = 547
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
    credential_path: Path | None = None,
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
    with TemporaryDirectory(prefix="sol01-snow-eval-") as temp_dir:
        eval_workspace = Path(temp_dir)
        _write_eval_credential(
            eval_workspace / "snowflake_credential.json",
            credential_path=credential_path,
        )
        completed = (runner or _run_subprocess)(command, cwd=eval_workspace)

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
        correct_tasks=summary["correct_tasks"],
        attempted_tasks=summary["attempted_tasks"],
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            returncode=completed.returncode,
            cmd=command,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return summary


def run_persisted_eval(
    run_id: str,
    *,
    expected_instance_ids: list[str],
    python_executable: str | None = None,
    outputs_root: Path | None = None,
    artifact_tag: str | None = None,
    credential_path: Path | None = None,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Persist scored CSVs, run eval, and write durable per-instance records."""

    run_paths = ensure_run_paths(
        run_id,
        outputs_root=outputs_root or REPO_ROOT / "methods" / "sol01" / "outputs",
    )
    persisted_result_dir = scored_csv_dir_for(run_paths)
    copied_csv_count = _copy_scored_csvs(run_paths.csv_dir, persisted_result_dir)
    expected_ids = set(expected_instance_ids)

    if copied_csv_count == 0:
        summary = _build_no_csv_summary(
            run_id=run_id,
            run_paths=run_paths,
            expected_ids=expected_ids,
            result_dir=persisted_result_dir,
        )
        _write_per_instance_records(
            per_instance_eval_path_for(run_paths),
            summary["per_instance"],
        )
        return summary

    summary = run_official_eval(
        run_id,
        python_executable=python_executable,
        outputs_root=outputs_root,
        expected_instance_ids=expected_instance_ids,
        artifact_tag=artifact_tag,
        result_dir=persisted_result_dir,
        credential_path=credential_path,
        runner=runner,
    )
    per_instance_rows = _per_instance_rows(
        expected_ids=expected_ids,
        instance_scores=summary.get("instance_scores", {}),
        result_dir=persisted_result_dir,
    )
    summary["per_instance"] = per_instance_rows
    _write_per_instance_records(per_instance_eval_path_for(run_paths), per_instance_rows)
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
    """Parse the evaluator stdout into stable benchmark summary fields."""

    final_match = FINAL_SCORE_RE.search(stdout)
    real_match = REAL_SCORE_RE.search(stdout)

    summary: dict[str, Any] = {
        "attempted_tasks": 0,
        "correct_tasks": 0,
        "attempted_score": 0.0,
        "benchmark_total": BENCHMARK_TOTAL,
        "benchmark_score": 0.0,
        "instance_scores": _parse_instance_scores(stdout),
    }

    if final_match:
        summary["attempted_tasks"] = int(final_match.group(3))
        summary["correct_tasks"] = int(final_match.group(2))
        summary["attempted_score"] = float(final_match.group(1))

    if real_match:
        summary["benchmark_score"] = float(real_match.group(1))
    elif final_match:
        summary["benchmark_score"] = int(final_match.group(2)) / BENCHMARK_TOTAL

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


def _per_instance_rows(
    *,
    expected_ids: set[str],
    instance_scores: dict[str, int],
    result_dir: Path,
) -> list[dict[str, Any]]:
    """Build stable per-task eval records with a failure reason when available."""

    present_ids = {path.stem for path in result_dir.glob("*.csv")}
    rows: list[dict[str, Any]] = []
    for instance_id in sorted(expected_ids):
        score = instance_scores.get(instance_id)
        csv_present = instance_id in present_ids
        if csv_present:
            if score == 1:
                failure_reason = None
            elif score is not None:
                failure_reason = "official_fail"
            else:
                failure_reason = "eval_failed"
        else:
            failure_reason = "missing_csv"
        rows.append(
            {
                "instance_id": instance_id,
                "score": score,
                "passed": score == 1,
                "csv_present": csv_present,
                "failure_reason": failure_reason,
            }
        )
    return rows


def _missing_instance_ids(result_dir: Path, *, expected_ids: set[str]) -> list[str]:
    """List task IDs that do not have a CSV in the run output."""

    present_ids = {path.stem for path in result_dir.glob("*.csv")}
    return sorted(expected_ids - present_ids)


def _missing_csv_count(result_dir: Path, *, expected_ids: set[str]) -> int:
    """Count how many tasks still do not have a CSV output."""

    return len(_missing_instance_ids(result_dir, expected_ids=expected_ids))


def _copy_scored_csvs(source_dir: Path, destination_dir: Path) -> int:
    """Copy CSV outputs into the durable eval input directory."""

    destination_dir.mkdir(parents=True, exist_ok=True)
    for stale_csv in destination_dir.glob("*.csv"):
        stale_csv.unlink()
    copied = 0
    for source in sorted(source_dir.glob("*.csv")):
        copy2(source, destination_dir / source.name)
        copied += 1
    return copied


def _build_no_csv_summary(
    *,
    run_id: str,
    run_paths: RunPaths,
    expected_ids: set[str],
    result_dir: Path,
) -> dict[str, Any]:
    """Create a local summary when there are no scored CSVs to evaluate."""

    stdout_path = run_paths.eval_dir / "official_stdout.txt"
    stderr_path = run_paths.eval_dir / "official_stderr.txt"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    per_instance = _per_instance_rows(
        expected_ids=expected_ids,
        instance_scores={},
        result_dir=result_dir,
    )
    summary = {
        "attempted_tasks": 0,
        "correct_tasks": 0,
        "attempted_score": 0.0,
        "benchmark_total": BENCHMARK_TOTAL,
        "benchmark_score": 0.0,
        "instance_scores": {},
        "run_id": run_id,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "result_dir": str(result_dir),
        "per_instance": per_instance,
        "missing_csv_count": len(expected_ids),
        "missing_instance_ids": sorted(expected_ids),
        "returncode": 0,
    }
    summary_path = run_paths.eval_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _write_per_instance_records(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one JSONL row per expected task in stable order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _expected_instance_ids(run_paths: RunPaths) -> set[str]:
    """Use manifest task IDs when present, otherwise fall back to all tasks."""

    if run_paths.manifest_path.exists():
        manifest = json.loads(run_paths.manifest_path.read_text(encoding="utf-8"))
        task_ids = manifest.get("task_ids")
        if isinstance(task_ids, list) and all(isinstance(item, str) for item in task_ids):
            return set(task_ids)
    return {task.instance_id for task in load_tasks()}


def _write_eval_credential(destination: Path, *, credential_path: Path | None) -> None:
    """Render the evaluator's local credential file without mutating vendored files."""

    payload = _eval_credential_payload(credential_path)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _eval_credential_payload(credential_path: Path | None) -> dict[str, Any]:
    """Return connector-compatible Snowflake credentials for the official evaluator."""

    source_path = credential_path or DEFAULT_CREDENTIAL_PATH
    if source_path.exists():
        raw_payload = json.loads(source_path.read_text(encoding="utf-8"))
    else:
        raw_payload = {}

    username = raw_payload.get("user") or raw_payload.get("username") or "<your_username>"
    payload: dict[str, Any] = {
        "user": username,
        "password": raw_payload.get("password") or "<your_generated_token>",
        "account": raw_payload.get("account") or "RSRSBDK-YDB67606",
        "role": raw_payload.get("role") or "PARTICIPANT",
        "warehouse": raw_payload.get("warehouse") or "COMPUTE_WH_PARTICIPANT",
    }
    session_parameters = raw_payload.get("session_parameters")
    if isinstance(session_parameters, dict):
        payload["session_parameters"] = session_parameters
    return payload


def _run_subprocess(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the evaluator subprocess and return its captured stdout."""

    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
