"""Run the official Spider2-snow evaluator and save a summary."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from shutil import copy2, rmtree
from typing import Any

from sol01.execution.snowflake_runner import DEFAULT_CREDENTIAL_PATH
from sol01.infra.logging import get_logger
from sol01.infra.policy import DEFAULT_EVAL_DATASET_POLICY
from sol01.loading.tasks import REPO_ROOT, load_tasks
from sol01.output.output import (
    RunPaths,
    ensure_run_paths,
    eval_command_path_for,
    eval_credential_path_for,
    eval_input_csv_dir_for,
    eval_log_path_for,
    eval_metadata_jsonl_path_for,
    eval_per_instance_path_for,
    eval_stderr_path_for,
    eval_stdout_path_for,
    eval_summary_path_for,
    eval_temp_dir_for,
    eval_workspace_suite_dir_for,
    per_instance_eval_path_for,
    scored_csv_dir_for,
)

EVALUATION_SUITE_DIR = REPO_ROOT / "spider2-snow" / "evaluation_suite"
EVALUATE_SCRIPT = EVALUATION_SUITE_DIR / "evaluate.py"
GOLD_DIR = EVALUATION_SUITE_DIR / "gold"
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
    eval_id = _eval_id_for(artifact_tag)
    scored_result_dir = result_dir or run_paths.csv_dir
    eval_input_dir = eval_input_csv_dir_for(run_paths, eval_id=eval_id)
    eval_cwd = eval_workspace_suite_dir_for(run_paths, eval_id=eval_id)
    eval_temp_dir = eval_temp_dir_for(run_paths, eval_id=eval_id)
    eval_metadata_jsonl_path = eval_metadata_jsonl_path_for(run_paths, eval_id=eval_id)
    eval_credential_path = eval_credential_path_for(run_paths, eval_id=eval_id)
    eval_command_path = eval_command_path_for(run_paths, eval_id=eval_id)
    eval_stdout_path = eval_stdout_path_for(run_paths, eval_id=eval_id)
    eval_stderr_path = eval_stderr_path_for(run_paths, eval_id=eval_id)
    eval_summary_path = eval_summary_path_for(run_paths, eval_id=eval_id)
    eval_per_instance_path = eval_per_instance_path_for(run_paths, eval_id=eval_id)
    eval_log_path = eval_log_path_for(run_paths, eval_id=eval_id)
    _prepare_eval_workspace(
        eval_cwd=eval_cwd,
        eval_metadata_jsonl_path=eval_metadata_jsonl_path,
        eval_temp_dir=eval_temp_dir,
        credential_path=credential_path,
        eval_credential_path=eval_credential_path,
    )
    copied_csv_count = _copy_scored_csvs(scored_result_dir, eval_input_dir)
    logger.info(
        "evaluation start",
        run_id=run_id,
        result_dir=str(eval_input_dir),
        expected_instance_count=len(expected_instance_ids) if expected_instance_ids else None,
        artifact_tag=artifact_tag,
    )
    command = build_eval_command(
        run_paths,
        python_executable=python_executable,
        result_dir=eval_input_dir,
        temp_dir=eval_temp_dir,
    )
    started_at = datetime.now(UTC).isoformat()
    _write_json(
        eval_command_path,
        {
            "argv": command,
            "credential_staged_path": str(eval_credential_path),
            "cwd": str(eval_cwd),
            "eval_id": eval_id,
            "metadata_jsonl_path": str(eval_metadata_jsonl_path),
            "result_dir": str(eval_input_dir),
            "run_id": run_id,
            "started_at": started_at,
            "temp_dir": str(eval_temp_dir),
        },
    )
    completed = (runner or _run_subprocess)(command, cwd=eval_cwd)
    completed_at = datetime.now(UTC).isoformat()

    stdout_text = completed.stdout or ""
    stderr_text = completed.stderr or ""
    _write_eval_outputs(
        run_paths,
        artifact_tag=artifact_tag,
        eval_stdout_path=eval_stdout_path,
        eval_stderr_path=eval_stderr_path,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
    )

    # Some injected test runners may return None instead of an empty string.
    expected_ids = (
        set(expected_instance_ids)
        if expected_instance_ids is not None
        else _expected_instance_ids(run_paths)
    )
    summary = parse_eval_stdout(
        stdout_text,
        expected_task_count=len(expected_ids) if expected_ids else None,
    )
    summary["run_id"] = run_id
    summary["eval_id"] = eval_id
    summary["cwd"] = str(eval_cwd)
    summary["command_path"] = str(eval_command_path)
    summary["result_dir"] = str(eval_input_dir)
    summary["temp_dir"] = str(eval_temp_dir)
    summary["metadata_jsonl_path"] = str(eval_metadata_jsonl_path)
    summary["credential_staged_path"] = str(eval_credential_path)
    summary["stdout_path"] = str(eval_stdout_path)
    summary["stderr_path"] = str(eval_stderr_path)
    summary["log_path"] = str(eval_log_path)
    summary["copied_csv_count"] = copied_csv_count
    summary["per_instance"] = _per_instance_summary(
        expected_ids=expected_ids,
        instance_scores=summary.get("instance_scores", {}),
        result_dir=eval_input_dir,
    )
    summary["missing_csv_count"] = _missing_csv_count(
        eval_input_dir,
        expected_ids=expected_ids,
    )
    summary["missing_instance_ids"] = _missing_instance_ids(
        eval_input_dir,
        expected_ids=expected_ids,
    )
    summary["returncode"] = completed.returncode

    _write_eval_artifacts(
        run_paths,
        artifact_tag=artifact_tag,
        eval_id=eval_id,
        eval_summary_path=eval_summary_path,
        eval_per_instance_path=eval_per_instance_path,
        summary=summary,
    )
    _write_json(
        eval_command_path,
        {
            "argv": command,
            "completed_at": completed_at,
            "credential_staged_path": str(eval_credential_path),
            "cwd": str(eval_cwd),
            "eval_id": eval_id,
            "log_path": str(eval_log_path),
            "metadata_jsonl_path": str(eval_metadata_jsonl_path),
            "result_dir": str(eval_input_dir),
            "returncode": completed.returncode,
            "run_id": run_id,
            "started_at": started_at,
            "stderr_path": str(eval_stderr_path),
            "stdout_path": str(eval_stdout_path),
            "temp_dir": str(eval_temp_dir),
        },
    )
    logger.info(
        "evaluation complete",
        run_id=run_id,
        returncode=completed.returncode,
        stdout_path=str(eval_stdout_path),
        stderr_path=str(eval_stderr_path),
        summary_path=str(eval_summary_path),
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
    eval_id = _eval_id_for(artifact_tag)
    eval_input_dir = eval_input_csv_dir_for(run_paths, eval_id=eval_id)
    eval_cwd = eval_workspace_suite_dir_for(run_paths, eval_id=eval_id)
    eval_temp_dir = eval_temp_dir_for(run_paths, eval_id=eval_id)
    eval_metadata_jsonl_path = eval_metadata_jsonl_path_for(run_paths, eval_id=eval_id)
    eval_credential_path = eval_credential_path_for(run_paths, eval_id=eval_id)
    eval_stdout_path = eval_stdout_path_for(run_paths, eval_id=eval_id)
    eval_stderr_path = eval_stderr_path_for(run_paths, eval_id=eval_id)
    eval_log_path = eval_log_path_for(run_paths, eval_id=eval_id)
    persisted_result_dir = scored_csv_dir_for(run_paths)
    copied_csv_count = _copy_scored_csvs(run_paths.csv_dir, persisted_result_dir)
    expected_ids = set(expected_instance_ids)

    if copied_csv_count == 0:
        eval_input_dir.mkdir(parents=True, exist_ok=True)
        _prepare_eval_workspace(
            eval_cwd=eval_cwd,
            eval_metadata_jsonl_path=eval_metadata_jsonl_path,
            eval_temp_dir=eval_temp_dir,
            credential_path=credential_path,
            eval_credential_path=eval_credential_path,
        )
        eval_log_path.write_text("", encoding="utf-8")
        _write_eval_outputs(
            run_paths,
            artifact_tag=artifact_tag,
            eval_stdout_path=eval_stdout_path,
            eval_stderr_path=eval_stderr_path,
            stdout_text="",
            stderr_text="",
        )
        summary = _build_no_csv_summary(
            run_id=run_id,
            eval_id=eval_id,
            run_paths=run_paths,
            expected_ids=expected_ids,
            result_dir=eval_input_dir,
            cwd=eval_cwd,
            temp_dir=eval_temp_dir,
            metadata_jsonl_path=eval_metadata_jsonl_path,
            credential_staged_path=eval_credential_path,
            stdout_path=eval_stdout_path,
            stderr_path=eval_stderr_path,
            log_path=eval_log_path,
        )
        _write_eval_artifacts(
            run_paths,
            artifact_tag=artifact_tag,
            eval_id=eval_id,
            eval_summary_path=eval_summary_path_for(run_paths, eval_id=eval_id),
            eval_per_instance_path=eval_per_instance_path_for(run_paths, eval_id=eval_id),
            summary=summary,
        )
        return summary

    eval_failed = False
    try:
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
    except subprocess.CalledProcessError as exc:
        eval_failed = True
        summary = _load_failed_eval_summary(
            run_id=run_id,
            run_paths=run_paths,
            result_dir=eval_input_dir,
            artifact_tag=artifact_tag,
            error=exc,
        )
    per_instance_rows = _per_instance_rows(
        expected_ids=expected_ids,
        instance_scores=summary.get("instance_scores", {}),
        result_dir=eval_input_dir,
        eval_failed=eval_failed,
    )
    summary["per_instance"] = per_instance_rows
    summary["missing_csv_count"] = _missing_csv_count(
        eval_input_dir,
        expected_ids=expected_ids,
    )
    summary["missing_instance_ids"] = _missing_instance_ids(
        eval_input_dir,
        expected_ids=expected_ids,
    )
    summary["result_dir"] = str(eval_input_dir)
    _write_eval_artifacts(
        run_paths,
        artifact_tag=artifact_tag,
        eval_id=eval_id,
        eval_summary_path=eval_summary_path_for(run_paths, eval_id=eval_id),
        eval_per_instance_path=eval_per_instance_path_for(run_paths, eval_id=eval_id),
        summary=summary,
    )
    return summary


def build_eval_command(
    run_paths: RunPaths,
    *,
    python_executable: str | None = None,
    result_dir: Path | None = None,
    temp_dir: Path | None = None,
) -> list[str]:
    """Build the official evaluator subprocess command."""

    command = [
        python_executable or sys.executable,
        str(EVALUATE_SCRIPT),
        "--result_dir",
        str(result_dir or run_paths.csv_dir),
        "--mode",
        "exec_result",
        "--gold_dir",
        str(GOLD_DIR),
    ]
    if temp_dir is not None:
        command.extend(["--temp_dir", str(temp_dir)])
    return command


def parse_eval_stdout(stdout: str, *, expected_task_count: int | None = None) -> dict[str, Any]:
    """Parse the evaluator stdout into stable benchmark summary fields."""

    final_match = FINAL_SCORE_RE.search(stdout)
    real_match = REAL_SCORE_RE.search(stdout)
    benchmark_total = _benchmark_total(
        expected_task_count=expected_task_count,
        final_match=final_match,
        real_match=real_match,
    )

    summary: dict[str, Any] = {
        "attempted_tasks": 0,
        "correct_tasks": 0,
        "attempted_score": 0.0,
        "benchmark_total": benchmark_total,
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
        summary["benchmark_score"] = int(final_match.group(2)) / benchmark_total

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
    eval_failed: bool = False,
) -> list[dict[str, Any]]:
    """Build stable per-task eval records with a failure reason when available."""

    present_ids = {path.stem for path in result_dir.glob("*.csv")}
    rows: list[dict[str, Any]] = []
    for instance_id in sorted(expected_ids):
        score = instance_scores.get(instance_id)
        csv_present = instance_id in present_ids
        if csv_present:
            if eval_failed:
                failure_reason = "eval_failed"
            elif score == 1:
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


def _load_failed_eval_summary(
    *,
    run_id: str,
    run_paths: RunPaths,
    result_dir: Path,
    artifact_tag: str | None,
    error: subprocess.CalledProcessError,
) -> dict[str, Any]:
    """Load the summary written before a failed official eval raised."""

    summary_path = _summary_path_for(run_paths, artifact_tag=artifact_tag)
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        eval_id = _eval_id_for(artifact_tag)
        expected_ids = _expected_instance_ids(run_paths)
        summary = parse_eval_stdout(
            error.output or "",
            expected_task_count=len(expected_ids) if expected_ids else None,
        )
        summary["run_id"] = run_id
        summary["eval_id"] = eval_id
        summary["cwd"] = str(eval_workspace_suite_dir_for(run_paths, eval_id=eval_id))
        summary["command_path"] = str(eval_command_path_for(run_paths, eval_id=eval_id))
        summary["temp_dir"] = str(eval_temp_dir_for(run_paths, eval_id=eval_id))
        summary["metadata_jsonl_path"] = str(
            eval_metadata_jsonl_path_for(run_paths, eval_id=eval_id)
        )
        summary["credential_staged_path"] = str(
            eval_credential_path_for(run_paths, eval_id=eval_id)
        )
        summary["stdout_path"] = str(eval_stdout_path_for(run_paths, eval_id=eval_id))
        summary["stderr_path"] = str(eval_stderr_path_for(run_paths, eval_id=eval_id))
        summary["log_path"] = str(eval_log_path_for(run_paths, eval_id=eval_id))
        summary["result_dir"] = str(result_dir)
    summary["returncode"] = error.returncode
    summary["eval_error"] = "official_eval_failed"
    summary["result_dir"] = str(result_dir)
    return summary


def _missing_instance_ids(result_dir: Path, *, expected_ids: set[str]) -> list[str]:
    """List task IDs that do not have a CSV in the run output."""

    present_ids = {path.stem for path in result_dir.glob("*.csv")}
    return sorted(expected_ids - present_ids)


def _missing_csv_count(result_dir: Path, *, expected_ids: set[str]) -> int:
    """Count how many tasks still do not have a CSV output."""

    return len(_missing_instance_ids(result_dir, expected_ids=expected_ids))


def _copy_scored_csvs(source_dir: Path, destination_dir: Path) -> int:
    """Copy CSV outputs into the durable eval input directory."""

    if source_dir.resolve() == destination_dir.resolve():
        return len(list(destination_dir.glob("*.csv")))
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
    eval_id: str,
    run_paths: RunPaths,
    expected_ids: set[str],
    result_dir: Path,
    cwd: Path,
    temp_dir: Path,
    metadata_jsonl_path: Path,
    credential_staged_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    """Create a local summary when there are no scored CSVs to evaluate."""

    per_instance = _per_instance_rows(
        expected_ids=expected_ids,
        instance_scores={},
        result_dir=result_dir,
    )
    summary = {
        "attempted_tasks": 0,
        "correct_tasks": 0,
        "attempted_score": 0.0,
        "benchmark_total": len(expected_ids) or DEFAULT_EVAL_DATASET_POLICY.default_task_count,
        "benchmark_score": 0.0,
        "cwd": str(cwd),
        "credential_staged_path": str(credential_staged_path),
        "eval_id": eval_id,
        "log_path": str(log_path),
        "metadata_jsonl_path": str(metadata_jsonl_path),
        "instance_scores": {},
        "run_id": run_id,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "result_dir": str(result_dir),
        "temp_dir": str(temp_dir),
        "per_instance": per_instance,
        "missing_csv_count": len(expected_ids),
        "missing_instance_ids": sorted(expected_ids),
        "returncode": 0,
    }
    return summary


def _official_stdout_path_for(run_paths: RunPaths, *, artifact_tag: str | None) -> Path:
    """Return the convenience stdout path for one eval invocation."""

    suffix = f".{artifact_tag}" if artifact_tag else ""
    return run_paths.eval_dir / f"official_stdout{suffix}.txt"


def _official_stderr_path_for(run_paths: RunPaths, *, artifact_tag: str | None) -> Path:
    """Return the convenience stderr path for one eval invocation."""

    suffix = f".{artifact_tag}" if artifact_tag else ""
    return run_paths.eval_dir / f"official_stderr{suffix}.txt"


def _eval_id_for(artifact_tag: str | None) -> str:
    """Use the artifact tag as the durable eval ID when one is supplied."""

    return artifact_tag or "default"


def _prepare_eval_workspace(
    *,
    eval_cwd: Path,
    eval_metadata_jsonl_path: Path,
    eval_temp_dir: Path,
    credential_path: Path | None,
    eval_credential_path: Path,
) -> None:
    """Create the durable workspace tree expected by the official evaluator."""

    eval_cwd.mkdir(parents=True, exist_ok=True)
    if eval_temp_dir.exists():
        rmtree(eval_temp_dir)
    eval_temp_dir.mkdir(parents=True, exist_ok=True)
    eval_metadata_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    copy2(REPO_ROOT / "spider2-snow" / "spider2-snow.jsonl", eval_metadata_jsonl_path)
    _write_eval_credential(
        eval_credential_path,
        credential_path=credential_path,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON artifact with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_eval_outputs(
    run_paths: RunPaths,
    *,
    artifact_tag: str | None,
    eval_stdout_path: Path,
    eval_stderr_path: Path,
    stdout_text: str,
    stderr_text: str,
) -> None:
    """Persist stdout and stderr in both durable and convenience locations."""

    eval_stdout_path.write_text(stdout_text, encoding="utf-8")
    eval_stderr_path.write_text(stderr_text, encoding="utf-8")
    _official_stdout_path_for(run_paths, artifact_tag=artifact_tag).write_text(
        stdout_text,
        encoding="utf-8",
    )
    _official_stderr_path_for(run_paths, artifact_tag=artifact_tag).write_text(
        stderr_text,
        encoding="utf-8",
    )


def _write_eval_artifacts(
    run_paths: RunPaths,
    *,
    artifact_tag: str | None,
    eval_id: str,
    eval_summary_path: Path,
    eval_per_instance_path: Path,
    summary: dict[str, Any],
) -> None:
    """Persist the final summary in both durable and convenience locations."""

    summary_payload = dict(summary)
    summary_payload["eval_id"] = eval_id
    _write_eval_summary(eval_summary_path, summary_payload)
    _write_eval_summary(_summary_path_for(run_paths, artifact_tag=artifact_tag), summary_payload)
    _write_per_instance_records(eval_per_instance_path, summary_payload["per_instance"])
    _write_per_instance_records(
        per_instance_eval_path_for(run_paths), summary_payload["per_instance"]
    )


def _summary_path_for(run_paths: RunPaths, *, artifact_tag: str | None) -> Path:
    """Return the saved eval summary path for one artifact tag."""

    suffix = f".{artifact_tag}" if artifact_tag else ""
    return run_paths.eval_dir / f"summary{suffix}.json"


def _write_eval_summary(path: Path, summary: dict[str, Any]) -> None:
    """Write an eval summary with stable formatting."""

    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _benchmark_total(
    *,
    expected_task_count: int | None,
    final_match: re.Match[str] | None,
    real_match: re.Match[str] | None,
) -> int:
    """Resolve the benchmark denominator from the active eval slice when possible."""

    if real_match:
        reported_total = int(real_match.group(3))
        if reported_total > 0:
            return reported_total
    if expected_task_count:
        return expected_task_count
    if final_match:
        reported_total = int(final_match.group(3))
        if reported_total > 0:
            return reported_total
    return DEFAULT_EVAL_DATASET_POLICY.default_task_count


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
