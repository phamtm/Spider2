"""Analyze one run's traces and eval summary into concise failure reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sol01.analysis.trace_diagnostics import summarize_failed_question
from sol01.infra.logging import get_logger
from sol01.loading.category_metadata import (
    CATEGORY_BATCHES_DIR,
    SPIDER2_SNOW_PATH,
    CategoryMetadata,
    load_category_metadata,
)
from sol01.loading.tasks import REPO_ROOT
from sol01.output.output import RunPaths, ensure_run_paths

OUTPUTS_ROOT = REPO_ROOT / "methods" / "sol01" / "outputs"
FAILURE_CATEGORIES = (
    "validation",
    "execution",
    "empty_result",
    "critic",
    "missing_csv",
    "retrieval_miss",
)
FAILURE_HINTS = ("possible_aggregation_issue", "possible_date_filter_issue")
logger = get_logger(__name__)


def analyze_run(
    run_id: str,
    *,
    outputs_root: Path = OUTPUTS_ROOT,
    dataset_path: Path = SPIDER2_SNOW_PATH,
    batch_dir: Path = CATEGORY_BATCHES_DIR,
) -> dict[str, Any]:
    """Read one run's artifacts and write stable analysis outputs."""

    run_paths = ensure_run_paths(run_id, outputs_root=outputs_root)
    logger.info("analysis start", run_id=run_id, outputs_root=str(outputs_root))
    traces = _load_traces(run_paths)
    eval_summary = _load_optional_json(run_paths.eval_dir / "summary.json")
    trace_index = {trace["instance_id"]: trace for trace in traces}
    category_records = load_category_metadata(dataset_path=dataset_path, batch_dir=batch_dir)
    category_map = {record.instance_id: record for record in category_records}
    result_rows, category_rows = _analysis_result_rows(
        run_paths,
        traces,
        trace_index,
        eval_summary,
    )
    category_metadata = _category_metadata_snapshot(
        dataset_path=dataset_path,
        batch_dir=batch_dir,
        records=category_records,
    )
    category_coverage = _category_result_coverage(result_rows, category_map)

    by_category = {category: [] for category in FAILURE_CATEGORIES}
    for trace in traces:
        classification = _trace_classification(trace)
        hints = _trace_hints(trace)
        for category, evidence in classification.items():
            by_category[category].append(_trace_record(trace, evidence=evidence, hints=hints))

    for instance_id in _missing_csv_ids(run_paths, trace_index, eval_summary):
        trace = trace_index.get(instance_id)
        by_category["missing_csv"].append(
            _trace_record(
                trace,
                instance_id=instance_id,
                evidence=["CSV output was missing from eval summary or run artifacts."],
            )
        )

    for records in by_category.values():
        records.sort(key=lambda record: record["instance_id"])

    failed_questions = [
        _failed_question_record(trace)
        for trace in traces
        if str(trace.get("status") or "") == "failed"
    ]

    report = {
        "run_id": run_id,
        "trace_count": len(traces),
        "status_counts": _status_counts(traces),
        "eval_summary": eval_summary,
        "category_counts": {category: len(records) for category, records in by_category.items()},
        "category_metadata": category_metadata,
        "category_rows": category_rows,
        "category_coverage": category_coverage,
        "by_category": by_category,
        "by_database": _database_summary(traces, by_category),
        "by_primary_tier": _category_result_summary(result_rows, category_map, kind="tier"),
        "by_tag": _category_result_summary(result_rows, category_map, kind="tag"),
        "failed_questions": failed_questions,
    }

    failures_path = run_paths.analysis_dir / "failures.json"
    failures_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_path = run_paths.analysis_dir / "summary.md"
    summary_path.write_text(_render_summary(report), encoding="utf-8")
    logger.info(
        "analysis complete",
        run_id=run_id,
        trace_count=len(traces),
        summary_path=str(summary_path),
        failures_path=str(failures_path),
    )
    return report


def _load_traces(run_paths: RunPaths) -> list[dict[str, Any]]:
    """Load all per-task traces in stable instance-id order."""

    traces = []
    for path in sorted(run_paths.traces_dir.glob("*.json")):
        traces.append(json.loads(path.read_text(encoding="utf-8")))
    return traces


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    """Read one JSON file when it exists."""

    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_record(
    trace: dict[str, Any] | None,
    *,
    instance_id: str | None = None,
    evidence: list[str] | None = None,
    hints: list[str] | None = None,
) -> dict[str, Any]:
    """Keep only the fields that make failure reports easy to scan."""

    if trace is None:
        return {
            "instance_id": instance_id,
            "db": None,
            "status": "missing_trace",
            "question": None,
            "evidence": evidence or [],
            "hints": hints or [],
        }
    return {
        "instance_id": trace.get("instance_id", instance_id),
        "db": trace.get("db"),
        "status": trace.get("status"),
        "question": trace.get("question"),
        "evidence": evidence or [],
        "hints": hints or [],
    }


def _trace_classification(trace: dict[str, Any]) -> dict[str, list[str]]:
    """Classify one trace into conservative failure buckets with evidence."""

    categories: dict[str, list[str]] = {}
    status = str(trace.get("status") or "")
    attempts = trace.get("attempts") or []
    final_attempt = attempts[-1] if attempts else {}
    final_validation = final_attempt.get("validation") or {}
    final_execution = final_attempt.get("execution_result") or {}
    final_trace_execution = trace.get("final_execution") or {}

    if status != "success" and attempts:
        if not final_validation.get("ok", False):
            categories["validation"] = _validation_evidence(final_validation)
        elif not final_execution.get("ok", False):
            categories["execution"] = _execution_evidence(final_execution)

    is_failed = status == "failed"

    if _is_empty_result(trace):
        categories["empty_result"] = ["Final execution succeeded with zero rows."]
    if is_failed and _has_critic_signal(trace):
        categories["critic"] = _critic_evidence(trace)
    retrieval_evidence = _retrieval_miss_evidence(trace, final_attempt) if is_failed else []
    if retrieval_evidence:
        categories["retrieval_miss"] = retrieval_evidence

    if status == "success" and final_trace_execution and not final_trace_execution.get("ok", True):
        categories["execution"] = _execution_evidence(final_trace_execution)

    return categories


def _validation_evidence(validation: dict[str, Any]) -> list[str]:
    """Keep validation evidence short enough for reports."""

    errors = [str(error) for error in validation.get("errors", []) if error]
    return errors[:3] or ["Final validation failed."]


def _execution_evidence(execution: dict[str, Any]) -> list[str]:
    """Keep execution evidence short enough for reports."""

    error = execution.get("error")
    return [str(error)] if error else ["Final execution failed."]


def _critic_evidence(trace: dict[str, Any]) -> list[str]:
    """Collect the critic issues that caused repair or failure."""

    evidence = []
    for attempt in trace.get("attempts") or []:
        critic = attempt.get("critic") or {}
        if critic.get("should_repair"):
            evidence.append("Critic requested a repair.")
        evidence.extend(str(issue) for issue in critic.get("issues", []) if issue)
    return evidence[:3] or ["Critic reported an issue."]


def _is_empty_result(trace: dict[str, Any]) -> bool:
    """Treat successful zero-row outputs as a distinct result issue."""

    final_execution = trace.get("final_execution") or {}
    if trace.get("status") != "success":
        return False
    return int(final_execution.get("row_count") or 0) == 0


def _has_critic_signal(trace: dict[str, Any]) -> bool:
    """Mark traces where the critic found issues or forced a repair."""

    for attempt in trace.get("attempts") or []:
        critic = attempt.get("critic") or {}
        if critic.get("should_repair") or critic.get("issues"):
            return True
    return False


def _retrieval_miss_evidence(
    trace: dict[str, Any],
    final_attempt: dict[str, Any],
) -> list[str]:
    """Report retrieval misses only when table evidence points there."""

    schema = trace.get("schema_selection") or {}
    validation = final_attempt.get("validation") or {}
    execution = final_attempt.get("execution_result") or {}
    error_text = " ".join(
        [
            str(item)
            for item in [
                *validation.get("errors", []),
                execution.get("error"),
            ]
            if item
        ]
    ).lower()
    expanded_tables = schema.get("expanded_tables") or []
    has_table_error = "not allowed" in error_text and "table" in error_text

    evidence = []
    if "schema_selection" not in trace:
        return evidence
    if not expanded_tables and has_table_error:
        evidence.append(
            "Schema selection expanded no tables and validation reported a table error."
        )
    elif has_table_error:
        evidence.append("Validation reported a table outside the selected schema.")
    return evidence


def _trace_hints(trace: dict[str, Any]) -> list[str]:
    """Attach weak signals as hints, never as root-cause categories."""

    status = str(trace.get("status") or "")
    if status != "failed":
        return []
    attempts = trace.get("attempts") or []
    final_attempt = attempts[-1] if attempts else {}
    text = _analysis_text(trace, final_attempt)
    hints = []
    if any(keyword in text for keyword in ("group by", "aggregate", "aggregation", "sum(", "avg(")):
        hints.append("possible_aggregation_issue")
    if any(
        keyword in text for keyword in ("date", "timestamp", "month", "year", "quarter", "between")
    ):
        hints.append("possible_date_filter_issue")
    return [hint for hint in FAILURE_HINTS if hint in hints]


def _failed_question_record(trace: dict[str, Any]) -> dict[str, Any]:
    """Summarize the verification signals for one failed question."""

    summary = summarize_failed_question(trace)
    return {
        "instance_id": summary.get("instance_id"),
        "db": trace.get("db"),
        "question": trace.get("question"),
        "status": summary.get("status"),
        "verification_checks": summary.get("verification_checks", []),
        "failed_checks": summary.get("failed_checks", []),
        "repair_triggers": summary.get("repair_triggers", []),
        "ranking_reasons": summary.get("ranking_reasons", []),
        "diagnostics": summary.get("diagnostics"),
    }


def _analysis_text(trace: dict[str, Any], final_attempt: dict[str, Any]) -> str:
    """Collect plain text signals used by the lightweight heuristics."""

    critic = final_attempt.get("critic") or {}
    validation = final_attempt.get("validation") or {}
    execution = final_attempt.get("execution_result") or {}
    parts = [
        trace.get("question"),
        final_attempt.get("sql"),
        final_attempt.get("explanation"),
        " ".join(validation.get("errors", [])),
        execution.get("error"),
        " ".join(critic.get("issues", [])),
        critic.get("repair_focus"),
    ]
    return " ".join(str(part) for part in parts if part).lower()


def _analysis_result_rows(
    run_paths: RunPaths,
    traces: list[dict[str, Any]],
    trace_index: dict[str, dict[str, Any]],
    eval_summary: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return category rows and the source they came from."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_instance = eval_summary.get("per_instance") if isinstance(eval_summary, dict) else None
    if isinstance(per_instance, list):
        source = "eval_summary.per_instance"
        for row in per_instance:
            if not isinstance(row, dict):
                continue
            instance_id = row.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id or instance_id in seen:
                continue
            rows.append(
                {
                    "instance_id": instance_id,
                    "passed": bool(row.get("passed")),
                    "failure_reason": row.get("failure_reason"),
                }
            )
            seen.add(instance_id)
    else:
        source = "trace_fallback"
        for trace in traces:
            instance_id = str(trace.get("instance_id") or "")
            if not instance_id or instance_id in seen:
                continue
            rows.append(
                {
                    "instance_id": instance_id,
                    "passed": str(trace.get("status") or "") == "success",
                    "failure_reason": None,
                }
            )
            seen.add(instance_id)

    source_row_count = len(rows)
    for instance_id in _missing_csv_ids(run_paths, trace_index, eval_summary):
        if instance_id in seen:
            continue
        rows.append(
            {
                "instance_id": instance_id,
                "passed": False,
                "failure_reason": "missing_csv",
            }
        )
        seen.add(instance_id)

    source_detail = (
        "eval summary per_instance" if source == "eval_summary.per_instance" else "trace fallback"
    )
    return rows, {
        "source": source,
        "source_detail": source_detail,
        "source_row_count": source_row_count,
    }


def _category_metadata_snapshot(
    *,
    dataset_path: Path,
    batch_dir: Path,
    records: list[CategoryMetadata],
) -> dict[str, Any]:
    """Summarize the exact category metadata used for one analysis run."""

    payload = [record.model_dump(exclude_none=True) for record in records]
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "dataset_path": str(dataset_path),
        "batch_dir": str(batch_dir),
        "record_count": len(records),
        "sha256": fingerprint,
    }


def _category_result_summary(
    result_rows: list[dict[str, Any]],
    category_map: dict[str, Any],
    *,
    kind: str,
) -> dict[Any, dict[str, Any]]:
    """Summarize pass/fail counts by tier or tag."""

    buckets: dict[Any, dict[str, Any]] = {}
    for row in result_rows:
        instance_id = str(row.get("instance_id") or "")
        metadata = category_map.get(instance_id)
        if metadata is None:
            continue

        keys = [metadata.primary_tier] if kind == "tier" else list(metadata.tags)
        for key in keys:
            bucket = buckets.setdefault(
                key,
                {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "instance_ids": [],
                },
            )
            bucket["total"] += 1
            if row.get("passed"):
                bucket["passed"] += 1
            else:
                bucket["failed"] += 1
            bucket["instance_ids"].append(instance_id)

    if kind == "tier":
        return dict(sorted(buckets.items(), key=lambda item: item[0]))
    return dict(
        sorted(
            buckets.items(),
            key=lambda item: (-item[1]["failed"], -item[1]["passed"], item[0]),
        )
    )


def _category_result_coverage(
    result_rows: list[dict[str, Any]],
    category_map: dict[str, Any],
) -> dict[str, int]:
    """Count how many category rows are mapped versus unmapped."""

    mapped = 0
    unmapped = 0
    for row in result_rows:
        instance_id = str(row.get("instance_id") or "")
        if instance_id and instance_id in category_map:
            mapped += 1
        else:
            unmapped += 1
    return {"mapped": mapped, "unmapped": unmapped, "total": mapped + unmapped}


def _missing_csv_ids(
    run_paths: RunPaths,
    trace_index: dict[str, dict[str, Any]],
    eval_summary: dict[str, Any] | None,
) -> set[str]:
    """Combine eval summary output with local trace checks for missing CSVs."""

    instance_ids = set()
    if eval_summary is not None and isinstance(eval_summary.get("missing_instance_ids"), list):
        for item in eval_summary.get("missing_instance_ids", []):
            if isinstance(item, str):
                instance_ids.add(item)
    else:
        for instance_id in _manifest_task_ids(run_paths):
            if not (run_paths.csv_dir / f"{instance_id}.csv").exists():
                instance_ids.add(instance_id)

    for instance_id, trace in trace_index.items():
        if trace.get("status") != "success":
            continue
        csv_path = trace.get("csv_path")
        if not csv_path:
            instance_ids.add(instance_id)
            continue
        if not Path(csv_path).exists():
            instance_ids.add(instance_id)

    return instance_ids


def _manifest_task_ids(run_paths: RunPaths) -> set[str]:
    """Read the run manifest task IDs when they are available."""

    manifest = _load_optional_json(run_paths.manifest_path)
    if manifest is None:
        return set()
    task_ids = manifest.get("task_ids")
    if not isinstance(task_ids, list):
        return set()
    if not all(isinstance(item, str) for item in task_ids):
        return set()
    return set(task_ids)


def _status_counts(traces: list[dict[str, Any]]) -> dict[str, int]:
    """Count how many traces ended in each status."""

    counts = {"success": 0, "failed": 0, "skipped": 0}
    for trace in traces:
        status = str(trace.get("status") or "")
        if status in counts:
            counts[status] += 1
    return counts


def _database_summary(
    traces: list[dict[str, Any]],
    by_category: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Summarize totals and failure buckets by database."""

    summary: dict[str, dict[str, Any]] = {}
    for trace in traces:
        db = str(trace.get("db") or "unknown")
        bucket = summary.setdefault(
            db,
            {
                "total": 0,
                "success": 0,
                "failed": 0,
                "skipped": 0,
                "category_counts": {category: 0 for category in FAILURE_CATEGORIES},
            },
        )
        bucket["total"] += 1
        status = str(trace.get("status") or "")
        if status in ("success", "failed", "skipped"):
            bucket[status] += 1

    for category, records in by_category.items():
        for record in records:
            db = str(record.get("db") or "unknown")
            bucket = summary.setdefault(
                db,
                {
                    "total": 0,
                    "success": 0,
                    "failed": 0,
                    "skipped": 0,
                    "category_counts": {name: 0 for name in FAILURE_CATEGORIES},
                },
            )
            bucket["category_counts"][category] += 1

    return dict(sorted(summary.items()))


def _render_summary(report: dict[str, Any]) -> str:
    """Write a short markdown summary for quick run triage."""

    eval_summary = report.get("eval_summary") or {}
    lines = [f"# Analysis for {report['run_id']}", ""]
    lines.append(
        "- Traces: "
        f"{report['trace_count']} "
        f"(success {report['status_counts']['success']}, "
        f"failed {report['status_counts']['failed']}, "
        f"skipped {report['status_counts']['skipped']})"
    )

    if eval_summary:
        lines.append(
            "- Eval: "
            f"{eval_summary.get('correct_tasks', 0)}/"
            f"{eval_summary.get('attempted_tasks', 0)} correct, "
            f"missing CSV {eval_summary.get('missing_csv_count', 0)}"
        )

    lines.append("")
    lines.append("## Failure Buckets")
    for category in FAILURE_CATEGORIES:
        records = report["by_category"][category]
        if not records:
            continue
        instance_ids = ", ".join(
            record["instance_id"] for record in records if record["instance_id"]
        )
        lines.append(f"- {category}: {len(records)} ({instance_ids})")

    failed_questions = report.get("failed_questions") or []
    if failed_questions:
        lines.append("")
        lines.append("## Failed Questions")
        for row in failed_questions:
            summary_bits = []
            if row.get("verification_checks"):
                summary_bits.append(f"checks: {', '.join(row['verification_checks'])}")
            if row.get("failed_checks"):
                summary_bits.append(f"failed: {', '.join(row['failed_checks'])}")
            if row.get("repair_triggers"):
                summary_bits.append(f"repair: {', '.join(row['repair_triggers'])}")
            if row.get("ranking_reasons"):
                summary_bits.append(f"ranking: {', '.join(row['ranking_reasons'])}")
            if row.get("diagnostics"):
                summary_bits.append(str(row["diagnostics"]))
            lines.append(
                f"- {row['instance_id']}: "
                + "; ".join(summary_bits if summary_bits else ["no diagnostics available"])
            )

    lines.append("")
    lines.append("## Category Coverage")
    rows = report["category_rows"]
    coverage = report["category_coverage"]
    lines.append(f"- source: {rows['source_detail']} ({rows['source_row_count']} rows)")
    lines.append(
        f"- mapped: {coverage['mapped']}, "
        f"unmapped: {coverage['unmapped']}, total {coverage['total']}"
    )

    lines.append("")
    lines.append("## Category Metadata")
    metadata = report["category_metadata"]
    lines.append(f"- dataset: {metadata['dataset_path']}")
    lines.append(f"- batches: {metadata['batch_dir']}")
    lines.append(f"- records: {metadata['record_count']}")
    lines.append(f"- sha256: {metadata['sha256']}")

    lines.append("")
    lines.append("## By Database")
    for db, bucket in report["by_database"].items():
        lines.append(
            f"- {db}: total {bucket['total']}, "
            f"success {bucket['success']}, failed {bucket['failed']}, skipped {bucket['skipped']}"
        )

    lines.append("")
    lines.append("## By Primary Tier")
    for tier, bucket in report["by_primary_tier"].items():
        lines.append(
            f"- tier {tier}: passed {bucket['passed']}, "
            f"failed {bucket['failed']}, total {bucket['total']}"
        )

    lines.append("")
    lines.append("## By Tag")
    lines.append("- Tag buckets overlap; one task can appear in multiple tags.")
    for tag, bucket in report["by_tag"].items():
        lines.append(
            f"- {tag}: passed {bucket['passed']}, "
            f"failed {bucket['failed']}, total {bucket['total']}"
        )

    lines.append("")
    return "\n".join(lines)
