"""Analyze one run's traces and eval summary into concise failure reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sol01.config import RuntimeConfig
from sol01.coordinator import run_task
from sol01.logging import get_logger
from sol01.models import RetrievalMode, Task
from sol01.output import RunPaths, ensure_run_paths
from sol01.tasks import REPO_ROOT

OUTPUTS_ROOT = REPO_ROOT / "methods" / "sol01" / "outputs"
DEFAULT_RETRIEVAL_FIXTURE = (
    REPO_ROOT / "methods" / "sol01" / "tests" / "fixtures" / "retrieval_cases.json"
)
FAILURE_CATEGORIES = (
    "validation",
    "execution",
    "empty_result",
    "critic",
    "missing_csv",
    "retrieval_miss",
    "aggregation_issue",
    "date_filter_issue",
)
logger = get_logger(__name__)


def analyze_run(
    run_id: str,
    *,
    outputs_root: Path = OUTPUTS_ROOT,
) -> dict[str, Any]:
    """Read one run's artifacts and write stable analysis outputs."""

    run_paths = ensure_run_paths(run_id, outputs_root=outputs_root)
    logger.info("analysis start", run_id=run_id, outputs_root=str(outputs_root))
    traces = _load_traces(run_paths)
    eval_summary = _load_optional_json(run_paths.eval_dir / "summary.json")
    trace_index = {trace["instance_id"]: trace for trace in traces}

    by_category = {category: [] for category in FAILURE_CATEGORIES}
    for trace in traces:
        record = _trace_record(trace)
        for category in _trace_categories(trace):
            by_category[category].append(record)

    for instance_id in _missing_csv_ids(run_paths, trace_index, eval_summary):
        trace = trace_index.get(instance_id)
        by_category["missing_csv"].append(_trace_record(trace, instance_id=instance_id))

    for records in by_category.values():
        records.sort(key=lambda record: record["instance_id"])

    report = {
        "run_id": run_id,
        "trace_count": len(traces),
        "status_counts": _status_counts(traces),
        "eval_summary": eval_summary,
        "category_counts": {category: len(records) for category, records in by_category.items()},
        "by_category": by_category,
        "by_database": _database_summary(traces, by_category),
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


def compare_retrieval_modes(
    run_id: str,
    *,
    config: RuntimeConfig,
    fixture_path: Path = DEFAULT_RETRIEVAL_FIXTURE,
    outputs_root: Path = OUTPUTS_ROOT,
) -> dict[str, Any]:
    """Run the lexical and LLM-only retrieval paths over one fixed case set."""

    run_paths = ensure_run_paths(run_id, outputs_root=outputs_root)
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    cases = fixture.get("cases", [])
    internal_root = run_paths.analysis_dir / "_retrieval_compare"

    results = []
    for case in cases:
        task = Task(
            instance_id=case["instance_id"],
            db=case["db"],
            question=case["question"],
            external_knowledge=case.get("external_knowledge"),
        )
        expected_tables = list(case.get("expected_tables", []))
        mode_results = {
            mode: _run_retrieval_mode_case(
                task,
                expected_tables=expected_tables,
                mode=mode,
                config=config,
                outputs_root=internal_root,
            )
            for mode in ("lexical", "llm_only")
        }
        results.append(
            {
                "instance_id": task.instance_id,
                "db": task.db,
                "question": task.question,
                "expected_tables": expected_tables,
                "modes": mode_results,
            }
        )

    report = {
        "run_id": run_id,
        "fixture_path": str(fixture_path),
        "case_count": len(results),
        "cases": results,
        "summary": {
            "lexical": _mode_summary(results, "lexical"),
            "llm_only": _mode_summary(results, "llm_only"),
        },
    }

    compare_json = run_paths.analysis_dir / "retrieval_compare.json"
    compare_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compare_md = run_paths.analysis_dir / "retrieval_compare.md"
    compare_md.write_text(_render_retrieval_compare(report), encoding="utf-8")
    logger.info(
        "retrieval compare complete",
        run_id=run_id,
        case_count=len(results),
        compare_json=str(compare_json),
        compare_md=str(compare_md),
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
) -> dict[str, Any]:
    """Keep only the fields that make failure reports easy to scan."""

    if trace is None:
        return {
            "instance_id": instance_id,
            "db": None,
            "status": "missing_trace",
            "question": None,
        }
    return {
        "instance_id": trace.get("instance_id", instance_id),
        "db": trace.get("db"),
        "status": trace.get("status"),
        "question": trace.get("question"),
    }


def _trace_categories(trace: dict[str, Any]) -> set[str]:
    """Classify one trace into one or more failure buckets."""

    categories: set[str] = set()
    status = str(trace.get("status") or "")
    attempts = trace.get("attempts") or []
    final_attempt = attempts[-1] if attempts else {}
    final_validation = final_attempt.get("validation") or {}
    final_execution = final_attempt.get("execution_result") or {}
    final_trace_execution = trace.get("final_execution") or {}

    if status != "success" and attempts:
        if not final_validation.get("ok", False):
            categories.add("validation")
        elif not final_execution.get("ok", False):
            categories.add("execution")

    is_failed = status == "failed"

    if _is_empty_result(trace):
        categories.add("empty_result")
    if is_failed and _has_critic_signal(trace):
        categories.add("critic")
    if is_failed and _looks_like_retrieval_miss(trace, final_attempt):
        categories.add("retrieval_miss")
    if is_failed and _looks_like_aggregation_issue(trace, final_attempt):
        categories.add("aggregation_issue")
    if is_failed and _looks_like_date_filter_issue(trace, final_attempt):
        categories.add("date_filter_issue")

    if status == "success" and final_trace_execution and not final_trace_execution.get("ok", True):
        categories.add("execution")

    return categories


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


def _looks_like_retrieval_miss(
    trace: dict[str, Any],
    final_attempt: dict[str, Any],
) -> bool:
    """Use a narrow heuristic for schema-selection misses."""

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
    if "schema_selection" in trace and not schema.get("expanded_tables"):
        return True
    return "not allowed" in error_text and "table" in error_text


def _looks_like_aggregation_issue(
    trace: dict[str, Any],
    final_attempt: dict[str, Any],
) -> bool:
    """Tag traces that likely failed on totals, grouping, or rollups."""

    text = _analysis_text(trace, final_attempt)
    keywords = ("group by", "aggregate", "aggregation", "sum(", "avg(", "count(")
    return any(keyword in text for keyword in keywords)


def _looks_like_date_filter_issue(
    trace: dict[str, Any],
    final_attempt: dict[str, Any],
) -> bool:
    """Tag traces that likely failed on date or time filtering."""

    text = _analysis_text(trace, final_attempt)
    keywords = ("date", "time", "month", "year", "quarter", "strftime", "between")
    return any(keyword in text for keyword in keywords)


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

    lines.append("")
    lines.append("## By Database")
    for db, bucket in report["by_database"].items():
        lines.append(
            f"- {db}: total {bucket['total']}, "
            f"success {bucket['success']}, failed {bucket['failed']}, skipped {bucket['skipped']}"
        )

    lines.append("")
    return "\n".join(lines)


def _run_retrieval_mode_case(
    task: Task,
    *,
    expected_tables: list[str],
    mode: RetrievalMode,
    config: RuntimeConfig,
    outputs_root: Path,
) -> dict[str, Any]:
    """Run one fixture question with one retrieval mode and capture its outcome."""

    run_paths = ensure_run_paths(f"{task.instance_id}-{mode}", outputs_root=outputs_root)
    answer = run_task(
        task,
        run_paths=run_paths,
        config=config.model_copy(update={"retrieval_mode": mode}),
        force=True,
    )
    trace = json.loads(Path(answer.trace_path).read_text(encoding="utf-8"))
    schema = trace.get("schema_selection") or {}
    expanded_tables = list(schema.get("expanded_tables") or [])
    return {
        "selected_tables": list(schema.get("selected_tables") or []),
        "expanded_tables": expanded_tables,
        "selection_prompt_chars": int(schema.get("selection_prompt_chars") or 0),
        "candidate_table_count": int(schema.get("candidate_table_count") or 0),
        "missing_expected_tables": sorted(
            table for table in expected_tables if table not in expanded_tables
        ),
        "final_sql_outcome": answer.status,
        "trace_path": answer.trace_path,
        "sql_path": trace.get("sql_path"),
    }


def _mode_summary(results: list[dict[str, Any]], mode: RetrievalMode) -> dict[str, Any]:
    """Roll per-case retrieval results into a small scorecard."""

    prompt_chars = 0
    miss_count = 0
    success_count = 0
    for case in results:
        outcome = case["modes"][mode]
        prompt_chars += outcome["selection_prompt_chars"]
        if outcome["missing_expected_tables"]:
            miss_count += 1
        if outcome["final_sql_outcome"] == "success":
            success_count += 1
    case_count = len(results)
    return {
        "case_count": case_count,
        "success_count": success_count,
        "miss_count": miss_count,
        "average_selection_prompt_chars": round(prompt_chars / case_count, 1)
        if case_count
        else 0.0,
    }


def _render_retrieval_compare(report: dict[str, Any]) -> str:
    """Write a concise markdown summary of the retrieval experiment."""

    lexical = report["summary"]["lexical"]
    llm_only = report["summary"]["llm_only"]
    lines = [f"# Retrieval comparison for {report['run_id']}", ""]
    lines.append(
        "- Lexical: "
        f"{lexical['success_count']}/{lexical['case_count']} success, "
        f"{lexical['miss_count']} table misses, "
        f"avg selector prompt {lexical['average_selection_prompt_chars']} chars"
    )
    lines.append(
        "- LLM-only: "
        f"{llm_only['success_count']}/{llm_only['case_count']} success, "
        f"{llm_only['miss_count']} table misses, "
        f"avg selector prompt {llm_only['average_selection_prompt_chars']} chars"
    )
    lines.append("")
    lines.append("## Cases")
    for case in report["cases"]:
        lines.append(f"- {case['instance_id']} ({case['db']})")
        for mode in ("lexical", "llm_only"):
            outcome = case["modes"][mode]
            misses = ", ".join(outcome["missing_expected_tables"]) or "none"
            selected = ", ".join(outcome["selected_tables"]) or "none"
            lines.append(
                f"  - {mode}: selected [{selected}], "
                f"selector prompt {outcome['selection_prompt_chars']} chars, "
                f"misses [{misses}], outcome {outcome['final_sql_outcome']}"
            )
    lines.append("")
    return "\n".join(lines)
