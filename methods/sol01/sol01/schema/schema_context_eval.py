"""Offline gold-table coverage evaluation for schema context selection."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sol01.infra.config import SchemaContextConfig
from sol01.infra.paths import REPO_ROOT
from sol01.loading.docs import load_document_text
from sol01.models import (
    SchemaContextObject,
    SchemaObject,
    SelectedSchemaObject,
    TableSchema,
    Task,
)
from sol01.schema.db_index import load_db_index
from sol01.schema.large_schema_summaries import load_large_schema_summary_registry
from sol01.schema.resolver import resolve_schema_context
from sol01.schema.schema_context import build_available_schema_context
from sol01.schema.schema_context_cache import SchemaContextCache, build_schema_context_cache

DEFAULT_GOLD_TABLE_PATH = REPO_ROOT / "methods" / "gold-tables" / "spider2-snow-gold-tables.jsonl"
DEFAULT_FAILURE_EVIDENCE_LIMIT = 5
DEFAULT_FAILURE_LIMIT = 20
DEFAULT_PROMPT_WIN_LIMIT = 20
DEFAULT_PROMPT_WIN_THRESHOLD = 0.25


def db_schema_summary(db_index: Mapping[str, TableSchema]) -> str:
    """Render the full-schema baseline used by offline schema context evaluation."""

    parts: list[str] = []
    for table_name in sorted(db_index):
        table = db_index[table_name]
        columns = ", ".join(_column_summary(column) for column in table.columns)
        parts.append(f"Table {table_name}: {columns}")
    return "\n".join(parts)


def _column_summary(column: Any) -> str:
    summary = column.name
    if column.type:
        summary += f" [{column.type}]"
    if column.description:
        summary += f" - {column.description}"
    elif column.sample_values:
        preview = ", ".join(column.sample_values[:2])
        summary += f" - sample values: {preview}"
    return summary


@dataclass(frozen=True)
class SchemaContextEvalReport:
    """Aggregate and per-task schema-context-eval results."""

    task_count: int
    object_cutoff: int
    covered_task_count: int
    pre_resolver_gold_recall: float
    pre_resolver_any_gold_recall: float
    post_resolver_gold_recall: float
    post_resolver_all_gold_recall: float
    family_expansion_success: float | None
    average_prompt_reduction: float
    tasks: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    recall_regressions: list[dict[str, Any]]
    prompt_size_wins: list[dict[str, Any]]
    hallucinated_column_failures: list[dict[str, Any]]

    def summary(self) -> dict[str, Any]:
        """Return compact metrics suitable for JSON output or CLI display."""

        return {
            "task_count": self.task_count,
            "object_cutoff": self.object_cutoff,
            "covered_task_count": self.covered_task_count,
            "pre_resolver_gold_recall": self.pre_resolver_gold_recall,
            "pre_resolver_any_gold_recall": self.pre_resolver_any_gold_recall,
            "post_resolver_gold_recall": self.post_resolver_gold_recall,
            "post_resolver_all_gold_recall": self.post_resolver_all_gold_recall,
            "family_expansion_success": self.family_expansion_success,
            "average_prompt_reduction": self.average_prompt_reduction,
            "failure_count": len(self.failures),
            "recall_regression_count": len(self.recall_regressions),
            "prompt_size_win_count": len(self.prompt_size_wins),
            "hallucinated_column_failure_count": len(self.hallucinated_column_failures),
        }

    def payload(self) -> dict[str, Any]:
        """Return the complete report as a JSON-serializable object."""

        return {
            "summary": self.summary(),
            "tasks": self.tasks,
            "failures": self.failures,
            "recall_regressions": self.recall_regressions,
            "prompt_size_wins": self.prompt_size_wins,
            "hallucinated_column_failures": self.hallucinated_column_failures,
        }


def load_gold_tables(path: Path = DEFAULT_GOLD_TABLE_PATH) -> dict[str, list[str]]:
    """Load offline gold-table annotations keyed by Spider2 instance id."""

    gold_tables: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            instance_id = str(record.get("instance_id") or "").strip()
            tables = record.get("gold_tables")
            if not instance_id or not isinstance(tables, list):
                raise ValueError(f"Invalid gold-table row at {path}:{line_number}")
            gold_tables[instance_id] = _stable_tables(str(table) for table in tables)
    return gold_tables


def run_schema_context_eval(
    tasks: Sequence[Task],
    *,
    gold_tables_by_instance: Mapping[str, Sequence[str]],
    config: SchemaContextConfig | None = None,
    db_index_loader: Callable[[str], Mapping[str, TableSchema]] = load_db_index,
    schema_context_cache_loader: Callable[
        [str, Mapping[str, TableSchema], SchemaContextConfig],
        SchemaContextCache,
    ]
    | None = None,
    document_loader: Callable[[str], str] = load_document_text,
    failure_evidence_limit: int = DEFAULT_FAILURE_EVIDENCE_LIMIT,
    failure_limit: int = DEFAULT_FAILURE_LIMIT,
    covered_only: bool = False,
    baseline_tasks: Mapping[str, Mapping[str, Any]] | None = None,
    trace_dirs: Sequence[Path] = (),
) -> SchemaContextEvalReport:
    """Evaluate schema context and resolver coverage against offline gold tables."""

    config = config or SchemaContextConfig()
    schema_context_cache_loader = schema_context_cache_loader or _build_index_for_eval
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for task in tasks:
        gold_tables = _stable_tables(gold_tables_by_instance.get(task.instance_id, ()))
        if not gold_tables:
            continue
        covered_summary_ids = _covered_summary_ids(gold_tables)
        if covered_only and not covered_summary_ids:
            continue

        db_index = dict(db_index_loader(task.db))
        schema_context_cache = schema_context_cache_loader(task.db, db_index, config)
        linked_docs = _task_linked_docs(task, document_loader)
        schema_context_objects, context_diagnostics = build_available_schema_context(
            schema_context_cache,
            task.question,
            linked_docs=linked_docs,
            config=config,
        )
        cutoff_objects = list(schema_context_objects[: config.object_cutoff])
        selected_objects = [
            SelectedSchemaObject(
                object_id=item.schema_object.object_id,
                role=_role_for_schema_object(item.schema_object),
                reason="offline schema context evaluation",
            )
            for item in cutoff_objects
        ]
        resolved = resolve_schema_context(
            db=task.db,
            selected_objects=selected_objects,
            canonical_schema_objects=schema_context_cache.objects,
            db_index=db_index,
            question=task.question,
            schema_context_evidence=cutoff_objects,
        )

        context_tables = _tables_from_schema_context_objects(cutoff_objects)
        resolved_tables = _stable_tables(resolved.resolved_tables)
        context_gold_tables = _gold_overlap(gold_tables, context_tables)
        resolved_gold_tables = _gold_overlap(gold_tables, resolved_tables)
        missing_tables = [
            table
            for table in gold_tables
            if _normalize_table(table) not in _normalized(resolved_tables)
        ]
        pre_recall = _gold_recall(gold_tables, context_tables)
        post_recall = _gold_recall(gold_tables, resolved_tables)
        pre_any_gold = bool(context_gold_tables)
        post_all_gold = not missing_tables
        family_case, family_success = _family_expansion_result(
            gold_tables,
            cutoff_objects,
            resolved_tables,
        )
        full_schema_chars = len(db_schema_summary(db_index))
        resolved_prompt_chars = len(resolved.prompt_context)
        prompt_reduction = _prompt_reduction(full_schema_chars, resolved_prompt_chars)
        prompt_chars_saved = max(0, full_schema_chars - resolved_prompt_chars)
        row = {
            "instance_id": task.instance_id,
            "db": task.db,
            "gold_tables": gold_tables,
            "covered_gold_tables": _covered_tables(gold_tables),
            "covered_summary_ids": covered_summary_ids,
            "schema_context_object_ids": [item.schema_object.object_id for item in cutoff_objects],
            "context_tables": context_tables,
            "context_gold_tables": context_gold_tables,
            "resolved_tables": resolved_tables,
            "resolved_gold_tables": resolved_gold_tables,
            "pre_resolver_gold_recall": pre_recall,
            "pre_resolver_any_gold": pre_any_gold,
            "post_resolver_gold_recall": post_recall,
            "post_resolver_all_gold": post_all_gold,
            "missing_gold_tables": missing_tables,
            "family_expansion_case": family_case,
            "family_expansion_success": family_success,
            "full_schema_chars": full_schema_chars,
            "resolved_prompt_chars": resolved_prompt_chars,
            "prompt_chars_saved": prompt_chars_saved,
            "prompt_reduction": prompt_reduction,
            "context_diagnostics": context_diagnostics,
        }
        rows.append(row)
        if missing_tables and len(failures) < failure_limit:
            failures.append(
                {
                    "instance_id": task.instance_id,
                    "db": task.db,
                    "missing_gold_tables": missing_tables,
                    "top_evidence": _top_evidence(cutoff_objects, limit=failure_evidence_limit),
                }
            )

    return _build_report(
        rows,
        object_cutoff=config.object_cutoff,
        failures=failures,
        baseline_tasks=baseline_tasks or {},
        hallucinated_column_failures=_hallucinated_column_failures(trace_dirs),
    )


def load_schema_context_eval_task_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Load task rows from a previous persisted schema-context-eval report."""

    task_rows_path = path / "tasks.jsonl" if path.is_dir() else path
    rows: list[dict[str, Any]] = []
    if task_rows_path.suffix == ".jsonl":
        with task_rows_path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        payload = json.loads(task_rows_path.read_text(encoding="utf-8"))
        rows = list(payload.get("tasks") or [])
    return {
        str(row.get("instance_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("instance_id")
    }


def write_schema_context_eval_report(report: SchemaContextEvalReport, output_dir: Path) -> Path:
    """Persist schema-context-eval artifacts under the repo-local outputs tree."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report.payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(report.summary(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "tasks.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in report.tasks),
        encoding="utf-8",
    )
    (output_dir / "failures.json").write_text(
        json.dumps(
            {
                "recall_regressions": report.recall_regressions,
                "missing_gold_tables": report.failures,
                "hallucinated_column_failures": report.hallucinated_column_failures,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_render_report_markdown(report), encoding="utf-8")
    return output_dir


def _build_index_for_eval(
    db: str,
    db_index: Mapping[str, TableSchema],
    config: SchemaContextConfig,
) -> SchemaContextCache:
    """Build schema context artifacts without touching runtime coordinator paths."""

    return build_schema_context_cache(
        db,
        db_index=db_index,
        config=config,
    )


def _build_report(
    rows: list[dict[str, Any]],
    *,
    object_cutoff: int,
    failures: list[dict[str, Any]],
    baseline_tasks: Mapping[str, Mapping[str, Any]],
    hallucinated_column_failures: list[dict[str, Any]],
) -> SchemaContextEvalReport:
    task_count = len(rows)
    family_rows = [row for row in rows if row["family_expansion_case"]]
    return SchemaContextEvalReport(
        task_count=task_count,
        object_cutoff=object_cutoff,
        covered_task_count=sum(1 for row in rows if row["covered_summary_ids"]),
        pre_resolver_gold_recall=_mean_float(row["pre_resolver_gold_recall"] for row in rows),
        pre_resolver_any_gold_recall=_mean_bool(row["pre_resolver_any_gold"] for row in rows),
        post_resolver_gold_recall=_mean_float(row["post_resolver_gold_recall"] for row in rows),
        post_resolver_all_gold_recall=_mean_bool(row["post_resolver_all_gold"] for row in rows),
        family_expansion_success=(
            _mean_bool(row["family_expansion_success"] for row in family_rows)
            if family_rows
            else None
        ),
        average_prompt_reduction=_mean_float(row["prompt_reduction"] for row in rows),
        tasks=rows,
        failures=failures,
        recall_regressions=_recall_regressions(rows, baseline_tasks=baseline_tasks),
        prompt_size_wins=_prompt_size_wins(rows),
        hallucinated_column_failures=hallucinated_column_failures,
    )


def _task_linked_docs(task: Task, document_loader: Callable[[str], str]) -> list[str]:
    if not task.external_knowledge:
        return []
    return [document_loader(task.external_knowledge)]


def _role_for_schema_object(schema_object: SchemaObject) -> str:
    if schema_object.object_type in {"table", "family"}:
        return "primary"
    if schema_object.object_type == "join_candidate":
        return "join"
    if schema_object.object_type in {"column", "column_group"}:
        return "metric"
    if schema_object.object_type == "sample_value":
        return "filter"
    return "unknown"


def _tables_from_schema_context_objects(objects: Sequence[SchemaContextObject]) -> list[str]:
    tables: list[str] = []
    for item in objects:
        tables.extend(_schema_object_tables(item.schema_object))
    return _stable_tables(tables)


def _schema_object_tables(schema_object: SchemaObject) -> list[str]:
    tables: list[str] = []
    if schema_object.table_name:
        tables.append(schema_object.table_name)
    if table_full_name := schema_object.metadata.get("table_full_name"):
        tables.append(str(table_full_name))
    for key in ("member_table_refs",):
        value = schema_object.metadata.get(key)
        if isinstance(value, list):
            tables.extend(str(item) for item in value)
    for side_key in ("left", "right"):
        side = schema_object.metadata.get(side_key)
        if isinstance(side, dict) and side.get("table_full_name"):
            tables.append(str(side["table_full_name"]))
    return tables


def _tables_from_chunk_id(object_id: str) -> list[str]:
    if object_id.startswith("table:"):
        return [object_id.removeprefix("table:")]
    if object_id.startswith(("column:", "sample_value:")):
        return [object_id.split(":", 1)[1].split("#", 1)[0]]
    if object_id.startswith("join_candidate:"):
        body = object_id.split(":", 1)[1]
        left, _, right = body.partition("->")
        return [left.split("#", 1)[0], right.split("#", 1)[0]]
    return []


def _covered_summary_ids(gold_tables: Sequence[str]) -> list[str]:
    registry = load_large_schema_summary_registry()
    return sorted(
        {summary.summary_id for table in gold_tables for summary in registry.match_table_ref(table)}
    )


def _covered_tables(gold_tables: Sequence[str]) -> list[str]:
    registry = load_large_schema_summary_registry()
    return [table for table in gold_tables if registry.match_table_ref(table)]


def _gold_overlap(gold_tables: Sequence[str], candidate_tables: Sequence[str]) -> list[str]:
    candidate_lookup = _normalized(candidate_tables)
    return [table for table in gold_tables if _normalize_table(table) in candidate_lookup]


def _gold_recall(gold_tables: Sequence[str], candidate_tables: Sequence[str]) -> float:
    if not gold_tables:
        return 0.0
    return round(len(_gold_overlap(gold_tables, candidate_tables)) / len(gold_tables), 6)


def _family_expansion_result(
    gold_tables: Sequence[str],
    schema_context_objects: Sequence[SchemaContextObject],
    resolved_tables: Sequence[str],
) -> tuple[bool, bool | None]:
    """Report success for gold-table sets that should exercise family expansion."""

    normalized_gold = {_normalize_table(table) for table in gold_tables}
    if len(normalized_gold) < 2:
        return False, None

    resolved = _normalized(resolved_tables)
    for item in schema_context_objects:
        schema_object = item.schema_object
        if schema_object.object_type != "family":
            continue
        family_gold = normalized_gold.intersection(
            _normalized(_schema_object_tables(schema_object))
        )
        if len(family_gold) >= 2:
            return True, normalized_gold.issubset(resolved)
    return False, None


def _top_evidence(
    schema_context_objects: Sequence[SchemaContextObject],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for item in schema_context_objects:
        if item.planning_text:
            evidence.append(
                {
                    "object_id": item.schema_object.object_id,
                    "object_type": item.schema_object.object_type,
                    "rank": item.rank,
                    "score": item.score,
                    "text": " ".join(item.planning_text.split())[:500],
                }
            )
        if len(evidence) >= limit:
            return evidence
    return evidence


def _prompt_reduction(full_schema_chars: int, resolved_prompt_chars: int) -> float:
    if full_schema_chars <= 0:
        return 0.0
    return round(1.0 - (resolved_prompt_chars / full_schema_chars), 6)


def _mean_bool(values: Sequence[bool] | Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return round(sum(1 for item in items if item) / len(items), 6)


def _mean_float(values: Sequence[float] | Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return round(sum(float(item) for item in items) / len(items), 6)


def _recall_regressions(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_tasks: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    regressions: list[dict[str, Any]] = []
    for row in rows:
        baseline = baseline_tasks.get(str(row["instance_id"]))
        if baseline:
            regression = _baseline_recall_regression(row, baseline)
        else:
            regression = _incomplete_recall_regression(row)
        if regression:
            regressions.append(regression)
    return sorted(
        regressions,
        key=lambda item: (
            item["post_resolver_gold_recall"],
            item["pre_resolver_gold_recall"],
            item["instance_id"],
        ),
    )


def _baseline_recall_regression(
    row: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any] | None:
    pre_recall = float(row["pre_resolver_gold_recall"])
    post_recall = float(row["post_resolver_gold_recall"])
    baseline_pre = _baseline_recall_value(
        baseline,
        "pre_resolver_gold_recall",
        "pre_resolver_any_gold",
    )
    baseline_post = _baseline_recall_value(
        baseline,
        "post_resolver_gold_recall",
        "post_resolver_all_gold",
    )
    if pre_recall >= baseline_pre and post_recall >= baseline_post:
        return None
    stages = []
    if pre_recall < baseline_pre:
        stages.append("pre_resolver")
    if post_recall < baseline_post:
        stages.append("post_resolver")
    return {
        "instance_id": row["instance_id"],
        "db": row["db"],
        "stage": "+".join(stages),
        "pre_resolver_gold_recall": pre_recall,
        "baseline_pre_resolver_gold_recall": baseline_pre,
        "post_resolver_gold_recall": post_recall,
        "baseline_post_resolver_gold_recall": baseline_post,
        "missing_gold_tables": row["missing_gold_tables"],
        "covered_summary_ids": row["covered_summary_ids"],
    }


def _baseline_recall_value(
    baseline: Mapping[str, Any],
    recall_key: str,
    bool_key: str,
) -> float:
    if recall_key in baseline:
        return float(baseline[recall_key])
    if bool_key in baseline:
        return 1.0 if baseline[bool_key] else 0.0
    return 0.0


def _incomplete_recall_regression(row: Mapping[str, Any]) -> dict[str, Any] | None:
    pre_recall = float(row["pre_resolver_gold_recall"])
    post_recall = float(row["post_resolver_gold_recall"])
    if pre_recall >= 1.0 and post_recall >= 1.0:
        return None
    if pre_recall < 1.0:
        stage = "pre_resolver"
    else:
        stage = "post_resolver"
    return {
        "instance_id": row["instance_id"],
        "db": row["db"],
        "stage": stage,
        "pre_resolver_gold_recall": pre_recall,
        "post_resolver_gold_recall": post_recall,
        "missing_gold_tables": row["missing_gold_tables"],
        "covered_summary_ids": row["covered_summary_ids"],
    }


def _prompt_size_wins(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    wins = [
        {
            "instance_id": row["instance_id"],
            "db": row["db"],
            "full_schema_chars": row["full_schema_chars"],
            "resolved_prompt_chars": row["resolved_prompt_chars"],
            "prompt_chars_saved": row["prompt_chars_saved"],
            "prompt_reduction": row["prompt_reduction"],
            "covered_summary_ids": row["covered_summary_ids"],
        }
        for row in rows
        if float(row["prompt_reduction"]) >= DEFAULT_PROMPT_WIN_THRESHOLD
    ]
    return sorted(
        wins,
        key=lambda item: (-item["prompt_chars_saved"], item["instance_id"]),
    )[:DEFAULT_PROMPT_WIN_LIMIT]


def _hallucinated_column_failures(trace_dirs: Sequence[Path]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for trace_path in _iter_trace_files(trace_dirs):
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        errors = _hallucinated_column_errors(trace)
        if not errors:
            continue
        failures.append(
            {
                "instance_id": trace.get("instance_id") or trace_path.stem,
                "db": trace.get("db"),
                "status": trace.get("status"),
                "trace_path": str(trace_path),
                "error_count": len(errors),
                "errors": errors[:5],
            }
        )
    return sorted(failures, key=lambda item: str(item["instance_id"]))


def _iter_trace_files(trace_dirs: Sequence[Path]) -> list[Path]:
    trace_files: list[Path] = []
    for trace_dir in trace_dirs:
        if trace_dir.is_file():
            trace_files.append(trace_dir)
        elif trace_dir.exists():
            trace_files.extend(sorted(trace_dir.glob("*.json")))
    return trace_files


def _hallucinated_column_errors(trace: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for attempt in trace.get("attempts") or []:
        validation = attempt.get("validation") or {}
        for error in validation.get("errors") or []:
            text = str(error)
            if _is_hallucinated_column_error(text):
                errors.append(text)
    return _stable_strings(errors)


def _is_hallucinated_column_error(error: str) -> bool:
    text = error.casefold()
    return (
        "unknown column" in text
        or "unknown quoted column" in text
        or "no selected table has it" in text
    )


def _render_report_markdown(report: SchemaContextEvalReport) -> str:
    summary = report.summary()
    lines = [
        "# Schema Context Evaluation",
        "",
        f"- tasks: {summary['task_count']}",
        f"- covered schema tasks: {summary['covered_task_count']}",
        f"- object cutoff: {summary['object_cutoff']}",
        f"- pre-resolver gold recall: {summary['pre_resolver_gold_recall']:.3f}",
        f"- post-resolver gold recall: {summary['post_resolver_gold_recall']:.3f}",
        f"- average prompt reduction: {summary['average_prompt_reduction']:.3f}",
        f"- recall regressions: {summary['recall_regression_count']}",
        f"- hallucinated-column failures: {summary['hallucinated_column_failure_count']}",
        "",
        "## Recall Regressions",
    ]
    if report.recall_regressions:
        lines.extend(
            (
                f"- {item['instance_id']}: {item['stage']} "
                f"pre={item['pre_resolver_gold_recall']:.3f} "
                f"post={item['post_resolver_gold_recall']:.3f} "
                f"missing={', '.join(item['missing_gold_tables']) or 'none'}"
            )
            for item in report.recall_regressions[:10]
        )
    else:
        lines.append("- none")

    lines.extend(["", "## Prompt Size Wins"])
    if report.prompt_size_wins:
        lines.extend(
            (
                f"- {item['instance_id']}: saved {item['prompt_chars_saved']} chars "
                f"({item['prompt_reduction']:.1%})"
            )
            for item in report.prompt_size_wins[:10]
        )
    else:
        lines.append("- none")

    lines.extend(["", "## Hallucinated Column Failures"])
    if report.hallucinated_column_failures:
        lines.extend(
            (
                f"- {item['instance_id']}: {item['error_count']} validation error(s); "
                f"{item['errors'][0]}"
            )
            for item in report.hallucinated_column_failures[:10]
        )
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _stable_tables(values: Sequence[str] | Any) -> list[str]:
    seen: set[str] = set()
    tables: list[str] = []
    for value in values:
        table = str(value).strip()
        if not table:
            continue
        normalized = _normalize_table(table)
        if normalized in seen:
            continue
        seen.add(normalized)
        tables.append(table)
    return tables


def _stable_strings(values: Sequence[str] | Any) -> list[str]:
    seen: set[str] = set()
    strings: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        strings.append(item)
    return strings


def _normalized(values: Sequence[str]) -> set[str]:
    return {_normalize_table(value) for value in values}


def _normalize_table(table: str) -> str:
    return table.strip().strip('"').casefold()
