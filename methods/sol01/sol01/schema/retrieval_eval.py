"""Offline gold-table coverage evaluation for schema retrieval."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sol01.infra.config import SchemaRetrievalConfig
from sol01.infra.paths import REPO_ROOT
from sol01.loading.docs import load_document_text
from sol01.models import (
    RetrievedSchemaObject,
    SchemaObject,
    SelectedSchemaObject,
    TableSchema,
    Task,
)
from sol01.schema.embedding import EmbeddingProvider, RerankerProvider
from sol01.schema.hybrid_retrieval import retrieve_schema_objects
from sol01.schema.resolver import resolve_schema_context
from sol01.schema.retrieval import db_schema_summary, load_db_index
from sol01.schema.retrieval_index import SchemaRetrievalIndex, build_retrieval_index

DEFAULT_GOLD_TABLE_PATH = REPO_ROOT / "methods" / "gold-tables" / "spider2-snow-gold-tables.jsonl"
DEFAULT_FAILURE_EVIDENCE_LIMIT = 5
DEFAULT_FAILURE_LIMIT = 20


@dataclass(frozen=True)
class RetrievalEvalReport:
    """Aggregate and per-task retrieval-eval results."""

    task_count: int
    object_cutoff: int
    pre_resolver_any_gold_recall: float
    post_resolver_all_gold_recall: float
    family_expansion_success: float | None
    average_prompt_reduction: float
    tasks: list[dict[str, Any]]
    failures: list[dict[str, Any]]

    def summary(self) -> dict[str, Any]:
        """Return compact metrics suitable for JSON output or CLI display."""

        return {
            "task_count": self.task_count,
            "object_cutoff": self.object_cutoff,
            "pre_resolver_any_gold_recall": self.pre_resolver_any_gold_recall,
            "post_resolver_all_gold_recall": self.post_resolver_all_gold_recall,
            "family_expansion_success": self.family_expansion_success,
            "average_prompt_reduction": self.average_prompt_reduction,
            "failure_count": len(self.failures),
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


def run_retrieval_eval(
    tasks: Sequence[Task],
    *,
    gold_tables_by_instance: Mapping[str, Sequence[str]],
    config: SchemaRetrievalConfig | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    reranker_provider: RerankerProvider | None = None,
    db_index_loader: Callable[[str], Mapping[str, TableSchema]] = load_db_index,
    retrieval_index_loader: Callable[
        [str, Mapping[str, TableSchema], SchemaRetrievalConfig, EmbeddingProvider | None],
        SchemaRetrievalIndex,
    ]
    | None = None,
    document_loader: Callable[[str], str] = load_document_text,
    failure_evidence_limit: int = DEFAULT_FAILURE_EVIDENCE_LIMIT,
    failure_limit: int = DEFAULT_FAILURE_LIMIT,
) -> RetrievalEvalReport:
    """Evaluate retrieval and resolver coverage against offline gold tables."""

    config = config or SchemaRetrievalConfig()
    retrieval_index_loader = retrieval_index_loader or _build_index_for_eval
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for task in tasks:
        gold_tables = _stable_tables(gold_tables_by_instance.get(task.instance_id, ()))
        if not gold_tables:
            continue

        db_index = dict(db_index_loader(task.db))
        retrieval_index = retrieval_index_loader(task.db, db_index, config, embedding_provider)
        linked_docs = _task_linked_docs(task, document_loader)
        retrieved_objects, retrieval_diagnostics = retrieve_schema_objects(
            retrieval_index,
            task.question,
            linked_docs=linked_docs,
            embedding_provider=embedding_provider,
            reranker_provider=reranker_provider,
            config=config,
        )
        cutoff_objects = list(retrieved_objects[: config.object_top_k])
        selected_objects = [
            SelectedSchemaObject(
                object_id=item.schema_object.object_id,
                role=_role_for_schema_object(item.schema_object),
                reason="offline retrieval evaluation",
            )
            for item in cutoff_objects
        ]
        resolved = resolve_schema_context(
            db=task.db,
            selected_objects=selected_objects,
            canonical_schema_objects=retrieval_index.objects,
            db_index=db_index,
            question=task.question,
            retrieval_evidence=cutoff_objects,
        )

        retrieved_tables = _tables_from_retrieved_objects(cutoff_objects)
        resolved_tables = _stable_tables(resolved.allowed_tables)
        missing_tables = [
            table
            for table in gold_tables
            if _normalize_table(table) not in _normalized(resolved_tables)
        ]
        pre_any_gold = any(
            _normalize_table(table) in _normalized(retrieved_tables) for table in gold_tables
        )
        post_all_gold = not missing_tables
        family_case, family_success = _family_expansion_result(
            gold_tables,
            cutoff_objects,
            resolved_tables,
        )
        full_schema_chars = len(db_schema_summary(db_index))
        resolved_prompt_chars = len(resolved.prompt_context)
        prompt_reduction = _prompt_reduction(full_schema_chars, resolved_prompt_chars)
        row = {
            "instance_id": task.instance_id,
            "db": task.db,
            "gold_tables": gold_tables,
            "retrieved_object_ids": [item.schema_object.object_id for item in cutoff_objects],
            "retrieved_tables": retrieved_tables,
            "resolved_tables": resolved_tables,
            "pre_resolver_any_gold": pre_any_gold,
            "post_resolver_all_gold": post_all_gold,
            "missing_gold_tables": missing_tables,
            "family_expansion_case": family_case,
            "family_expansion_success": family_success,
            "full_schema_chars": full_schema_chars,
            "resolved_prompt_chars": resolved_prompt_chars,
            "prompt_reduction": prompt_reduction,
            "retrieval_diagnostics": retrieval_diagnostics,
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

    return _build_report(rows, object_cutoff=config.object_top_k, failures=failures)


def _build_index_for_eval(
    db: str,
    db_index: Mapping[str, TableSchema],
    config: SchemaRetrievalConfig,
    embedding_provider: EmbeddingProvider | None,
) -> SchemaRetrievalIndex:
    """Build retrieval artifacts without touching runtime coordinator paths."""

    return build_retrieval_index(
        db,
        db_index=db_index,
        embedding_provider=embedding_provider,
        config=config,
    )


def _build_report(
    rows: list[dict[str, Any]],
    *,
    object_cutoff: int,
    failures: list[dict[str, Any]],
) -> RetrievalEvalReport:
    task_count = len(rows)
    family_rows = [row for row in rows if row["family_expansion_case"]]
    return RetrievalEvalReport(
        task_count=task_count,
        object_cutoff=object_cutoff,
        pre_resolver_any_gold_recall=_mean_bool(row["pre_resolver_any_gold"] for row in rows),
        post_resolver_all_gold_recall=_mean_bool(row["post_resolver_all_gold"] for row in rows),
        family_expansion_success=(
            _mean_bool(row["family_expansion_success"] for row in family_rows)
            if family_rows
            else None
        ),
        average_prompt_reduction=_mean_float(row["prompt_reduction"] for row in rows),
        tasks=rows,
        failures=failures,
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


def _tables_from_retrieved_objects(objects: Sequence[RetrievedSchemaObject]) -> list[str]:
    tables: list[str] = []
    for item in objects:
        tables.extend(_schema_object_tables(item.schema_object))
        for chunk in item.chunks:
            tables.extend(_tables_from_chunk_id(chunk.chunk.object_id))
            tables.extend(_tables_from_chunk_id(chunk.chunk.chunk_id.split("::", 1)[0]))
            for parent_id in chunk.chunk.parent_object_ids:
                tables.extend(_tables_from_chunk_id(parent_id))
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


def _family_expansion_result(
    gold_tables: Sequence[str],
    retrieved_objects: Sequence[RetrievedSchemaObject],
    resolved_tables: Sequence[str],
) -> tuple[bool, bool | None]:
    """Report success for gold-table sets that should exercise family expansion."""

    normalized_gold = {_normalize_table(table) for table in gold_tables}
    if len(normalized_gold) < 2:
        return False, None

    resolved = _normalized(resolved_tables)
    for item in retrieved_objects:
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
    retrieved_objects: Sequence[RetrievedSchemaObject],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for item in retrieved_objects:
        for retrieved_chunk in item.chunks:
            chunk = retrieved_chunk.chunk
            text = chunk.prompt_text or chunk.rerank_text or chunk.bm25_text or chunk.text
            evidence.append(
                {
                    "object_id": item.schema_object.object_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_type": chunk.chunk_type,
                    "rank": item.rank,
                    "score": item.score,
                    "text": " ".join(text.split())[:500],
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


def _normalized(values: Sequence[str]) -> set[str]:
    return {_normalize_table(value) for value in values}


def _normalize_table(table: str) -> str:
    return table.strip().strip('"').casefold()
