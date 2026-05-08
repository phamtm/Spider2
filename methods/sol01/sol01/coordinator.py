"""Coordinate retrieval, LLM calls, validation, execution, and trace writing."""

from __future__ import annotations

import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from sol01.candidates.evaluator import _dataframe_records, evaluate_candidate
from sol01.candidates.scoring import _best_attempt
from sol01.candidates.verification import (
    _augment_intent_with_value_groundings,
    _table_schemas_for_selection,
)
from sol01.infra.config import RuntimeConfig
from sol01.infra.logging import get_logger
from sol01.llm.llm_logging import LLMCallLogger
from sol01.llm.prompt_builders import (
    _candidate_review_prompt,
    _planning_user_prompt,
    _question_preview,
    _schema_expansion_prompt,
    _semantic_repair_prompt,
    _sql_generation_batch_prompt,
    _sql_reference_context,
    _sql_repair_prompt,
    schema_expansion_trigger,
)
from sol01.models import (
    CandidateReviewReport,
    ConfidenceReport,
    ExecutionResult,
    FinalAnswer,
    Intent,
    PlanningDecision,
    SchemaExpansionDecision,
    SchemaSelection,
    SQLCandidate,
    SQLCandidateBatch,
    Task,
)
from sol01.output.output import (
    OUTPUTS_ROOT,
    RunPaths,
    csv_path_for,
    ensure_run_paths,
    llm_call_log_path_for,
    should_skip_task,
    trace_path_for,
    write_manifest,
    write_sql,
    write_trace,
)
from sol01.schema.index import CACHE_PATH
from sol01.schema.retrieval import db_schema_summary, load_db_index, sanitize_llm_tables

logger = get_logger(__name__)
LLMClient: Any | None = None

__all__ = ["run_task", "run_tasks"]


def load_document_text(file_name: str) -> str:
    """Load task-linked document text without importing the docs module at startup."""

    from sol01.loading.docs import load_document_text as _load_document_text  # noqa: PLC0415

    return _load_document_text(file_name)


def _llm_client_class() -> Any:
    """Return the live LLM client class, importing it only when needed."""

    global LLMClient
    if LLMClient is None:
        from sol01.llm.client import LLMClient as _LLMClient  # noqa: PLC0415

        LLMClient = _LLMClient
    return LLMClient


class StructuredLLM(Protocol):
    """Minimal LLM interface the coordinator needs for structured calls."""

    def load_prompt(self, prompt_name: str) -> Any: ...

    def run_structured(
        self,
        user_prompt: str,
        *,
        prompt_name: str,
        output_type: type[Any],
        model: Any = None,
    ) -> Any: ...

    def run_structured_with_prompt(
        self,
        user_prompt: str,
        *,
        prompt: Any,
        output_type: type[Any],
        model: Any = None,
    ) -> Any: ...


def run_tasks(
    tasks: list[Task],
    *,
    run_id: str,
    config: RuntimeConfig,
    llm_client: StructuredLLM | None = None,
    outputs_root: Path | None = None,
    force: bool = False,
    skip_failed: bool = False,
) -> list[FinalAnswer]:
    """Run a batch of tasks and write a manifest before processing them."""

    run_paths = ensure_run_paths(run_id, outputs_root=outputs_root or OUTPUTS_ROOT)
    write_manifest(
        run_paths,
        manifest={
            "task_ids": [task.instance_id for task in tasks],
            "model": config.model,
            "provider_routing": config.provider_routing,
            "config": {
                "concurrency": config.concurrency,
            },
        },
    )
    logger.info(
        "run start",
        run_id=run_id,
        task_count=len(tasks),
        model=getattr(config, "model", None),
        provider_routing=getattr(config, "provider_routing", None),
        concurrency=getattr(config, "concurrency", None),
        force=force,
        skip_failed=skip_failed,
    )

    _prewarm_schema_indexes(tasks)
    results = _run_task_batch(
        tasks,
        run_paths=run_paths,
        config=config,
        llm_client=llm_client,
        force=force,
        skip_failed=skip_failed,
    )
    logger.info(
        "run complete",
        run_id=run_id,
        success_count=sum(1 for result in results if result.status == "success"),
        failed_count=sum(1 for result in results if result.status == "failed"),
        skipped_count=sum(1 for result in results if result.status == "skipped"),
    )
    return results


def _prewarm_schema_indexes(tasks: list[Task], *, cache_path: Path = CACHE_PATH) -> None:
    """Load each selected database into the local schema cache before workers start."""

    seen: set[str] = set()
    for task in tasks:
        if task.db in seen:
            continue
        seen.add(task.db)
        load_db_index(task.db, cache_path=cache_path)


def _run_task_batch(
    tasks: list[Task],
    *,
    run_paths: RunPaths,
    config: RuntimeConfig,
    llm_client: StructuredLLM | None,
    force: bool,
    skip_failed: bool,
) -> list[FinalAnswer]:
    """Run tasks sequentially or with bounded concurrency, preserving input order."""

    if config.concurrency <= 1 or llm_client is not None:
        return [
            _run_single_batch_task(
                task,
                run_paths=run_paths,
                config=config,
                llm_client=llm_client,
                force=force,
                skip_failed=skip_failed,
            )
            for task in tasks
        ]

    results: list[FinalAnswer | None] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        future_to_index = {
            executor.submit(
                _run_single_batch_task,
                task,
                run_paths=run_paths,
                config=config,
                llm_client=None,
                force=force,
                skip_failed=skip_failed,
            ): index
            for index, task in enumerate(tasks)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
    return [result for result in results if result is not None]


def _run_single_batch_task(
    task: Task,
    *,
    run_paths: RunPaths,
    config: RuntimeConfig,
    llm_client: StructuredLLM | None,
    force: bool,
    skip_failed: bool,
) -> FinalAnswer:
    """Run one task and convert unexpected exceptions into failed traces."""

    try:
        return run_task(
            task,
            run_paths=run_paths,
            config=config,
            llm_client=llm_client,
            force=force,
            skip_failed=skip_failed,
        )
    except Exception as exc:
        return _record_batch_task_failure(
            task=task,
            run_paths=run_paths,
            live_logging_enabled=llm_client is None,
            error=exc,
        )


def _record_batch_task_failure(
    *,
    task: Task,
    run_paths: RunPaths,
    live_logging_enabled: bool,
    error: Exception,
) -> FinalAnswer:
    """Write a failed trace for an unexpected batch-level task crash."""

    task_trace_path = trace_path_for(run_paths, instance_id=task.instance_id)
    task_llm_log_path = llm_call_log_path_for(run_paths, instance_id=task.instance_id)
    trace_payload: dict[str, Any] = {
        "instance_id": task.instance_id,
        "db": task.db,
        "question": task.question,
        "status": "failed",
        "prompt_hashes": {},
        "attempts": [],
        "final_sql": None,
        "csv_path": None,
        "error": {
            "type": error.__class__.__name__,
            "message": str(error) or error.__class__.__name__,
        },
    }
    if live_logging_enabled:
        trace_payload["llm_call_log_path"] = str(task_llm_log_path)
    write_trace(run_paths, instance_id=task.instance_id, trace=trace_payload)
    logger.exception(
        "task crashed",
        instance_id=task.instance_id,
        db=task.db,
        error_type=error.__class__.__name__,
        error_message=str(error) or error.__class__.__name__,
        trace_path=str(task_trace_path),
    )
    return FinalAnswer(
        instance_id=task.instance_id,
        status="failed",
        sql=None,
        csv_path=None,
        trace_path=str(task_trace_path),
    )


# ---------------------------------------------------------------------------
# Per-task pipeline stages
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _TaskCtx:
    """Shared context threaded through each pipeline stage of run_task."""

    task: Task
    client: StructuredLLM
    intent: Intent
    schema: SchemaSelection
    table_schemas: dict[str, Any]
    sql_reference_context: str
    docs_context: str
    prompt_hashes: dict[str, str]


def _check_skip(
    task: Task,
    run_paths: RunPaths,
    task_trace_path: Path,
    *,
    force: bool,
    skip_failed: bool,
) -> FinalAnswer | None:
    """Return an existing FinalAnswer if the task should be skipped, else None."""

    if force or not should_skip_task(
        run_paths, instance_id=task.instance_id, skip_failed=skip_failed
    ):
        return None
    existing_trace = json.loads(task_trace_path.read_text(encoding="utf-8"))
    logger.info(
        "task skipped",
        instance_id=task.instance_id,
        db=task.db,
        run_root=str(run_paths.root),
        trace_path=str(task_trace_path),
        csv_path=existing_trace.get("csv_path"),
    )
    return FinalAnswer(
        instance_id=task.instance_id,
        status="skipped",
        sql=existing_trace.get("final_sql"),
        csv_path=existing_trace.get("csv_path"),
        trace_path=str(task_trace_path),
    )


def _build_context(
    task: Task,
    client: StructuredLLM,
    prompt_hashes: dict[str, str],
    run_paths: RunPaths,
) -> _TaskCtx:
    """Retrieve schema, extract intent, and assemble the shared pipeline context."""

    logger.info(
        "task start",
        instance_id=task.instance_id,
        db=task.db,
        question_preview=_question_preview(task.question),
        question_length=len(task.question),
        run_root=str(run_paths.root),
    )
    docs_context = (
        load_document_text(task.external_knowledge)
        if task.external_knowledge
        else "No task-linked document context."
    )
    schema, intent = _run_planning(task, client, prompt_hashes, docs_context)
    table_schemas = _table_schemas_for_selection(schema)
    intent = _augment_intent_with_value_groundings(
        intent, task=task, schema=schema, table_schemas=table_schemas
    )
    logger.info(
        "schema selected",
        instance_id=task.instance_id,
        selected_count=len(schema.selected_tables),
        expanded_count=len(schema.expanded_tables),
        selected_tables=schema.selected_tables,
        expanded_tables=schema.expanded_tables,
        confidence=schema.confidence,
    )
    logger.info(
        "intent extracted",
        instance_id=task.instance_id,
        metrics=intent.metrics,
        filters=intent.filters,
        time_constraints=intent.time_constraints,
    )
    return _TaskCtx(
        task=task,
        client=client,
        intent=intent,
        schema=schema,
        table_schemas=table_schemas,
        sql_reference_context=_sql_reference_context(schema, table_schemas),
        docs_context=docs_context,
        prompt_hashes=prompt_hashes,
    )


def _run_planning(
    task: Task,
    client: StructuredLLM,
    prompt_hashes: dict[str, str],
    docs_context: str,
) -> tuple[SchemaSelection, Intent]:
    """Run the combined table-selection and intent-extraction LLM call."""

    db_index = load_db_index(task.db)
    decision = _run_prompt(
        client,
        prompt_hashes=prompt_hashes,
        prompt_name="planning",
        output_type=PlanningDecision,
        user_prompt=_planning_user_prompt(
            task,
            task.db,
            docs_context,
            db_schema_summary(db_index),
        ),
    )
    selected_tables = sanitize_llm_tables(decision.selected_tables, db_index)
    confidence = decision.confidence if selected_tables else 0.0
    rationale = decision.rationale.strip()
    if not selected_tables:
        rationale = f"{rationale} No valid tables matched the schema summary.".strip()
    elif selected_tables != decision.selected_tables[: len(selected_tables)]:
        rationale = (
            f"{rationale} Ignored unknown or duplicate table names returned by the model."
        ).strip()

    schema = SchemaSelection(
        db=task.db,
        retrieval_mode="llm_only",
        selected_tables=selected_tables,
        expanded_tables=list(selected_tables),
        rationale=rationale,
        confidence=confidence,
        selection_prompt_chars=len(
            _planning_user_prompt(task, task.db, docs_context, db_schema_summary(db_index))
        ),
        candidate_table_count=len(db_index),
    )
    return schema, decision.intent


def _log_candidate(instance_id: str, attempt: dict[str, Any]) -> None:
    """Log a standard candidate-evaluated event."""

    logger.info(
        "candidate evaluated",
        instance_id=instance_id,
        stage=attempt["stage"],
        validation_ok=attempt["validation"]["ok"],
        execution_ok=attempt["execution_result"]["ok"],
        score=attempt["score"],
        elapsed_seconds=attempt["elapsed_seconds"],
        row_count=attempt["execution_result"]["row_count"],
    )


def _generate_initial_candidates(
    ctx: _TaskCtx,
    attempts: list[dict[str, Any]],
    *,
    initial_candidates: int,
    max_attempts: int,
) -> dict[str, Any] | None:
    """Generate up to initial_candidates SQL candidates; appends to attempts in place."""

    logger.info(
        "generating candidates",
        instance_id=ctx.task.instance_id,
        initial_candidates=initial_candidates,
        max_attempts=max_attempts,
    )
    candidate_limit = min(initial_candidates, max_attempts - len(attempts))
    if candidate_limit <= 0:
        return _best_attempt(attempts)

    batch = _run_prompt(
        ctx.client,
        prompt_hashes=ctx.prompt_hashes,
        prompt_name="sql_generation_batch",
        output_type=SQLCandidateBatch,
        user_prompt=_sql_generation_batch_prompt(
            ctx.task,
            ctx.intent,
            ctx.sql_reference_context,
            ctx.docs_context,
            candidate_count=candidate_limit,
        ),
    )
    for candidate_index, candidate in enumerate(batch.candidates[:candidate_limit]):
        stage = f"initial_{candidate_index + 1}"
        attempt = evaluate_candidate(
            task=ctx.task,
            candidate=candidate,
            intent=ctx.intent,
            schema=ctx.schema,
            table_schemas=ctx.table_schemas,
            stage=stage,
        )
        attempts.append(attempt)
        _log_candidate(ctx.task.instance_id, attempt)
    return _best_attempt(attempts)


def _repair_failed_execution(
    ctx: _TaskCtx,
    attempts: list[dict[str, Any]],
    best_attempt: dict[str, Any] | None,
    *,
    max_attempts: int,
) -> dict[str, Any] | None:
    """Repair the best attempt if it failed execution; appends to attempts in place."""

    if (
        best_attempt is None
        or best_attempt["execution_result"]["ok"]
        or len(attempts) >= max_attempts
    ):
        return best_attempt

    logger.info(
        "repair requested",
        instance_id=ctx.task.instance_id,
        stage="repair",
        best_stage=best_attempt["stage"],
    )
    repaired_candidate = _run_prompt(
        ctx.client,
        prompt_hashes=ctx.prompt_hashes,
        prompt_name="sql_repair",
        output_type=SQLCandidate,
        user_prompt=_sql_repair_prompt(
            ctx.task,
            ctx.intent,
            best_attempt,
            ctx.sql_reference_context,
            ctx.docs_context,
        ),
    )
    attempt = evaluate_candidate(
        task=ctx.task,
        candidate=repaired_candidate,
        intent=ctx.intent,
        schema=ctx.schema,
        table_schemas=ctx.table_schemas,
        stage="repair",
    )
    attempts.append(attempt)
    _log_candidate(ctx.task.instance_id, attempt)
    return _best_attempt(attempts)


def _review_and_repair(
    ctx: _TaskCtx,
    attempts: list[dict[str, Any]],
    best_attempt: dict[str, Any] | None,
    *,
    max_attempts: int,
    semantic_repairs: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Ask the model to adjudicate executable attempts using local observations."""

    if best_attempt is None or not best_attempt["execution_result"]["ok"]:
        return best_attempt, None

    executable_attempts = [attempt for attempt in attempts if attempt["execution_result"]["ok"]]
    review_reason = _candidate_review_reason(executable_attempts)

    logger.info(
        "candidate review requested",
        instance_id=ctx.task.instance_id,
        best_stage=best_attempt["stage"],
        executable_attempts=len(executable_attempts),
        reason=review_reason,
    )
    review = _run_prompt(
        ctx.client,
        prompt_hashes=ctx.prompt_hashes,
        prompt_name="candidate_review",
        output_type=CandidateReviewReport,
        user_prompt=_candidate_review_prompt(
            ctx.task,
            ctx.intent,
            executable_attempts,
            ctx.sql_reference_context,
            ctx.docs_context,
            baseline_stage=best_attempt["stage"],
            review_reason=review_reason,
        ),
    )
    review_payload = {
        "review_reason": review_reason,
        **review.model_dump(mode="json"),
    }
    if review.preferred_stage:
        preferred_attempt = next(
            (
                attempt
                for attempt in executable_attempts
                if attempt["stage"] == review.preferred_stage
            ),
            None,
        )
        if preferred_attempt is not None:
            best_attempt = preferred_attempt

    best_attempt["critic"] = {
        "confidence": review.confidence,
        "issues": review.issues,
        "should_repair": review.should_repair,
        "repair_focus": review.repair_focus,
    }
    best_attempt["candidate_review"] = review_payload
    logger.info(
        "candidate review complete",
        instance_id=ctx.task.instance_id,
        preferred_stage=review.preferred_stage,
        should_repair=review.should_repair,
        confidence=review.confidence,
        issues=review.issues,
    )

    if not review.should_repair or semantic_repairs < 1:
        return best_attempt, review_payload

    budget_exhausted = len(attempts) >= max_attempts
    if budget_exhausted:
        best_attempt["repair_skipped_reason"] = "attempt budget exhausted"
        logger.info(
            "critic repair skipped",
            instance_id=ctx.task.instance_id,
            reason="attempt budget exhausted",
        )
        return best_attempt, review_payload

    logger.info(
        "semantic repair requested",
        instance_id=ctx.task.instance_id,
        stage="critic_repair",
        focus=review.repair_focus,
    )
    repaired_candidate = _run_prompt(
        ctx.client,
        prompt_hashes=ctx.prompt_hashes,
        prompt_name="sql_repair",
        output_type=SQLCandidate,
        user_prompt=_semantic_repair_prompt(
            ctx.task,
            ctx.intent,
            best_attempt,
            ConfidenceReport(
                confidence=review.confidence,
                issues=review.issues,
                should_repair=review.should_repair,
                repair_focus=review.repair_focus,
            ),
            ctx.sql_reference_context,
            ctx.docs_context,
        ),
    )
    attempt = evaluate_candidate(
        task=ctx.task,
        candidate=repaired_candidate,
        intent=ctx.intent,
        schema=ctx.schema,
        table_schemas=ctx.table_schemas,
        stage="critic_repair",
    )
    attempts.append(attempt)
    _log_candidate(ctx.task.instance_id, attempt)

    new_best = attempt if attempt["execution_result"]["ok"] else _best_attempt(attempts)
    return new_best, review_payload


def _candidate_review_reason(executable_attempts: list[dict[str, Any]]) -> str:
    """Return the trace reason for model-led final candidate adjudication."""

    if len(executable_attempts) == 1:
        return "final adjudication of the only executable candidate"
    return "final adjudication across executable candidates using local observations"


def _expand_schema_selection(
    schema: SchemaSelection,
    additional_tables: list[str],
    db_index: dict[str, Any],
) -> tuple[SchemaSelection, list[str]]:
    """Return an expanded SchemaSelection and the list of tables actually added."""

    sanitized = sanitize_llm_tables(additional_tables, db_index)
    combined = list(schema.expanded_tables)
    added: list[str] = []
    for table in sanitized:
        if table not in combined:
            combined.append(table)
            added.append(table)
    return (
        schema.model_copy(update={"expanded_tables": combined, "selected_tables": combined}),
        added,
    )


def _rebuild_context_for_expansion(
    ctx: _TaskCtx,
    expanded_schema: SchemaSelection,
) -> _TaskCtx:
    """Rebuild selected-schema context while preserving the original answer contract."""

    new_table_schemas = _table_schemas_for_selection(expanded_schema)
    intent = _augment_intent_with_value_groundings(
        ctx.intent, task=ctx.task, schema=expanded_schema, table_schemas=new_table_schemas
    )
    return _TaskCtx(
        task=ctx.task,
        client=ctx.client,
        intent=intent,
        schema=expanded_schema,
        table_schemas=new_table_schemas,
        sql_reference_context=_sql_reference_context(expanded_schema, new_table_schemas),
        docs_context=ctx.docs_context,
        prompt_hashes=ctx.prompt_hashes,
    )


def _attempt_schema_expansion(
    ctx: _TaskCtx,
    attempts: list[dict[str, Any]],
    best_attempt: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, _TaskCtx | None]:
    """Run one schema-expansion recovery attempt when evidence warrants it.

    Returns (updated_best_attempt, schema_expansion_payload, expanded_ctx).
    expanded_ctx is non-None only when new tables were actually added.  The
    expansion attempt is outside the normal max_attempts budget.
    """

    if best_attempt is None:
        return best_attempt, None, None

    trigger = schema_expansion_trigger(best_attempt)
    if trigger is None:
        return best_attempt, None, None

    logger.info(
        "schema expansion triggered",
        instance_id=ctx.task.instance_id,
        trigger=trigger,
    )

    try:
        db_index = load_db_index(ctx.task.db)
    except Exception:
        logger.warning(
            "schema expansion skipped: db_index unavailable",
            instance_id=ctx.task.instance_id,
            exc_info=True,
        )
        return best_attempt, None, None

    deterministic_tables = _deterministic_expansion_tables(
        trigger, best_attempt, ctx.schema, db_index
    )
    expansion_payload: dict[str, Any] = {
        "trigger": trigger,
        "decision": None,
        "added_tables": [],
        "outcome": "no_new_tables",
    }

    if deterministic_tables:
        requested_tables = deterministic_tables
        expansion_payload["decision"] = {
            "source": "deterministic",
            "additional_tables": deterministic_tables,
        }
    else:
        decision = _run_prompt(
            ctx.client,
            prompt_hashes=ctx.prompt_hashes,
            prompt_name="schema_expansion",
            output_type=SchemaExpansionDecision,
            user_prompt=_schema_expansion_prompt(
                ctx.task,
                best_attempt,
                trigger,
                ctx.schema,
                db_schema_summary(db_index),
            ),
        )
        expansion_payload["decision"] = decision.model_dump(mode="json")
        if not decision.should_expand:
            expansion_payload["outcome"] = "model_declined"
            logger.info(
                "schema expansion declined by model",
                instance_id=ctx.task.instance_id,
            )
            return best_attempt, expansion_payload, None
        requested_tables = decision.additional_tables

    expanded_schema, added_tables = _expand_schema_selection(ctx.schema, requested_tables, db_index)
    expansion_payload["added_tables"] = added_tables

    if not added_tables:
        expansion_payload["outcome"] = "no_new_tables"
        logger.info(
            "schema expansion skipped: no new valid tables",
            instance_id=ctx.task.instance_id,
            requested=requested_tables,
        )
        return best_attempt, expansion_payload, None

    logger.info(
        "schema expanding",
        instance_id=ctx.task.instance_id,
        added_tables=added_tables,
    )

    expanded_ctx = _rebuild_context_for_expansion(ctx, expanded_schema)

    expansion_stage = "schema_expansion"
    expansion_payload["expansion_attempt_stage"] = expansion_stage
    logger.info(
        "candidate request",
        instance_id=ctx.task.instance_id,
        stage=expansion_stage,
        prompt_name="sql_generation_batch",
    )
    batch = _run_prompt(
        expanded_ctx.client,
        prompt_hashes=expanded_ctx.prompt_hashes,
        prompt_name="sql_generation_batch",
        output_type=SQLCandidateBatch,
        user_prompt=_sql_generation_batch_prompt(
            expanded_ctx.task,
            expanded_ctx.intent,
            expanded_ctx.sql_reference_context,
            expanded_ctx.docs_context,
            candidate_count=1,
        ),
    )
    if not batch.candidates:
        expansion_payload["outcome"] = "expanded_no_candidate"
        return _best_attempt(attempts), expansion_payload, expanded_ctx
    expansion_attempt = evaluate_candidate(
        task=expanded_ctx.task,
        candidate=batch.candidates[0],
        intent=expanded_ctx.intent,
        schema=expanded_ctx.schema,
        table_schemas=expanded_ctx.table_schemas,
        stage=expansion_stage,
    )
    attempts.append(expansion_attempt)
    _log_candidate(ctx.task.instance_id, expansion_attempt)

    if expansion_attempt["execution_result"]["ok"]:
        expansion_payload["outcome"] = "expanded"
    else:
        expansion_payload["outcome"] = "expanded_failed"

    return _best_attempt(attempts), expansion_payload, expanded_ctx


def _deterministic_expansion_tables(
    trigger: str,
    attempt: dict[str, Any],
    schema: SchemaSelection,
    db_index: dict[str, Any],
) -> list[str]:
    """Infer expansion tables from explicit names before asking the LLM."""

    text_parts = [
        trigger,
        str(attempt.get("sql") or ""),
        json.dumps(attempt.get("validation", {}), sort_keys=True),
        str(attempt.get("execution_result", {}).get("error") or ""),
        json.dumps(attempt.get("critic", {}), sort_keys=True),
    ]
    haystack = _identifier_search_text("\n".join(text_parts))
    current = set(schema.expanded_tables)
    selected: list[str] = []
    for table_name in sorted(db_index):
        if table_name in current:
            continue
        candidates = _table_name_aliases(table_name)
        if any(alias and alias in haystack for alias in candidates):
            selected.append(table_name)
    return selected


def _identifier_search_text(value: str) -> str:
    """Normalize free text so table aliases can be found with boundary spaces."""

    return f" {''.join(char.lower() if char.isalnum() else ' ' for char in value)} "


def _table_name_aliases(table_name: str) -> list[str]:
    """Return normalized aliases for a fully qualified table name."""

    parts = [part for part in table_name.lower().split(".") if part]
    aliases = {f" {parts[-1]} "} if parts else set()
    for start in range(len(parts)):
        aliases.add(f" {' '.join(parts[start:])} ")
    return sorted(aliases, key=len, reverse=True)


def _write_task_output(
    ctx: _TaskCtx,
    attempts: list[dict[str, Any]],
    best_attempt: dict[str, Any] | None,
    run_paths: RunPaths,
    task_trace_path: Path,
    task_llm_log_path: Path,
    *,
    candidate_review_payload: dict[str, Any] | None,
    schema_expansion_payload: dict[str, Any] | None,
    expanded_ctx: _TaskCtx | None,
    live_logging_enabled: bool,
    started_at: float,
) -> FinalAnswer:
    """Write final SQL, CSV, and trace; return the FinalAnswer."""

    task = ctx.task
    final_ctx = expanded_ctx if expanded_ctx is not None else ctx
    trace_payload: dict[str, Any] = {
        "instance_id": task.instance_id,
        "db": task.db,
        "question": task.question,
        "retrieval_mode": final_ctx.schema.retrieval_mode,
        "schema_selection": final_ctx.schema.model_dump(mode="json"),
        "intent": final_ctx.intent.model_dump(mode="json"),
        "prompt_hashes": ctx.prompt_hashes,
        "attempts": attempts,
    }
    if candidate_review_payload is not None:
        trace_payload["candidate_review"] = candidate_review_payload
    if schema_expansion_payload is not None:
        trace_payload["schema_expansion"] = schema_expansion_payload
    if live_logging_enabled:
        trace_payload["llm_call_log_path"] = str(task_llm_log_path)

    if best_attempt is not None and best_attempt["execution_result"]["ok"]:
        sql_path = write_sql(run_paths, instance_id=task.instance_id, sql=best_attempt["sql"])
        csv_path = csv_path_for(run_paths, instance_id=task.instance_id)
        best_dataframe = best_attempt["_dataframe"]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        best_dataframe.to_csv(csv_path, index=False)
        final_execution = ExecutionResult(
            ok=True,
            row_count=len(best_dataframe),
            columns=[str(column) for column in best_dataframe.columns],
            sample_rows=_dataframe_records(best_dataframe.head(3)),
            csv_path=str(csv_path),
            error=None,
        )
        trace_payload.update(
            {
                "status": "success",
                "final_sql": best_attempt["sql"],
                "sql_path": str(sql_path),
                "csv_path": str(csv_path),
                "final_execution": final_execution.model_dump(mode="json"),
            }
        )
        trace_payload["attempts"] = [_trace_attempt(a) for a in attempts]
        write_trace(run_paths, instance_id=task.instance_id, trace=trace_payload)
        elapsed_seconds = round(perf_counter() - started_at, 3)
        logger.info(
            "task complete",
            instance_id=task.instance_id,
            status="success",
            run_root=str(run_paths.root),
            attempts=len(attempts),
            best_stage=best_attempt["stage"],
            best_score=best_attempt["score"],
            row_count=len(best_dataframe),
            columns=[str(column) for column in best_dataframe.columns],
            elapsed_seconds=elapsed_seconds,
            sql_path=str(sql_path),
            csv_path=str(csv_path),
        )
        return FinalAnswer(
            instance_id=task.instance_id,
            status="success",
            sql=best_attempt["sql"],
            csv_path=str(csv_path),
            trace_path=str(task_trace_path),
        )

    trace_payload.update(
        {
            "status": "failed",
            "final_sql": best_attempt["sql"] if best_attempt is not None else None,
            "csv_path": None,
        }
    )
    trace_payload["attempts"] = [_trace_attempt(a) for a in attempts]
    write_trace(run_paths, instance_id=task.instance_id, trace=trace_payload)
    elapsed_seconds = round(perf_counter() - started_at, 3)
    logger.warning(
        "task complete",
        instance_id=task.instance_id,
        status="failed",
        run_root=str(run_paths.root),
        attempts=len(attempts),
        best_stage=best_attempt["stage"] if best_attempt is not None else None,
        best_score=best_attempt["score"] if best_attempt is not None else None,
        row_count=best_attempt["execution_result"]["row_count"] if best_attempt is not None else 0,
        elapsed_seconds=elapsed_seconds,
    )
    return FinalAnswer(
        instance_id=task.instance_id,
        status="failed",
        sql=best_attempt["sql"] if best_attempt is not None else None,
        csv_path=None,
        trace_path=str(task_trace_path),
    )


def run_task(
    task: Task,
    *,
    run_paths: RunPaths,
    config: RuntimeConfig,
    llm_client: StructuredLLM | None = None,
    force: bool = False,
    skip_failed: bool = False,
    initial_candidates: int = 3,
    max_attempts: int = 4,
    semantic_repairs: int = 1,
) -> FinalAnswer:
    """Run one task from retrieval through final trace writing."""

    started_at = perf_counter()
    live_logging_enabled = llm_client is None
    task_llm_log_path = llm_call_log_path_for(run_paths, instance_id=task.instance_id)
    client = llm_client or _llm_client_class()(
        config,
        call_logger=LLMCallLogger(task_llm_log_path),
    )
    task_trace_path = trace_path_for(run_paths, instance_id=task.instance_id)

    skipped = _check_skip(task, run_paths, task_trace_path, force=force, skip_failed=skip_failed)
    if skipped is not None:
        return skipped

    prompt_hashes: dict[str, str] = {}
    attempts: list[dict[str, Any]] = []

    ctx = _build_context(task, client, prompt_hashes, run_paths)

    best_attempt = _generate_initial_candidates(
        ctx, attempts, initial_candidates=initial_candidates, max_attempts=max_attempts
    )
    best_attempt = _repair_failed_execution(ctx, attempts, best_attempt, max_attempts=max_attempts)
    best_attempt, candidate_review_payload = _review_and_repair(
        ctx, attempts, best_attempt, max_attempts=max_attempts, semantic_repairs=semantic_repairs
    )
    best_attempt, schema_expansion_payload, expanded_ctx = _attempt_schema_expansion(
        ctx, attempts, best_attempt
    )

    return _write_task_output(
        ctx,
        attempts,
        best_attempt,
        run_paths,
        task_trace_path,
        task_llm_log_path,
        candidate_review_payload=candidate_review_payload,
        schema_expansion_payload=schema_expansion_payload,
        expanded_ctx=expanded_ctx,
        live_logging_enabled=live_logging_enabled,
        started_at=started_at,
    )


def _run_prompt(
    client: StructuredLLM,
    *,
    prompt_hashes: dict[str, str],
    prompt_name: str,
    output_type: type[Any],
    user_prompt: str,
) -> Any:
    """Load one prompt hash and run one structured LLM call."""

    prompt = client.load_prompt(prompt_name)
    prompt_hashes.setdefault(prompt_name, prompt.sha256)
    return client.run_structured_with_prompt(
        user_prompt,
        prompt=prompt,
        output_type=output_type,
    )


def _trace_attempt(attempt: dict[str, Any]) -> dict[str, Any]:
    """Drop non-serializable internal fields before writing the trace."""

    return {key: value for key, value in attempt.items() if not key.startswith("_")}
