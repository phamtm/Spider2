"""Coordinate schema context, LLM calls, validation, execution, and trace writing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from pydantic import BaseModel

from sol01.infra.config import (
    DEFAULT_SCHEMA_CONTEXT_VERSION,
    RuntimeConfig,
    SchemaContextConfig,
)
from sol01.infra.logging import get_logger
from sol01.llm.client import LLMClient, PromptSpec
from sol01.llm.llm_logging import LLMCallLogger
from sol01.models import (
    FinalAnswer,
    Task,
)
from sol01.output.output import (
    OUTPUTS_ROOT,
    RunPaths,
    ensure_run_paths,
    llm_call_log_path_for,
    trace_path_for,
    write_manifest,
    write_trace,
)
from sol01.pipeline import (
    TaskRun,
    attempt_schema_expansion,
    check_skip,
    generate_initial_candidates,
    plan_schema,
    repair_failed_execution,
    review_and_repair,
    write_task_output,
)
from sol01.schema.db_index import load_db_index
from sol01.schema.index import CACHE_PATH
from sol01.schema.schema_context_cache import build_schema_context_cache

logger = get_logger(__name__)

__all__ = ["run_task", "run_tasks"]


class StructuredLLM(Protocol):
    """Minimal LLM interface the coordinator needs for structured calls."""

    def load_prompt(self, prompt_name: str) -> PromptSpec: ...

    def run_structured(
        self,
        user_prompt: str,
        *,
        prompt_name: str,
        output_type: type[BaseModel],
        model: Any = None,
    ) -> BaseModel: ...

    def run_structured_with_prompt(
        self,
        user_prompt: str,
        *,
        prompt: PromptSpec,
        output_type: type[BaseModel],
        model: Any = None,
    ) -> BaseModel: ...


def run_tasks(
    tasks: list[Task],
    *,
    run_id: str,
    config: RuntimeConfig,
    schema_context_config: SchemaContextConfig,
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

    _prewarm_schema_context_caches(tasks, schema_context_config=schema_context_config)
    results = _run_task_batch(
        tasks,
        run_paths=run_paths,
        config=config,
        schema_context_config=schema_context_config,
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


def _prewarm_schema_context_caches(
    tasks: list[Task],
    *,
    cache_path: Path = CACHE_PATH,
    schema_context_config: SchemaContextConfig,
) -> None:
    """Build each selected database schema-context cache before workers start."""

    seen: set[str] = set()
    for task in tasks:
        if task.db in seen:
            continue
        seen.add(task.db)
        db_index = load_db_index(task.db, cache_path=cache_path)
        build_schema_context_cache(
            task.db,
            db_index=db_index,
            config=schema_context_config,
        )


def _run_task_batch(
    tasks: list[Task],
    *,
    run_paths: RunPaths,
    config: RuntimeConfig,
    schema_context_config: SchemaContextConfig,
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
                schema_context_config=schema_context_config,
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
                schema_context_config=schema_context_config,
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
    schema_context_config: SchemaContextConfig,
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
            schema_context_config=schema_context_config,
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


def run_task(
    task: Task,
    *,
    run_paths: RunPaths,
    config: RuntimeConfig,
    schema_context_config: SchemaContextConfig,
    llm_client: StructuredLLM | None = None,
    force: bool = False,
    skip_failed: bool = False,
    initial_candidates: int = 3,
    max_attempts: int = 4,
    semantic_repairs: int = 1,
) -> FinalAnswer:
    """Run one task through the explicit solver stage pipeline."""

    started_at = perf_counter()
    live_logging_enabled = llm_client is None
    task_llm_log_path = llm_call_log_path_for(run_paths, instance_id=task.instance_id)
    client = llm_client or LLMClient(
        config,
        call_logger=LLMCallLogger(task_llm_log_path),
    )
    task_trace_path = trace_path_for(run_paths, instance_id=task.instance_id)

    skipped = check_skip(
        task,
        run_paths,
        task_trace_path,
        force=force,
        skip_failed=skip_failed,
        expected_schema_context_version=DEFAULT_SCHEMA_CONTEXT_VERSION,
    )
    if skipped is not None:
        return skipped

    run = TaskRun(task=task, client=client, schema_context_config=schema_context_config)
    run = plan_schema(run, run_paths=run_paths)
    run = generate_initial_candidates(run, count=initial_candidates, max_attempts=max_attempts)
    run = repair_failed_execution(run, max_attempts=max_attempts)
    run = review_and_repair(run, max_attempts=max_attempts, semantic_repairs=semantic_repairs)
    run = attempt_schema_expansion(run)
    return write_task_output(
        run,
        run_paths=run_paths,
        task_trace_path=task_trace_path,
        task_llm_log_path=task_llm_log_path,
        live_logging_enabled=live_logging_enabled,
        started_at=started_at,
    )
