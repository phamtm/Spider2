"""Coordinate retrieval, LLM calls, validation, execution, and trace writing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from sol01.config import RuntimeConfig
from sol01.docs import get_metric_definition
from sol01.llm import LLMClient
from sol01.models import (
    ConfidenceReport,
    ExecutionResult,
    FinalAnswer,
    Intent,
    MetricDefinition,
    SchemaSelection,
    SQLCandidate,
    Task,
)
from sol01.output import (
    RunPaths,
    csv_path_for,
    ensure_run_paths,
    should_skip_task,
    trace_path_for,
    write_manifest,
    write_sql,
    write_trace,
)
from sol01.profiling import profile_dataframe
from sol01.retrieval import retrieve_schema
from sol01.sqlite_runner import _dataframe_records, fetch_query_dataframe
from sol01.validation import validate_sql


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
    force: bool = False,
    skip_failed: bool = False,
) -> list[FinalAnswer]:
    """Run a batch of tasks and write a manifest before processing them."""

    run_paths = ensure_run_paths(run_id)
    write_manifest(
        run_paths,
        manifest={
            "task_ids": [task.instance_id for task in tasks],
            "model": config.model,
            "provider_routing": config.provider_routing,
            "config": {
                "concurrency": config.concurrency,
                "max_schema_tables": config.max_schema_tables,
            },
        },
    )

    client = llm_client or LLMClient(config)
    return [
        run_task(
            task,
            run_paths=run_paths,
            config=config,
            llm_client=client,
            force=force,
            skip_failed=skip_failed,
        )
        for task in tasks
    ]


def run_task(
    task: Task,
    *,
    run_paths: RunPaths,
    config: RuntimeConfig,
    llm_client: StructuredLLM | None = None,
    db_path: Path | None = None,
    force: bool = False,
    skip_failed: bool = False,
    initial_candidates: int = 3,
    max_attempts: int = 4,
    semantic_repairs: int = 1,
) -> FinalAnswer:
    """Run one task from retrieval through final trace writing."""

    client = llm_client or LLMClient(config)
    task_trace_path = trace_path_for(run_paths, instance_id=task.instance_id)

    if not force and should_skip_task(
        run_paths, instance_id=task.instance_id, skip_failed=skip_failed
    ):
        existing_trace = json.loads(task_trace_path.read_text(encoding="utf-8"))
        return FinalAnswer(
            instance_id=task.instance_id,
            status="skipped",
            sql=existing_trace.get("final_sql"),
            csv_path=existing_trace.get("csv_path"),
            trace_path=str(task_trace_path),
        )

    prompt_hashes: dict[str, str] = {}
    attempts: list[dict[str, Any]] = []
    metric_definitions: list[MetricDefinition] = []
    critic_repairs_used = 0

    schema = retrieve_schema(
        task.question,
        task.db,
        max_tables=min(4, config.max_schema_tables),
        max_expanded_tables=config.max_schema_tables,
    )
    intent = _run_prompt(
        client,
        prompt_hashes=prompt_hashes,
        prompt_name="intent",
        output_type=Intent,
        user_prompt=_intent_user_prompt(task, schema),
    )

    for metric_name in intent.metrics:
        metric_definitions.append(
            get_metric_definition(metric_name, instance_id=task.instance_id, db=task.db)
        )

    schema_context = _schema_context(schema)
    docs_context = _docs_context(metric_definitions)

    for candidate_index in range(initial_candidates):
        if len(attempts) >= max_attempts:
            break
        candidate = _run_prompt(
            client,
            prompt_hashes=prompt_hashes,
            prompt_name="sql_generation",
            output_type=SQLCandidate,
            user_prompt=_sql_generation_prompt(task, intent, schema_context, docs_context),
        )
        attempts.append(
            _evaluate_candidate(
                task=task,
                candidate=candidate,
                schema=schema,
                db_path=db_path,
                stage=f"initial_{candidate_index + 1}",
            )
        )

    best_attempt = _best_attempt(attempts)

    if (
        best_attempt is not None
        and not best_attempt["execution_result"]["ok"]
        and len(attempts) < max_attempts
    ):
        repaired_candidate = _run_prompt(
            client,
            prompt_hashes=prompt_hashes,
            prompt_name="sql_repair",
            output_type=SQLCandidate,
            user_prompt=_sql_repair_prompt(task, best_attempt),
        )
        attempts.append(
            _evaluate_candidate(
                task=task,
                candidate=repaired_candidate,
                schema=schema,
                db_path=db_path,
                stage="repair",
            )
        )
        best_attempt = _best_attempt(attempts)

    if (
        best_attempt is not None
        and best_attempt["execution_result"]["ok"]
        and critic_repairs_used < semantic_repairs
        and len(attempts) < max_attempts
    ):
        critic = _run_prompt(
            client,
            prompt_hashes=prompt_hashes,
            prompt_name="result_critic",
            output_type=ConfidenceReport,
            user_prompt=_critic_prompt(task, best_attempt, schema_context),
        )
        best_attempt["critic"] = critic.model_dump(mode="json")
        if critic.should_repair:
            critic_repairs_used += 1
            repaired_candidate = _run_prompt(
                client,
                prompt_hashes=prompt_hashes,
                prompt_name="sql_repair",
                output_type=SQLCandidate,
                user_prompt=_semantic_repair_prompt(task, best_attempt, critic),
            )
            attempts.append(
                _evaluate_candidate(
                    task=task,
                    candidate=repaired_candidate,
                    schema=schema,
                    db_path=db_path,
                    stage="critic_repair",
                )
            )
            best_attempt = _best_attempt(attempts)

    trace_payload = {
        "instance_id": task.instance_id,
        "db": task.db,
        "question": task.question,
        "schema_selection": schema.model_dump(mode="json"),
        "intent": intent.model_dump(mode="json"),
        "metric_definitions": [metric.model_dump(mode="json") for metric in metric_definitions],
        "prompt_hashes": prompt_hashes,
        "attempts": attempts,
    }

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
        trace_payload["attempts"] = [_trace_attempt(attempt) for attempt in attempts]
        write_trace(run_paths, instance_id=task.instance_id, trace=trace_payload)
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
    trace_payload["attempts"] = [_trace_attempt(attempt) for attempt in attempts]
    write_trace(run_paths, instance_id=task.instance_id, trace=trace_payload)
    return FinalAnswer(
        instance_id=task.instance_id,
        status="failed",
        sql=best_attempt["sql"] if best_attempt is not None else None,
        csv_path=None,
        trace_path=str(task_trace_path),
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


def _evaluate_candidate(
    *,
    task: Task,
    candidate: SQLCandidate,
    schema: SchemaSelection,
    db_path: Path | None,
    stage: str,
) -> dict[str, Any]:
    """Validate and execute one candidate, then return a trace-ready attempt record."""

    validation = validate_sql(candidate.sql, allowed_tables=schema.expanded_tables)
    if validation.ok:
        try:
            dataframe = fetch_query_dataframe(candidate.sql, db=task.db, db_path=db_path)
            execution = ExecutionResult(
                ok=True,
                row_count=len(dataframe),
                columns=[str(column) for column in dataframe.columns],
                sample_rows=_dataframe_records(dataframe.head(3)),
                csv_path=None,
                error=None,
            )
        except Exception as exc:
            dataframe = None
            execution = ExecutionResult(
                ok=False,
                row_count=0,
                columns=[],
                sample_rows=[],
                csv_path=None,
                error=str(exc),
            )
    else:
        dataframe = None
        execution = ExecutionResult(
            ok=False,
            row_count=0,
            columns=[],
            sample_rows=[],
            csv_path=None,
            error="Validation failed before execution.",
        )

    attempt: dict[str, Any] = {
        "stage": stage,
        "sql": candidate.sql,
        "explanation": candidate.explanation,
        "assumptions": candidate.assumptions,
        "candidate_confidence": candidate.confidence,
        "validation": validation.model_dump(mode="json"),
        "execution_result": execution.model_dump(mode="json"),
        "score": _attempt_score(candidate=candidate, validation=validation, execution=execution),
    }

    if execution.ok:
        attempt["result_profile"] = profile_dataframe(dataframe)
        attempt["_dataframe"] = dataframe

    return attempt


def _attempt_score(
    *,
    candidate: SQLCandidate,
    validation: Any,
    execution: Any,
) -> float:
    """Prefer successfully executed candidates, then valid ones, then candidate confidence."""

    score = candidate.confidence
    if validation.ok:
        score += 10.0
    if execution.ok:
        score += 100.0
        if execution.columns:
            score += 2.0
        if execution.row_count > 0:
            score += 1.0
    return score


def _best_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the highest-scoring attempt so far."""

    if not attempts:
        return None
    return max(attempts, key=lambda attempt: float(attempt["score"]))


def _trace_attempt(attempt: dict[str, Any]) -> dict[str, Any]:
    """Drop non-serializable internal fields before writing the trace."""

    return {key: value for key, value in attempt.items() if not key.startswith("_")}


def _schema_context(schema: SchemaSelection) -> str:
    """Render a compact schema summary for prompt inputs."""

    return (
        f"DB: {schema.db}\n"
        f"Selected tables: {', '.join(schema.selected_tables)}\n"
        f"Expanded tables: {', '.join(schema.expanded_tables)}\n"
        f"Rationale: {schema.rationale}"
    )


def _docs_context(metric_definitions: list[MetricDefinition]) -> str:
    """Render retrieved metric definitions for SQL prompts."""

    if not metric_definitions:
        return "No metric-specific document context."
    return "\n\n".join(
        f"{metric.metric_name}: {metric.definition}" for metric in metric_definitions
    )


def _intent_user_prompt(task: Task, schema: SchemaSelection) -> str:
    """Build the user prompt for intent extraction."""

    return f"Question: {task.question}\n\nSchema context:\n{_schema_context(schema)}"


def _sql_generation_prompt(
    task: Task,
    intent: Intent,
    schema_context: str,
    docs_context: str,
) -> str:
    """Build the SQL-generation prompt body."""

    return (
        f"Question: {task.question}\n\n"
        f"Intent:\n{intent.model_dump_json(indent=2)}\n\n"
        f"Schema context:\n{schema_context}\n\n"
        f"Document context:\n{docs_context}"
    )


def _sql_repair_prompt(task: Task, attempt: dict[str, Any]) -> str:
    """Build a repair prompt using validation or execution feedback."""

    return (
        f"Question: {task.question}\n\n"
        f"Failed SQL:\n{attempt['sql']}\n\n"
        f"Validation:\n{json.dumps(attempt['validation'], indent=2, sort_keys=True)}\n\n"
        f"Execution:\n{json.dumps(attempt['execution_result'], indent=2, sort_keys=True)}"
    )


def _critic_prompt(task: Task, attempt: dict[str, Any], schema_context: str) -> str:
    """Build the critic prompt using the current best SQL and result profile."""

    return (
        f"Question: {task.question}\n\n"
        f"Schema context:\n{schema_context}\n\n"
        f"SQL:\n{attempt['sql']}\n\n"
        "Result profile:\n"
        f"{json.dumps(attempt.get('result_profile', {}), indent=2, sort_keys=True)}"
    )


def _semantic_repair_prompt(
    task: Task,
    attempt: dict[str, Any],
    critic: ConfidenceReport,
) -> str:
    """Build the repair prompt for one critic-triggered retry."""

    return (
        f"Question: {task.question}\n\n"
        f"Current SQL:\n{attempt['sql']}\n\n"
        f"Critic issues:\n{json.dumps(critic.model_dump(mode='json'), indent=2, sort_keys=True)}"
    )
