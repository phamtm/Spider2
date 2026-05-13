"""Validate, execute, profile, and score one SQL candidate into an attempt record."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from sol01.candidates.scoring import attempt_score_breakdown, verification_penalty_reasons
from sol01.candidates.verification import (
    infer_aggregate_grain,
    infer_filter_grounding_report,
    infer_output_shape_report,
)
from sol01.execution.profiling import profile_dataframe
from sol01.execution.snowflake_runner import dataframe_records, fetch_query_dataframe
from sol01.execution.validation import validate_sql
from sol01.infra.logging import get_logger
from sol01.models import (
    ExecutionResult,
    Intent,
    SchemaSelection,
    SQLCandidate,
    TableSchema,
    Task,
)

logger = get_logger(__name__)

__all__ = ["evaluate_candidate"]


def evaluate_candidate(
    *,
    task: Task,
    candidate: SQLCandidate,
    intent: Intent,
    schema: SchemaSelection,
    table_schemas: dict[str, TableSchema] | None = None,
    stage: str,
) -> dict[str, Any]:
    """Validate and execute one candidate, then return a trace-ready attempt record."""

    started_at = perf_counter()
    validation = validate_sql(
        candidate.sql,
        allowed_tables=schema.expanded_tables,
        table_schemas=table_schemas,
    )
    if validation.ok:
        try:
            dataframe = fetch_query_dataframe(candidate.sql, db=task.db)
            execution = ExecutionResult(
                ok=True,
                row_count=len(dataframe),
                columns=[str(column) for column in dataframe.columns],
                sample_rows=dataframe_records(dataframe.head(3)),
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

    aggregate_grain = infer_aggregate_grain(
        task=task,
        candidate=candidate,
        schema=schema,
        table_schemas=table_schemas or {},
        validation=validation,
        execution=execution,
    )
    result_profile = profile_dataframe(dataframe) if execution.ok else None
    shape_report = infer_output_shape_report(
        intent=intent,
        candidate=candidate,
        execution=execution,
        result_profile=result_profile,
    )
    filter_grounding_report = infer_filter_grounding_report(
        task=task,
        candidate=candidate,
        schema=schema,
        table_schemas=table_schemas or {},
        validation=validation,
        execution=execution,
    )
    logger.debug(
        "candidate processed",
        stage=stage,
        validation_ok=validation.ok,
        execution_ok=execution.ok,
        row_count=execution.row_count,
        error=execution.error,
    )
    attempt: dict[str, Any] = {
        "stage": stage,
        "sql": candidate.sql,
        "explanation": candidate.explanation,
        "assumptions": candidate.assumptions,
        "constraint_ledger": candidate.constraint_ledger,
        "unsupported_assumptions": candidate.unsupported_assumptions,
        "candidate_confidence": candidate.confidence,
        "validation": validation.model_dump(mode="json"),
        "execution_result": execution.model_dump(mode="json"),
        "filter_grounding_report": (
            filter_grounding_report.model_dump(mode="json")
            if filter_grounding_report is not None
            else None
        ),
        "shape_report": shape_report.model_dump(mode="json") if shape_report is not None else None,
        "score_breakdown": attempt_score_breakdown(
            intent=intent,
            candidate=candidate,
            validation=validation,
            execution=execution,
            aggregate_grain=aggregate_grain,
            result_profile=result_profile,
            shape_report=shape_report,
            filter_grounding_report=filter_grounding_report,
        ),
        "verification_penalty_reasons": verification_penalty_reasons(
            execution=execution,
            shape_report=shape_report,
            filter_grounding_report=filter_grounding_report,
            aggregate_grain=aggregate_grain,
        ),
    }
    attempt["score"] = sum(attempt["score_breakdown"].values())

    if result_profile is not None:
        attempt["result_profile"] = result_profile
        attempt["_dataframe"] = dataframe
    if aggregate_grain is not None:
        attempt["aggregate_grain"] = aggregate_grain.model_dump(mode="json")
    attempt["elapsed_seconds"] = round(perf_counter() - started_at, 3)

    return attempt
