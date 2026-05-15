"""Final trace, SQL, and CSV output helpers for one task run."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from sol01.candidates.selection import final_winner_reason, select_winner
from sol01.execution.snowflake_runner import dataframe_records
from sol01.infra.logging import get_logger
from sol01.models import ExecutionResult, FinalAnswer
from sol01.output.output import RunPaths, csv_path_for, write_sql, write_trace
from sol01.pipeline_state import TaskRun, current_best
from sol01.workflow import TASK_STATUS_FAILED, TASK_STATUS_SUCCESS

logger = get_logger(__name__)


def write_task_output(
    run: TaskRun,
    *,
    run_paths: RunPaths,
    task_trace_path: Path,
    task_llm_log_path: Path,
    live_logging_enabled: bool,
    started_at: float,
) -> FinalAnswer:
    """Write final SQL, CSV, and trace; return the FinalAnswer."""

    task = run.task
    best = current_best(run)
    final_selection = select_winner(run.attempts) if best is not None else None
    final_attempt_index = final_selection.index if final_selection is not None else None

    trace_payload: dict[str, object] = {
        "instance_id": task.instance_id,
        "db": task.db,
        "question": task.question,
        "schema_selection": run.schema.model_dump(mode="json"),
        "schema_context": run.schema_context,
        "solver_policy": run.policy.as_dict(),
        "intent": run.intent.model_dump(mode="json"),
        "prompt_hashes": run.prompt_hashes,
        "final_attempt_index": final_attempt_index,
        "final_attempt_reason": final_winner_reason(
            best,
            candidate_review_payload=run.candidate_review_payload,
        ),
        "attempts": [attempt.model_dump(mode="json") for attempt in run.attempts],
    }
    if run.candidate_review_payload is not None:
        trace_payload["candidate_review"] = run.candidate_review_payload.model_dump(mode="json")
    if run.recovery_payload is not None:
        trace_payload["recovery"] = run.recovery_payload.model_dump(mode="json")
    if live_logging_enabled:
        trace_payload["llm_call_log_path"] = str(task_llm_log_path)

    if best is not None and best.execution_result.ok:
        sql_path = write_sql(run_paths, instance_id=task.instance_id, sql=best.sql)
        csv_path = csv_path_for(run_paths, instance_id=task.instance_id)
        dataframe = best._dataframe
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(csv_path, index=False)
        final_execution = ExecutionResult(
            ok=True,
            row_count=len(dataframe),
            columns=[str(column) for column in dataframe.columns],
            sample_rows=dataframe_records(dataframe.head(3)),
            csv_path=str(csv_path),
            error=None,
        )
        trace_payload.update(
            {
                "status": TASK_STATUS_SUCCESS,
                "final_sql": best.sql,
                "sql_path": str(sql_path),
                "csv_path": str(csv_path),
                "final_execution": final_execution.model_dump(mode="json"),
            }
        )
        write_trace(run_paths, instance_id=task.instance_id, trace=trace_payload)
        elapsed = round(perf_counter() - started_at, 3)
        logger.info(
            "task complete",
            instance_id=task.instance_id,
            status=TASK_STATUS_SUCCESS,
            run_root=str(run_paths.root),
            attempts=len(run.attempts),
            best_stage=best.stage,
            best_score=best.score,
            row_count=len(dataframe),
            columns=[str(column) for column in dataframe.columns],
            elapsed_seconds=elapsed,
            sql_path=str(sql_path),
            csv_path=str(csv_path),
        )
        return FinalAnswer(
            instance_id=task.instance_id,
            status=TASK_STATUS_SUCCESS,
            sql=best.sql,
            csv_path=str(csv_path),
            trace_path=str(task_trace_path),
        )

    trace_payload.update(
        {
            "status": TASK_STATUS_FAILED,
            "final_sql": best.sql if best is not None else None,
            "csv_path": None,
        }
    )
    write_trace(run_paths, instance_id=task.instance_id, trace=trace_payload)
    elapsed = round(perf_counter() - started_at, 3)
    logger.warning(
        "task complete",
        instance_id=task.instance_id,
        status=TASK_STATUS_FAILED,
        run_root=str(run_paths.root),
        attempts=len(run.attempts),
        best_stage=best.stage if best is not None else None,
        best_score=best.score if best is not None else None,
        row_count=best.execution_result.row_count if best is not None else 0,
        elapsed_seconds=elapsed,
    )
    return FinalAnswer(
        instance_id=task.instance_id,
        status=TASK_STATUS_FAILED,
        sql=best.sql if best is not None else None,
        csv_path=None,
        trace_path=str(task_trace_path),
    )
