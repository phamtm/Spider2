"""Commands for inspecting local run artifacts."""

from __future__ import annotations

from typing import Annotated, Any

import typer

from sol01.llm.llm_call_logs import (
    build_llm_call_detail_sections,
    build_llm_call_summary_row,
    build_llm_call_summary_rows,
    format_llm_call_value,
    load_llm_call_log,
)
from sol01.output.output import OUTPUTS_ROOT
from sol01.output.registry import resolve_llm_call_log_path


def register(app: typer.Typer) -> None:
    app.command("llm-calls")(llm_calls)


def llm_calls(
    run_id: Annotated[
        str,
        typer.Option(help="Existing run ID to inspect."),
    ],
    instance_id: Annotated[
        str,
        typer.Option(help="Inspect one exact instance within the run."),
    ],
    call_id: Annotated[
        str | None,
        typer.Option(help="Show full details for one call ID."),
    ] = None,
    all_calls: Annotated[
        bool,
        typer.Option("--all", help="Show full details for every call."),
    ] = False,
) -> None:
    """Inspect the local LLM call log for one task."""

    if call_id is not None and all_calls:
        raise typer.BadParameter("Use either --call-id or --all, not both.")

    log_path = resolve_llm_call_log_path(
        {"run_id": run_id, "instance_id": instance_id},
        outputs_root=OUTPUTS_ROOT,
    )
    log = load_llm_call_log(log_path)
    if not log.records:
        typer.echo(f"No usable LLM call rows were found for {run_id}/{instance_id} at {log_path}.")
        raise typer.Exit(code=1)

    if log.errors:
        typer.secho(
            f"Skipped {len(log.errors)} corrupted LLM call row(s) while loading {log_path}.",
            err=True,
            fg=typer.colors.YELLOW,
        )

    summary_rows = build_llm_call_summary_rows(log)
    if call_id is None and not all_calls:
        echo_llm_call_summary(summary_rows)
        return

    selected_records = log.records
    if call_id is not None:
        selected_records = [record for record in log.records if record.call_id == call_id]
        if not selected_records:
            typer.echo(f"No call with ID {call_id} was found in {log_path}.")
            raise typer.Exit(code=1)

    for record in selected_records:
        echo_llm_call_detail(record)


def echo_llm_call_summary(summary_rows: list[dict[str, Any]]) -> None:
    """Print one compact summary line per LLM call."""

    for row in summary_rows:
        attempts = int(row["attempts"])
        error_state = str(row["error_state"] or "").strip()
        error_text = error_state if error_state and error_state != "—" else "no error"
        typer.echo(
            "- "
            f"{row['sequence']}: {row['prompt_name']} | {row['status']} | "
            f"{row['duration']} | {row['model']} | "
            f"{attempts} attempt{'s' if attempts != 1 else ''} | {error_text}"
        )


def echo_llm_call_detail(record: Any) -> None:
    """Print the full request and response payload for one LLM call."""

    summary = build_llm_call_summary_row(record)
    sections = build_llm_call_detail_sections(record)
    typer.echo(
        f"Call {summary['sequence']}: {summary['prompt_name']} | {summary['status']} | "
        f"{summary['duration']} | {summary['model']}"
    )
    typer.echo(f"Call ID: {summary['call_id']}")
    typer.echo(f"Attempts: {summary['attempts']}")
    typer.echo("")
    typer.echo("System prompt:")
    typer.echo(format_llm_call_value(sections["system_prompt"]))
    typer.echo("")
    typer.echo("User prompt:")
    typer.echo(format_llm_call_value(sections["user_prompt"]))
    typer.echo("")
    typer.echo("Output schema:")
    typer.echo(format_llm_call_value(sections["output_schema"]))
    typer.echo("")
    typer.echo("Validated response:")
    typer.echo(format_llm_call_value(sections["validated_output"]))
    typer.echo("")
    typer.echo("Attempts:")
    typer.echo(format_llm_call_value(sections["attempts"]))
    typer.echo("")
    typer.echo("Error:")
    typer.echo(format_llm_call_value(sections["error"]))
    typer.echo("")
