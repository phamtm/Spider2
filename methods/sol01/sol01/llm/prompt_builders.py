"""Build coordinator prompts and prompt-adjacent context blocks."""

from __future__ import annotations

import json
import re
from typing import Any

from sol01.models import (
    ConfidenceReport,
    Intent,
    SchemaSelection,
    TableSchema,
    Task,
)


def _comparison_attempt_summary(attempt: dict[str, Any]) -> dict[str, Any]:
    """Render one attempt in a compact, comparison-friendly format."""

    return {
        "stage": attempt["stage"],
        "sql": attempt["sql"],
        "assumptions": attempt.get("assumptions", []),
        "constraint_ledger": attempt.get("constraint_ledger", []),
        "unsupported_assumptions": attempt.get("unsupported_assumptions", []),
        "candidate_confidence": attempt["candidate_confidence"],
        "score": attempt["score"],
        "score_breakdown": attempt.get("score_breakdown", {}),
        "validation": attempt["validation"],
        "execution_result": attempt["execution_result"],
        "filter_grounding_report": attempt.get("filter_grounding_report"),
        "shape_report": attempt.get("shape_report"),
        "result_profile": attempt.get("result_profile", {}),
    }


def _infer_native_value_terms(
    task: Task,
    schema: SchemaSelection,
    table_schemas: dict[str, TableSchema],
) -> list[str]:
    """Return exact sample-value matches that should stay tied to native columns."""

    question_text = _normalized_text(task.question)
    terms: list[str] = []
    for table_name in schema.expanded_tables:
        table = table_schemas.get(table_name)
        if table is None:
            continue
        table_identity = table.full_name or table.name
        for column in table.columns:
            if not _column_looks_string_like(column.type):
                continue
            for sample_value in column.sample_values:
                if not _question_mentions_literal(question_text, sample_value):
                    continue
                term = f"{table_identity}.{column.name}={sample_value}"
                if term not in terms:
                    terms.append(term)
    return terms


def _normalized_text(value: str) -> str:
    """Lower-case and collapse whitespace in one text fragment."""

    return " ".join(value.lower().split())


def _question_mentions_literal(question_text: str, literal: str) -> bool:
    """Return True when a question appears to name one stored literal value."""

    normalized_literal = _normalized_text(literal)
    if not normalized_literal:
        return False

    literal_tokens = re.findall(r"[a-z0-9]+", normalized_literal)
    if not literal_tokens:
        return False
    question_tokens = set(re.findall(r"[a-z0-9]+", question_text))
    if len(normalized_literal) < 3 or any(len(token) < 3 for token in literal_tokens):
        return all(token in question_tokens for token in literal_tokens)

    if normalized_literal in question_text:
        return True

    return all(token in question_tokens for token in literal_tokens)


def _column_looks_string_like(column_type: str | None) -> bool:
    """Return True when a schema column looks like a text field."""

    if column_type is None:
        return True
    lowered = column_type.lower()
    return any(token in lowered for token in ("char", "text", "string", "varchar", "variant"))


def _schema_context(schema: SchemaSelection) -> str:
    """Render a compact schema summary for prompt inputs."""

    return (
        f"DB: {schema.db}\n"
        f"Selected tables: {', '.join(schema.selected_tables)}\n"
        f"Expanded tables: {', '.join(schema.expanded_tables)}\n"
        f"Rationale: {schema.rationale}"
    )


def _sql_reference_context(
    schema: SchemaSelection,
    table_schemas: dict[str, TableSchema],
) -> str:
    """Render deterministic selected-table context for cache-friendly SQL prompts."""

    lines = [
        "SQL reference context:",
        f"Database: {schema.db}",
        "Selected tables:",
    ]
    for table_name in sorted(schema.expanded_tables):
        lines.append(f"- {table_name}")

    if not table_schemas:
        return "\n".join(lines)

    lines.append("")
    lines.append("Selected table details:")
    for table_name in sorted(table_schemas):
        table = table_schemas[table_name]
        lines.append(f"Table: {table.full_name or table_name}")
        if table.ddl.strip():
            lines.append("DDL:")
            lines.append("```sql")
            lines.append(table.ddl.strip())
            lines.append("```")
        if table.columns:
            lines.append("Columns:")
            for column in table.columns:
                lines.append(f"- {_column_context_line(column)}")
        if table.sample_rows:
            lines.append("Sample rows:")
            for row in table.sample_rows[:3]:
                lines.append(json.dumps(row, sort_keys=True))
        lines.append("")
    return "\n".join(lines).rstrip()


def _column_context_line(column: Any) -> str:
    """Render one compact column line with exact name, type, docs, and samples."""

    line = column.name
    if column.type:
        line += f" [{column.type}]"
    if column.description:
        line += f" - {column.description}"
    if column.sample_values:
        preview = ", ".join(column.sample_values[:3])
        line += f" - sample values: {preview}"
    return line


def _question_preview(question: str, *, max_length: int = 120) -> str:
    """Shorten long questions so task logs stay readable."""

    normalized = " ".join(question.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1].rstrip() + "…"


def _intent_user_prompt(
    task: Task,
    schema: SchemaSelection,
    docs_context: str,
    table_schemas: dict[str, TableSchema] | None = None,
) -> str:
    """Build the user prompt for intent extraction."""

    grounded_literals = _grounded_literal_context(task, schema, table_schemas or {})
    prompt = (
        f"Question: {task.question}\n\n"
        f"Document context:\n{docs_context}\n\n"
        f"Schema context:\n{_schema_context(schema)}"
    )
    if grounded_literals:
        prompt += f"\n\n{grounded_literals}"
    return prompt


def _grounded_literal_context(
    task: Task,
    schema: SchemaSelection,
    table_schemas: dict[str, TableSchema],
) -> str | None:
    """Render native sample-value matches for intent extraction and repair prompts."""

    native_value_terms = _infer_native_value_terms(task, schema, table_schemas)
    if not native_value_terms:
        return None
    lines = ["Grounded literal values:"]
    for term in native_value_terms:
        lines.append(f"- {term}")
    lines.append(
        "Use these as native column values. Do not recast them as behavioral definitions "
        "unless the question explicitly asks for that."
    )
    return "\n".join(lines)


def _grounded_literal_context_from_intent(intent: Intent) -> str | None:
    """Render grounded literals already attached to the answer contract."""

    if not intent.native_value_terms:
        return None
    lines = ["Grounded literal values:"]
    for term in intent.native_value_terms:
        lines.append(f"- {term}")
    lines.append(
        "Use these as native column values. Do not recast them as behavioral definitions "
        "unless the question explicitly asks for that."
    )
    return "\n".join(lines)


def _grain_guidance_block(guidance: str | None) -> str:
    """Render an optional grain hint as a prompt section."""

    if not guidance:
        return ""
    return f"Grain guidance:\n{guidance}\n\n"


def _metric_source_guidance_block(guidance: str | None) -> str:
    """Render optional metric source guidance as a prompt section."""

    if not guidance:
        return ""
    return f"Metric source guidance:\n{guidance}\n\n"


def _sql_generation_prompt(
    task: Task,
    intent: Intent,
    sql_reference_context: str,
    docs_context: str,
    *,
    aggregate_grain_guidance: str | None = None,
    metric_source_guidance: str | None = None,
) -> str:
    """Build the SQL-generation prompt body."""

    grounded_literals = _grounded_literal_context_from_intent(intent)
    grounded_literal_block = f"{grounded_literals}\n\n" if grounded_literals else ""
    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context}\n\n"
        f"Question: {task.question}\n\n"
        f"Intent:\n{intent.model_dump_json(indent=2)}\n\n"
        f"{grounded_literal_block}"
        f"{_grain_guidance_block(aggregate_grain_guidance)}"
        f"{_metric_source_guidance_block(metric_source_guidance)}"
        "Write the SQL using only the reference context above."
    )


def _sql_repair_prompt(
    task: Task,
    intent: Intent | None,
    attempt: dict[str, Any],
    sql_reference_context: str,
    docs_context: str,
    *,
    aggregate_grain_guidance: str | None = None,
    metric_source_guidance: str | None = None,
) -> str:
    """Build a repair prompt using validation or execution feedback."""

    grounded_literals = (
        _grounded_literal_context_from_intent(intent) if intent is not None else None
    )
    grounded_literal_block = f"{grounded_literals}\n\n" if grounded_literals else ""

    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context}\n\n"
        f"Question: {task.question}\n\n"
        f"Failed SQL:\n{attempt['sql']}\n\n"
        f"Validation:\n{json.dumps(attempt['validation'], indent=2, sort_keys=True)}\n\n"
        f"Execution:\n{json.dumps(attempt['execution_result'], indent=2, sort_keys=True)}\n\n"
        f"{grounded_literal_block}"
        f"{_grain_guidance_block(aggregate_grain_guidance)}"
        f"{_metric_source_guidance_block(metric_source_guidance)}"
    )


def _critic_prompt(
    task: Task,
    intent: Intent,
    attempt: dict[str, Any],
    sql_reference_context: str,
    docs_context: str,
    *,
    metric_source_guidance: str | None = None,
) -> str:
    """Build the critic prompt using the current best SQL and result profile."""

    grounded_literals = _grounded_literal_context_from_intent(intent)
    grounded_literal_block = f"{grounded_literals}\n\n" if grounded_literals else ""
    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context}\n\n"
        f"Question: {task.question}\n\n"
        f"Answer contract:\n{intent.model_dump_json(indent=2)}\n\n"
        f"{grounded_literal_block}"
        f"{_metric_source_guidance_block(metric_source_guidance)}"
        f"SQL:\n{attempt['sql']}\n\n"
        "Candidate assumptions:\n"
        f"{json.dumps(attempt.get('assumptions', []), indent=2, sort_keys=True)}\n\n"
        "Candidate constraint ledger:\n"
        f"{json.dumps(attempt.get('constraint_ledger', []), indent=2, sort_keys=True)}\n\n"
        "Candidate unsupported assumptions:\n"
        f"{json.dumps(attempt.get('unsupported_assumptions', []), indent=2, sort_keys=True)}\n\n"
        "Execution result:\n"
        f"{json.dumps(attempt.get('execution_result', {}), indent=2, sort_keys=True)}\n\n"
        "Result profile:\n"
        f"{json.dumps(attempt.get('result_profile', {}), indent=2, sort_keys=True)}"
    )


def _aggregate_verification_prompt(
    task: Task,
    attempt: dict[str, Any],
    sql_reference_context: str,
    docs_context: str,
    *,
    reason: str,
) -> str:
    """Build the verification prompt for suspicious aggregate outputs."""

    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context}\n\n"
        f"Question: {task.question}\n\n"
        f"Suspicion: {reason}\n\n"
        f"SQL:\n{attempt['sql']}\n\n"
        "Execution result:\n"
        f"{json.dumps(attempt['execution_result'], indent=2, sort_keys=True)}\n\n"
        "Result profile:\n"
        f"{json.dumps(attempt.get('result_profile', {}), indent=2, sort_keys=True)}\n\n"
        "Check whether the aggregate output is plausible.\n"
        "If the result looks too small, inspect nearby value variants, filter selectivity, "
        "and the grain of the aggregation.\n"
        "Recommend repair only when the result is not trustworthy."
    )


def _candidate_comparison_prompt(
    task: Task,
    intent: Intent,
    attempts: list[dict[str, Any]],
    sql_reference_context: str,
    docs_context: str,
    *,
    aggregate_grain_guidance: str | None = None,
    metric_source_guidance: str | None = None,
    baseline_stage: str | None,
) -> str:
    """Build the comparison prompt for executable candidates."""

    comparison_candidates = [_comparison_attempt_summary(attempt) for attempt in attempts]
    grounded_literals = _grounded_literal_context_from_intent(intent)
    grounded_literal_block = f"{grounded_literals}\n\n" if grounded_literals else ""
    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context}\n\n"
        f"Question: {task.question}\n\n"
        f"Intent:\n{intent.model_dump_json(indent=2)}\n\n"
        f"{grounded_literal_block}"
        f"{_grain_guidance_block(aggregate_grain_guidance)}"
        f"{_metric_source_guidance_block(metric_source_guidance)}"
        f"Baseline stage: {baseline_stage or 'unknown'}\n\n"
        "Executable candidates:\n"
        f"{json.dumps(comparison_candidates, indent=2, sort_keys=True)}\n\n"
        "Compare every executable candidate above and choose the one "
        "that best fits the answer contract."
    )


def _aggregate_repair_prompt(
    task: Task,
    attempt: dict[str, Any],
    verification: ConfidenceReport,
    sql_reference_context: str,
    docs_context: str,
    intent: Intent | None = None,
) -> str:
    """Build the repair prompt after aggregate verification fails."""

    verification_json = json.dumps(verification.model_dump(mode="json"), indent=2, sort_keys=True)
    execution_json = json.dumps(attempt["execution_result"], indent=2, sort_keys=True)
    profile_json = json.dumps(attempt.get("result_profile", {}), indent=2, sort_keys=True)
    grounded_literals = (
        _grounded_literal_context_from_intent(intent) if intent is not None else None
    )
    grounded_literal_block = f"{grounded_literals}\n\n" if grounded_literals else ""
    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context}\n\n"
        f"Question: {task.question}\n\n"
        f"Current SQL:\n{attempt['sql']}\n\n"
        f"Verification:\n{verification_json}\n\n"
        f"Execution result:\n{execution_json}\n\n"
        f"Result profile:\n{profile_json}\n\n"
        f"{grounded_literal_block}"
    )


def _semantic_repair_prompt(
    task: Task,
    intent: Intent,
    attempt: dict[str, Any],
    critic: ConfidenceReport,
    sql_reference_context: str,
    docs_context: str,
    *,
    metric_source_guidance: str | None = None,
) -> str:
    """Build the repair prompt for one critic-triggered retry."""

    grounded_literals = _grounded_literal_context_from_intent(intent)
    grounded_literal_block = f"{grounded_literals}\n\n" if grounded_literals else ""
    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context}\n\n"
        f"Question: {task.question}\n\n"
        f"Current answer contract:\n{intent.model_dump_json(indent=2)}\n\n"
        f"{grounded_literal_block}"
        f"{_metric_source_guidance_block(metric_source_guidance)}"
        f"Current SQL:\n{attempt['sql']}\n\n"
        "Candidate assumptions:\n"
        f"{json.dumps(attempt.get('assumptions', []), indent=2, sort_keys=True)}\n\n"
        "Candidate constraint ledger:\n"
        f"{json.dumps(attempt.get('constraint_ledger', []), indent=2, sort_keys=True)}\n\n"
        "Candidate unsupported assumptions:\n"
        f"{json.dumps(attempt.get('unsupported_assumptions', []), indent=2, sort_keys=True)}\n\n"
        f"Critic issues:\n{json.dumps(critic.model_dump(mode='json'), indent=2, sort_keys=True)}"
    )
