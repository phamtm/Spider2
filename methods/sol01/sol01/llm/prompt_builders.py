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

# Execution error substrings that suggest a missing table in the schema.
_EXEC_TABLE_MISSING_SUBSTRINGS = (
    "does not exist",
    "invalid identifier",
    "object does not exist",
    "table not found",
    "unknown table",
    "002003",
    "000904",
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


def _planning_user_prompt(
    task: Task,
    db: str,
    docs_context: str,
    db_schema_summary: str,
) -> str:
    """Build the combined schema-selection and intent-extraction prompt."""

    return (
        f"Question: {task.question}\n\n"
        f"Database: {db}\n\n"
        f"Document context:\n{docs_context}\n\n"
        "Include all tables that are plausibly required to answer the question, "
        "including join and bridge tables. Omit only tables that are clearly irrelevant.\n\n"
        "For metric questions, include tables at every grain that may be needed. "
        "If one table already has the needed grouping keys, time key, filters, and a "
        "native metric column that is clearly grounded in the answer contract or whose "
        "semantics unambiguously match the question, that table is the preferred metric "
        "source; also include lower-grain detail tables when the question may need "
        "detail-level filters, grouping, output columns, an explicit formula, or when "
        "no clearly grounded native metric exists.\n\n"
        "Create the answer contract from the question and document context. Do not invent "
        "filters, current/latest rules, dedupe rules, status rules, or metric formulas that "
        "are not grounded.\n\n"
        f"Schema summary:\n{db_schema_summary}"
    )


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


def _sql_generation_prompt(
    task: Task,
    intent: Intent,
    sql_reference_context: str,
    docs_context: str,
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
        "Write the SQL using only the reference context above."
    )


def _sql_generation_batch_prompt(
    task: Task,
    intent: Intent,
    sql_reference_context: str,
    docs_context: str,
    *,
    candidate_count: int,
) -> str:
    """Build one prompt that asks for multiple candidate SQL queries."""

    base_prompt = _sql_generation_prompt(
        task,
        intent,
        sql_reference_context,
        docs_context,
    )
    return (
        f"{base_prompt}\n\n"
        f"Return exactly {candidate_count} meaningfully different SQL candidate(s) when "
        "there are genuine alternatives. Keep each candidate independently executable."
    )


def _sql_repair_prompt(
    task: Task,
    intent: Intent | None,
    attempt: dict[str, Any],
    sql_reference_context: str,
    docs_context: str,
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
    )


def _candidate_review_prompt(
    task: Task,
    intent: Intent,
    attempts: list[dict[str, Any]],
    sql_reference_context: str,
    docs_context: str,
    *,
    baseline_stage: str | None,
    review_reason: str,
) -> str:
    """Build the unified comparison and critic prompt."""

    comparison_candidates = [_comparison_attempt_summary(attempt) for attempt in attempts]
    grounded_literals = _grounded_literal_context_from_intent(intent)
    grounded_literal_block = f"{grounded_literals}\n\n" if grounded_literals else ""
    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context}\n\n"
        f"Question: {task.question}\n\n"
        f"Intent:\n{intent.model_dump_json(indent=2)}\n\n"
        f"{grounded_literal_block}"
        f"Baseline stage: {baseline_stage or 'unknown'}\n"
        f"Review reason: {review_reason}\n\n"
        "Executable candidates:\n"
        f"{json.dumps(comparison_candidates, indent=2, sort_keys=True)}\n\n"
        "Use local scores and verification reports as evidence, not as the final decision. "
        "Pick the candidate that best answers the contract, then decide whether that "
        "preferred candidate still needs repair. Consider wrong shape, missing or "
        "ungrounded filters, suspicious aggregations including tiny aggregate results, "
        "native value mismatches, metric-source mistakes, and unsupported assumptions. "
        "Recommend repair only for a concrete issue."
    )


def _semantic_repair_prompt(
    task: Task,
    intent: Intent,
    attempt: dict[str, Any],
    critic: ConfidenceReport,
    sql_reference_context: str,
    docs_context: str,
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
        f"Current SQL:\n{attempt['sql']}\n\n"
        "Candidate assumptions:\n"
        f"{json.dumps(attempt.get('assumptions', []), indent=2, sort_keys=True)}\n\n"
        "Candidate constraint ledger:\n"
        f"{json.dumps(attempt.get('constraint_ledger', []), indent=2, sort_keys=True)}\n\n"
        "Candidate unsupported assumptions:\n"
        f"{json.dumps(attempt.get('unsupported_assumptions', []), indent=2, sort_keys=True)}\n\n"
        f"Critic issues:\n{json.dumps(critic.model_dump(mode='json'), indent=2, sort_keys=True)}"
    )


def schema_expansion_trigger(attempt: dict[str, Any]) -> str | None:
    """Return why schema expansion is warranted for this attempt, or None.

    Only returns a non-None value when there is explicit evidence that the
    selected schema is incomplete: an unknown-table validation error, a
    column-absent-from-selected-tables validation warning, an
    object/table-not-found execution error, or a critic repair_focus that
    names schema_selection or missing_join.
    """
    validation = attempt.get("validation", {})
    for error in validation.get("errors", []):
        if "unknown table referenced" in error.lower():
            return f"validation_unknown_table: {error}"

    for warning in validation.get("warnings", []):
        if "no selected table has it" in warning.lower():
            return f"validation_missing_column: {warning}"

    exec_error = (attempt.get("execution_result", {}).get("error") or "").lower()
    if exec_error and any(t in exec_error for t in _EXEC_TABLE_MISSING_SUBSTRINGS):
        raw_error = attempt.get("execution_result", {}).get("error") or ""
        return f"execution_error: {raw_error[:300]}"

    critic = attempt.get("critic") or {}
    repair_focus = (critic.get("repair_focus") or "").lower()
    if "schema_selection" in repair_focus or "missing_join" in repair_focus:
        return f"critic_repair_focus: {critic.get('repair_focus')}"
    for issue in critic.get("issues", []):
        issue_lower = issue.lower()
        if any(kw in issue_lower for kw in ("schema_selection", "missing_join", "missing table")):
            return f"critic_issue: {issue}"

    return None


def _schema_expansion_prompt(
    task: Task,
    attempt: dict[str, Any],
    trigger: str,
    schema: SchemaSelection,
    db_schema_summary: str,
) -> str:
    """Build the prompt that asks the LLM which tables to add."""

    return (
        f"Question: {task.question}\n\n"
        f"Current selected tables: {', '.join(schema.expanded_tables)}\n\n"
        f"Evidence that the schema may be incomplete:\n{trigger}\n\n"
        f"Failed SQL:\n{attempt['sql']}\n\n"
        f"Validation:\n{json.dumps(attempt['validation'], indent=2, sort_keys=True)}\n\n"
        f"Execution error: {attempt.get('execution_result', {}).get('error') or 'none'}\n\n"
        f"All available tables in this database:\n{db_schema_summary}\n\n"
        "List only tables that genuinely fix the identified gap and exist in the database above. "
        "If the current schema already covers every table needed, set should_expand=false."
    )
