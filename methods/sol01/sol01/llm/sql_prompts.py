"""SQL generation, repair, and review prompt assembly."""

from __future__ import annotations

import json
import re
from typing import Any

from sol01.infra.strings import column_looks_string_like
from sol01.models import (
    AttemptRecord,
    ConfidenceReport,
    Intent,
    SchemaSelection,
    TableSchema,
    Task,
)


def infer_native_value_terms(
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
            if not column_looks_string_like(column.type):
                continue
            for sample_value in column.sample_values:
                if not _question_mentions_literal(question_text, sample_value):
                    continue
                term = f"{table_identity}.{column.name}={sample_value}"
                if term not in terms:
                    terms.append(term)
    return terms


def sql_generation_prompt(
    task: Task,
    intent: Intent,
    sql_reference_context: str,
    docs_context: str | None,
) -> str:
    """Build the SQL-generation prompt body."""

    grounded_literals = _grounded_literal_context_from_intent(intent)
    grounded_literal_block = f"{grounded_literals}\n\n" if grounded_literals else ""
    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context or 'No task-linked document context.'}\n\n"
        f"Question: {task.question}\n\n"
        f"Intent:\n{intent.model_dump_json(indent=2)}\n\n"
        f"{grounded_literal_block}"
        "Write the SQL using only the reference context above."
    )


def sql_generation_batch_prompt(
    task: Task,
    intent: Intent,
    sql_reference_context: str,
    docs_context: str | None,
    *,
    candidate_count: int,
) -> str:
    """Build one prompt that asks for multiple candidate SQL queries."""

    base_prompt = sql_generation_prompt(
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


def sql_repair_prompt(
    task: Task,
    intent: Intent | None,
    attempt: AttemptRecord,
    sql_reference_context: str,
    docs_context: str | None,
) -> str:
    """Build a repair prompt using validation or execution feedback."""

    grounded_literals = (
        _grounded_literal_context_from_intent(intent) if intent is not None else None
    )
    grounded_literal_block = f"{grounded_literals}\n\n" if grounded_literals else ""

    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context or 'No task-linked document context.'}\n\n"
        f"Question: {task.question}\n\n"
        f"Failed SQL:\n{attempt.sql}\n\n"
        f"Validation:\n"
        f"{json.dumps(attempt.validation.model_dump(mode='json'), indent=2, sort_keys=True)}"
        "\n\n"
        f"Execution:\n"
        f"{json.dumps(attempt.execution_result.model_dump(mode='json'), indent=2, sort_keys=True)}"
        "\n\n"
        f"{grounded_literal_block}"
    )


def candidate_review_prompt(
    task: Task,
    intent: Intent,
    attempts: list[AttemptRecord],
    sql_reference_context: str,
    docs_context: str | None,
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
        f"Document context:\n{docs_context or 'No task-linked document context.'}\n\n"
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


def semantic_repair_prompt(
    task: Task,
    intent: Intent,
    attempt: AttemptRecord,
    critic: ConfidenceReport,
    sql_reference_context: str,
    docs_context: str | None,
) -> str:
    """Build the repair prompt for one critic-triggered retry."""

    grounded_literals = _grounded_literal_context_from_intent(intent)
    grounded_literal_block = f"{grounded_literals}\n\n" if grounded_literals else ""
    return (
        f"{sql_reference_context}\n\n"
        f"Document context:\n{docs_context or 'No task-linked document context.'}\n\n"
        f"Question: {task.question}\n\n"
        f"Current answer contract:\n{intent.model_dump_json(indent=2)}\n\n"
        f"{grounded_literal_block}"
        f"Current SQL:\n{attempt.sql}\n\n"
        "Candidate assumptions:\n"
        f"{json.dumps(attempt.assumptions, indent=2, sort_keys=True)}\n\n"
        "Candidate constraint ledger:\n"
        f"{json.dumps(attempt.constraint_ledger, indent=2, sort_keys=True)}\n\n"
        "Candidate unsupported assumptions:\n"
        f"{json.dumps(attempt.unsupported_assumptions, indent=2, sort_keys=True)}\n\n"
        f"Critic issues:\n{json.dumps(critic.model_dump(mode='json'), indent=2, sort_keys=True)}"
    )


def _comparison_attempt_summary(attempt: AttemptRecord) -> dict[str, Any]:
    """Render one attempt in a compact, comparison-friendly format."""

    return {
        "stage": attempt.stage,
        "sql": attempt.sql,
        "assumptions": attempt.assumptions,
        "constraint_ledger": attempt.constraint_ledger,
        "unsupported_assumptions": attempt.unsupported_assumptions,
        "candidate_confidence": attempt.candidate_confidence,
        "evidence": (
            attempt.evidence.model_dump(mode="json") if attempt.evidence is not None else {}
        ),
        "validation": attempt.validation.model_dump(mode="json"),
        "execution_result": attempt.execution_result.model_dump(mode="json"),
        "filter_grounding_report": (
            attempt.filter_grounding_report.model_dump(mode="json")
            if attempt.filter_grounding_report is not None
            else None
        ),
        "shape_report": (
            attempt.shape_report.model_dump(mode="json")
            if attempt.shape_report is not None
            else None
        ),
        "result_profile": attempt.result_profile or {},
    }


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
