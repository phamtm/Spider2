"""Build coordinator prompts and prompt-adjacent context blocks."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from sol01.models import (
    ConfidenceReport,
    Intent,
    SchemaContextObject,
    SchemaPlanningDecision,
    SchemaSelection,
    SelectedSchemaObject,
    TableSchema,
    Task,
)
from sol01.schema.reference_context import render_sql_reference_context

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
_SCHEMA_CONTEXT_PLANNING_DOCS_CHAR_LIMIT = 6000
_SCHEMA_CONTEXT_PLANNING_EVIDENCE_CHAR_LIMIT = 16000


class PromptBudgetExceededError(ValueError):
    """Raised when a prompt cannot fit inside the configured character budget."""


def enforce_prompt_budget(prompt_name: str, prompt: str, max_chars: int) -> str:
    """Return a prompt only when it fits the configured character budget."""

    if len(prompt) > max_chars:
        raise PromptBudgetExceededError(
            f"{prompt_name} prompt is {len(prompt)} chars; configured budget is {max_chars}"
        )
    return prompt


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

    return render_sql_reference_context(
        db=schema.db,
        expanded_tables=schema.expanded_tables,
        table_schemas=table_schemas,
    )


def _question_preview(question: str, *, max_length: int = 120) -> str:
    """Shorten long questions so task logs stay readable."""

    normalized = " ".join(question.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1].rstrip() + "…"


def _schema_context_planning_user_prompt(
    task: Task,
    db: str,
    docs_context: str,
    schema_context_objects: Sequence[SchemaContextObject],
    *,
    max_docs_chars: int = _SCHEMA_CONTEXT_PLANNING_DOCS_CHAR_LIMIT,
    max_evidence_chars: int = _SCHEMA_CONTEXT_PLANNING_EVIDENCE_CHAR_LIMIT,
    max_total_chars: int | None = None,
) -> str:
    """Build a planner prompt from deterministic schema metadata objects."""

    docs_limit = max(0, max_docs_chars)
    evidence_limit = max(0, max_evidence_chars)
    prompt = _format_schema_context_planning_user_prompt(
        task,
        db,
        docs_context,
        schema_context_objects,
        max_docs_chars=docs_limit,
        max_evidence_chars=evidence_limit,
    )
    if max_total_chars is None or len(prompt) <= max_total_chars:
        return prompt

    # Keep exact object ids available, then shrink lossy evidence before docs.
    for _ in range(12):
        overflow = len(prompt) - max_total_chars
        if evidence_limit > 0:
            evidence_limit = max(0, evidence_limit - overflow - 256)
        elif docs_limit > 0:
            docs_limit = max(0, docs_limit - overflow - 256)
        else:
            break
        prompt = _format_schema_context_planning_user_prompt(
            task,
            db,
            docs_context,
            schema_context_objects,
            max_docs_chars=docs_limit,
            max_evidence_chars=evidence_limit,
        )
        if len(prompt) <= max_total_chars:
            return prompt

    return enforce_prompt_budget("planning", prompt, max_total_chars)


def _format_schema_context_planning_user_prompt(
    task: Task,
    db: str,
    docs_context: str,
    schema_context_objects: Sequence[SchemaContextObject],
    *,
    max_docs_chars: int,
    max_evidence_chars: int,
) -> str:
    """Render the planner prompt once with fixed context limits."""

    available_ids = [item.schema_object.object_id for item in schema_context_objects]
    evidence = _schema_context_object_evidence(
        schema_context_objects,
        max_chars=max_evidence_chars,
    )
    return (
        f"Question: {task.question}\n\n"
        f"Database: {db}\n\n"
        f"Document context:\n{_clip_context(docs_context, max_docs_chars)}\n\n"
        "Available schema metadata evidence:\n"
        f"{evidence}\n\n"
        "Available object ids:\n"
        f"{json.dumps(available_ids, indent=2)}\n\n"
        "Select only logical schema objects from the available schema metadata above. "
        "Do not invent object ids, table names, column names, joins, families, suffixes, "
        "versions, or date constraints that are not grounded in the question, documents, "
        "or available schema metadata. For large schemas, rely on curated summaries instead "
        "of raw wide-schema DDL.\n\n"
        "Return a SchemaPlanningDecision. Populate selected_objects with object ids from "
        "Available object ids and roles such as primary, supporting, join, filter, metric, "
        "dimension, or unknown. Populate constraints with any grounded date_start, date_end, "
        "years, suffixes, version, include_all, and notes. Include rationale, confidence, "
        "and an intent answer contract."
    )


def sanitize_schema_planning_decision(
    decision: SchemaPlanningDecision,
    schema_context_objects: Sequence[SchemaContextObject],
) -> tuple[SchemaPlanningDecision, dict[str, object]]:
    """Drop hallucinated object ids and normalize exact available table names."""

    available_ids = [item.schema_object.object_id for item in schema_context_objects]
    available_id_set = set(available_ids)
    exact_table_ids = {
        item.schema_object.table_name: item.schema_object.object_id
        for item in schema_context_objects
        if item.schema_object.object_type == "table" and item.schema_object.table_name
    }

    selected: list[SelectedSchemaObject] = []
    seen_ids: set[str] = set()
    rejected_object_ids: list[str] = []
    duplicate_object_ids: list[str] = []

    for selected_object in decision.selected_objects:
        object_id = selected_object.object_id
        if object_id not in available_id_set:
            rejected_object_ids.append(object_id)
            continue
        if object_id in seen_ids:
            duplicate_object_ids.append(object_id)
            continue
        selected.append(selected_object)
        seen_ids.add(object_id)

    normalized_table_names: list[str] = []
    rejected_table_names: list[str] = []
    for table_name in decision.selected_tables:
        object_id = exact_table_ids.get(table_name)
        if object_id is None:
            rejected_table_names.append(table_name)
            continue
        if object_id in seen_ids:
            duplicate_object_ids.append(object_id)
            continue
        selected.append(SelectedSchemaObject(object_id=object_id, role="unknown"))
        seen_ids.add(object_id)
        normalized_table_names.append(table_name)

    diagnostics = {
        "available_object_count": len(available_ids),
        "selected_object_count": len(selected),
        "rejected_object_ids": rejected_object_ids,
        "duplicate_object_ids": duplicate_object_ids,
        "normalized_table_names": normalized_table_names,
        "rejected_table_names": rejected_table_names,
    }
    confidence = decision.confidence if selected else 0.0
    rationale = decision.rationale.strip()
    if rejected_object_ids or rejected_table_names:
        rationale = (
            f"{rationale} Ignored ids or table names outside available schema metadata."
        ).strip()
    if duplicate_object_ids:
        rationale = f"{rationale} Ignored duplicate selections.".strip()
    if not selected:
        rationale = f"{rationale} No valid schema objects were selected.".strip()

    sanitized = decision.model_copy(
        update={
            "selected_objects": selected,
            "selected_tables": [],
            "confidence": confidence,
            "rationale": rationale,
        }
    )
    return sanitized, diagnostics


def _clip_context(text: str, max_chars: int) -> str:
    """Clip context text on a word boundary when possible."""

    normalized = text.strip()
    if max_chars < 1:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    clipped = normalized[:max_chars].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0].rstrip()
    return f"{clipped}..."


def _schema_context_object_evidence(
    schema_context_objects: Sequence[SchemaContextObject],
    *,
    max_chars: int,
) -> str:
    """Render schema object evidence with prompt-size bounds."""

    lines: list[str] = []
    remaining = max_chars
    for item in schema_context_objects:
        schema_object = item.schema_object
        header = (
            f"- id: {schema_object.object_id}\n"
            f"  type: {schema_object.object_type}\n"
            f"  name: {schema_object.name}"
        )
        if schema_object.table_name:
            header += f"\n  table: {schema_object.table_name}"
        if schema_object.column_name:
            header += f"\n  column: {schema_object.column_name}"
        if schema_object.description:
            header += f"\n  description: {_single_line(schema_object.description)}"
        if item.score is not None:
            header += f"\n  score: {item.score:.4f}"

        evidence_lines = [header]
        for context_chunk in item.chunks[:3]:
            chunk = context_chunk.chunk
            text = (
                chunk.prompt_text or chunk.source_definition or chunk.inferred_usage or chunk.text
            )
            if text:
                evidence_lines.append(f"  evidence: {_single_line(text, max_length=500)}")
        rendered = "\n".join(evidence_lines)
        if len(rendered) + 2 > remaining:
            break
        lines.append(rendered)
        remaining -= len(rendered) + 2

    return "\n\n".join(lines) if lines else "(no schema context objects)"


def _single_line(text: str, *, max_length: int = 500) -> str:
    """Collapse whitespace and clip one evidence line."""

    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."


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
        error_lower = error.lower()
        if "unknown table referenced" in error_lower:
            return f"validation_unknown_table: {error}"
        if "unknown column" in error_lower:
            return f"validation_unknown_column: {error}"

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
