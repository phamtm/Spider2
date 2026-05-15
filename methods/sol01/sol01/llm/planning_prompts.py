"""Planner prompt assembly and planner-output sanitizing."""

from __future__ import annotations

import json
from collections.abc import Sequence

from sol01.infra.config import SchemaContextConfig
from sol01.infra.policy import DEFAULT_PROMPT_BUDGET_POLICY
from sol01.llm.prompt_budget import fit_prompt_budget
from sol01.models import (
    SchemaContextObject,
    SchemaPlanningDecision,
    SelectedSchemaObject,
    TableSchema,
    Task,
)
from sol01.schema.exact_reference_context import render_exact_sql_reference_context
from sol01.schema.schema_profiles import load_schema_profile_catalog


def sql_reference_context(
    schema,
    table_schemas: dict[str, TableSchema],
) -> str:
    """Render deterministic selected-table context for cache-friendly SQL prompts."""

    return render_exact_sql_reference_context(
        db=schema.db,
        expanded_tables=schema.expanded_tables,
        table_schemas=table_schemas,
        schema_profile_catalog=load_schema_profile_catalog(schema.db),
    )


def schema_context_planning_user_prompt(
    task: Task,
    db: str,
    docs_context: str | None,
    schema_context_objects: Sequence[SchemaContextObject],
    *,
    schema_context_config: SchemaContextConfig | None = None,
    max_docs_chars: int | None = None,
    max_evidence_chars: int | None = None,
    max_total_chars: int | None = None,
) -> str:
    """Build a planner prompt from deterministic schema metadata objects."""

    config = schema_context_config or SchemaContextConfig()
    docs_limit = max(
        0,
        config.max_linked_doc_chars if max_docs_chars is None else max_docs_chars,
    )
    evidence_limit = max(
        0,
        config.planning_evidence_chars if max_evidence_chars is None else max_evidence_chars,
    )
    total_limit = config.max_schema_prompt_chars if max_total_chars is None else max_total_chars

    def render_prompt(docs_chars: int, evidence_chars: int) -> str:
        return _format_schema_context_planning_user_prompt(
            task,
            db,
            docs_context,
            schema_context_objects,
            max_docs_chars=docs_chars,
            max_evidence_chars=evidence_chars,
        )

    fit = fit_prompt_budget(
        prompt_name="planning",
        total_limit=total_limit,
        docs_limit=docs_limit,
        evidence_limit=evidence_limit,
        render_prompt=render_prompt,
    )
    return fit.prompt


def sanitize_schema_planning_decision(
    decision: SchemaPlanningDecision,
    schema_context_objects: Sequence[SchemaContextObject],
) -> tuple[SchemaPlanningDecision, dict[str, object]]:
    """Drop hallucinated object ids and reject duplicates."""

    available_ids = [item.schema_object.object_id for item in schema_context_objects]
    available_id_set = set(available_ids)

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

    diagnostics = {
        "available_object_count": len(available_ids),
        "selected_object_count": len(selected),
        "rejected_object_ids": rejected_object_ids,
        "duplicate_object_ids": duplicate_object_ids,
    }
    confidence = decision.confidence if selected else 0.0
    rationale = decision.rationale.strip()
    if rejected_object_ids:
        rationale = (f"{rationale} Ignored ids outside available schema metadata.").strip()
    if duplicate_object_ids:
        rationale = f"{rationale} Ignored duplicate selections.".strip()
    if not selected:
        rationale = f"{rationale} No valid schema objects were selected.".strip()

    sanitized = decision.model_copy(
        update={
            "selected_objects": selected,
            "confidence": confidence,
            "rationale": rationale,
        }
    )
    return sanitized, diagnostics


def _format_schema_context_planning_user_prompt(
    task: Task,
    db: str,
    docs_context: str | None,
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
    effective_docs = docs_context or "No task-linked document context."
    return (
        f"Question: {task.question}\n\n"
        f"Database: {db}\n\n"
        f"Document context:\n{_clip_context(effective_docs, max_docs_chars)}\n\n"
        "Available schema metadata evidence:\n"
        f"{evidence}\n\n"
        "Available object ids:\n"
        f"{json.dumps(available_ids, indent=2)}\n\n"
        "Select only logical schema objects from the planner-visible schema metadata above. "
        "This is already the complete selector input for this database: either generated "
        "schema profiles or the full logical metadata object set. "
        "Do not invent object ids, table names, column names, joins, families, suffixes, "
        "versions, or date constraints that are not grounded in the question, documents, "
        "or planner-visible schema metadata. "
        "When compact schema profiles are present, rely on them "
        "instead of raw wide-schema DDL.\n\n"
        "Return a SchemaPlanningDecision. Populate selected_objects with object ids from "
        "Available object ids and roles such as primary, supporting, join, filter, metric, "
        "dimension, or unknown. Populate constraints with any grounded date_start, date_end, "
        "years, suffixes, version, include_all, and notes. Include rationale, confidence, "
        "and an intent answer contract."
    )


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

        evidence_lines = [header]
        if item.planning_text:
            evidence_lines.append(
                "  evidence: "
                + _single_line(
                    item.planning_text,
                    max_length=DEFAULT_PROMPT_BUDGET_POLICY.evidence_line_max_chars,
                )
            )
        rendered = "\n".join(evidence_lines)
        if len(rendered) + 2 > remaining:
            break
        lines.append(rendered)
        remaining -= len(rendered) + 2

    return "\n\n".join(lines) if lines else "(no schema context objects)"


def _single_line(
    text: str,
    *,
    max_length: int = DEFAULT_PROMPT_BUDGET_POLICY.evidence_line_max_chars,
) -> str:
    """Collapse whitespace and clip one evidence line."""

    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."
