"""Prompt assembly for exact selected-table schema grounding."""

from __future__ import annotations

import json
from collections.abc import Sequence

from sol01.models import Intent, Task


def schema_grounding_user_prompt(
    task: Task,
    intent: Intent,
    sql_prompt_context: str,
    requested_terms: Sequence[dict[str, object]],
) -> str:
    """Build the schema-grounding prompt body."""

    return (
        f"{sql_prompt_context}\n\n"
        f"Question: {task.question}\n\n"
        f"Intent:\n{intent.model_dump_json(indent=2)}\n\n"
        "Requested schema-facing terms:\n"
        f"{json.dumps(list(requested_terms), indent=2)}\n\n"
        "Bind requested terms only to exact table and column names shown in the SQL "
        "reference context above. If a requested term does not have an exact selected-table "
        "column match, return it as unresolved instead of guessing."
    )
