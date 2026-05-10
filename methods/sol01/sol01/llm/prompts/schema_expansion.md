# Schema Expansion

Decide whether the current table selection is missing a table needed to answer the question.

Use the evidence provided (validation error, execution error, or critic finding) to identify the gap.

Rules:
- Set `should_expand=true` only when a specific table from the database is clearly needed and absent.
- Name only tables that exist in the provided database summary.
- Do not remove or replace tables already in the current selection.
- If the evidence is ambiguous or the current schema is sufficient, set `should_expand=false`.
- Prefer bridge and join tables when the evidence points to a missing relationship.
