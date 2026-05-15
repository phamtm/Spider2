# Schema Grounding

Ground exact schema-facing terms against the already selected tables.

Rules:
- use only exact table names and exact column names that appear in the SQL reference context
- do not invent tables, columns, joins, metrics, aliases, or rewritten schema terms
- do not expand schema scope or suggest new tables
- if a requested term is not exactly supported by the selected-table metadata, return it in unresolved_terms
- prefer the closest exact physical column even when the requested wording is slightly different
- metrics may bind to more than one exact column when the question implies a derived calculation
- preserve requested_term wording from the prompt so the caller can trace what was or was not grounded

Return a SchemaGrounding object with:
- bindings: exact requested_term -> table_name + column_name pairs
- unresolved_terms: requested terms that could not be grounded exactly
- warnings: any short cautions that the SQL generator should know
