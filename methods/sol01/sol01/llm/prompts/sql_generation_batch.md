# SQL Generation Batch

Write several Snowflake query candidates that answer the question.

Rules for every candidate:
- return one read-only query
- use Snowflake SQL syntax
- use column identifiers exactly as shown in the DDL/schema context
- quote columns that are quoted in the DDL; do not write bare mixed-case or lower-case column names
- preserve fully qualified table names when they appear in the schema context
- when a grouped entity has a stable identifier and a display label, keep both in the SELECT and GROUP BY unless the question explicitly asks to omit the identifier
- if the answer contract includes `native_value_terms`, keep those as native column values and do not turn them into behavioral definitions
- when answering max/min/top/bottom over grouped counts or metrics, return the winning group key plus the metric; do not collapse to only MAX(metric) unless the answer contract explicitly asks for only the scalar value
- prefer clear joins and explicit column names
- stay within the provided schema context
- use the answer contract as the task boundary
- choose the metric source at the requested answer grain
- do not add filters, current/latest rules, dedupe rules, status rules, limits, or row narrowing that the contract does not ground
- record every row-narrowing, dedupe, ordering, and top-k choice in the constraint ledger, with its grounding

Return genuinely different candidates only when there are plausible alternatives in grain, metric source, or join path.
