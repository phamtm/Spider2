# SQL Generation

Write one Snowflake query that answers the question.

Rules:
- return one read-only query
- use Snowflake SQL syntax
- use column identifiers exactly as shown in the DDL/schema context
- quote columns that are quoted in the DDL; do not write bare mixed-case or lower-case column names
- preserve fully qualified table names when they appear in the schema context
- when a grouped entity has a stable identifier and a display label, keep both in the SELECT and GROUP BY unless the question explicitly asks to omit the identifier
- if the answer contract includes `native_value_terms`, keep those as native column values and do not turn them into behavioral definitions
- when answering max/min/top/bottom over grouped counts or metrics, return the winning group key plus the metric; do not collapse to only MAX(metric) unless the answer contract explicitly asks for only the scalar value
- prefer clear joins and explicit column names
- avoid unnecessary complexity
- stay within the provided schema context
- use the answer contract as the task boundary
- prefer the simplest query that satisfies the contract
- do not add filters, current/latest rules, dedupe rules, status rules, limits, or row narrowing that the contract does not ground
- record every row-narrowing, dedupe, ordering, and top-k choice in the constraint ledger, with its grounding
- place any useful but ungrounded choice in unsupported_assumptions instead of silently applying it
