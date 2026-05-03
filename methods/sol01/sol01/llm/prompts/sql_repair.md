# SQL Repair

Repair the SQL using the validation or execution feedback.

Keep the original question and document context as the source of truth.
For validation or execution repairs, preserve the answer contract.
For critic or semantic repairs, re-check whether the current contract or SQL preserved an unsupported assumption.
Change only what is needed to make the Snowflake query valid and correct.
Use column identifiers exactly as shown in the DDL/schema context.
Quote columns that are quoted in the DDL; do not leave mixed-case or lower-case columns bare.
- when a grouped entity has a stable identifier and a display label, keep both in the SELECT and GROUP BY unless the question explicitly asks to omit the identifier
- when repairing max/min/top/bottom over grouped counts or metrics, preserve the winning group key plus the metric; do not collapse to only MAX(metric) unless the answer contract explicitly asks for only the scalar value
- remove filters, current/latest rules, dedupe rules, status rules, limits, or row narrowing when the repair feedback says they are not grounded
- if the answer contract includes `native_value_terms`, keep those values tied to the named column and do not rewrite them as a behavioral definition
- for metric-source repairs, prefer a native metric column at the requested answer grain only when it is clearly grounded in the answer contract or its semantics unambiguously match the question; keep lower-grain detail joins when no clearly grounded native metric exists
- when several native metric columns exist, choose by column-name semantics from the question; do not treat subtotal, total due, tax, freight, or line-item formulas as interchangeable
