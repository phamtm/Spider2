# SQL Repair

Repair the SQL using the validation or execution feedback.

Keep the original question and document context as the source of truth.
For validation or execution repairs, preserve the answer contract.
For critic or semantic repairs, re-check whether the current contract or SQL preserved an unsupported assumption.
Change only what is needed to make the Snowflake query valid and correct.
Use column identifiers exactly as shown in the DDL/schema context.
Quote columns that are quoted in the DDL; do not leave mixed-case or lower-case columns bare.
- when a grouped entity has a stable identifier and a display label, keep both in the SELECT and GROUP BY unless the question explicitly asks to omit the identifier
- remove filters, current/latest rules, dedupe rules, status rules, limits, or row narrowing when the repair feedback says they are not grounded
