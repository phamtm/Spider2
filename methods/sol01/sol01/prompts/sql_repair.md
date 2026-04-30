# SQL Repair

Repair the SQL using the validation or execution feedback.

Keep the intent the same.
Change only what is needed to make the Snowflake query valid and correct.
Use column identifiers exactly as shown in the DDL/schema context.
Quote columns that are quoted in the DDL; do not leave mixed-case or lower-case columns bare.
- when a grouped entity has a stable identifier and a display label, keep both in the SELECT and GROUP BY unless the question explicitly asks to omit the identifier
