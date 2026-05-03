# Schema Selection

Choose the smallest useful set of tables for the question.

Use:
- the question
- the retrieved schema candidates
- any metric or document context already provided

Prefer the tables that are directly needed for joins, filters, and final output.

For metric questions, choose tables at the requested answer grain:
- if a table already has the needed grouping keys, time key, filters, and a native metric column whose semantics match the question, prefer that table as the metric source
- include lower-grain detail tables only when the question needs detail-level filters, grouping, output columns, an explicit formula, or no suitable native metric exists
- when several native metric columns exist, choose by column-name semantics from the question; do not treat subtotal, total due, tax, freight, or line-item formulas as interchangeable
