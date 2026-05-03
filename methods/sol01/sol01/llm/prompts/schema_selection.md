# Schema Selection

Include all tables that are plausibly required to answer the question, including join and bridge tables.

Use:
- the question
- the retrieved schema candidates
- any metric or document context already provided

Include every table that could plausibly be needed for joins, filters, or final output. Omit only tables that are clearly irrelevant to the question.

For metric questions, include tables at every grain that may be needed:
- if a table has the needed grouping keys, time key, filters, and a native metric column that is clearly grounded in the answer contract or whose semantics unambiguously match the question, that table is the preferred metric source
- always also include lower-grain detail tables when the question may need detail-level filters, grouping, output columns, an explicit formula, or when no clearly grounded native metric exists
- when several native metric columns exist, choose by column-name semantics from the question; do not treat subtotal, total due, tax, freight, or line-item formulas as interchangeable
