Compare the executable SQL candidates and pick the one that is closest to the answer contract.

Use the question, reference context, and result profiles to decide which candidate should win.
Prefer the candidate that best matches the requested output shape, grain, filters, and table usage.
Prefer the candidate whose assumptions and constraint ledger are best grounded in the answer contract.
For metric questions, prefer candidates that use a native metric column at the requested answer grain only when that column is clearly grounded in the answer contract or its semantics unambiguously match the question.
Do not penalize a lower-grain detail reconstruction when no clearly grounded native metric exists.
When several native metric columns exist, choose by column-name semantics from the question; do not treat subtotal, total due, tax, freight, or line-item formulas as interchangeable.
Do not reward an executable candidate for adding unrequested filters, current/latest rules, dedupe rules, status rules, limits, or row narrowing.
If the contract includes `native_value_terms`, prefer candidates that use those values as native column values instead of inventing a derived business class.
If a candidate preserves a stable grouping identifier alongside a display label, prefer it over a candidate that drops the identifier when both satisfy the same grain.
If the current best candidate should not win, explain why and name the better stage.

Return a structured comparison with the baseline stage, the preferred stage, all compared stages, and short reasons.
