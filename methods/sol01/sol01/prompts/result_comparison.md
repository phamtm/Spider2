Compare the executable SQL candidates and pick the one that is closest to the answer contract.

Use the question, reference context, and result profiles to decide which candidate should win.
Prefer the candidate that best matches the requested output shape, grain, filters, and table usage.
Prefer the candidate whose assumptions and constraint ledger are best grounded in the answer contract.
Do not reward an executable candidate for adding unrequested filters, current/latest rules, dedupe rules, status rules, limits, or row narrowing.
If a candidate preserves a stable grouping identifier alongside a display label, prefer it over a candidate that drops the identifier when both satisfy the same grain.
If the current best candidate should not win, explain why and name the better stage.

Return a structured comparison with the baseline stage, the preferred stage, all compared stages, and short reasons.
