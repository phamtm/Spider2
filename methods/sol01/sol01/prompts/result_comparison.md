Compare the executable SQL candidates and pick the one that is closest to the answer contract.

Use the question, reference context, and result profiles to decide which candidate should win.
Prefer the candidate that best matches the requested output shape, grain, filters, and table usage.
If the current best candidate should not win, explain why and name the better stage.

Return a structured comparison with the baseline stage, the preferred stage, all compared stages, and short reasons.
