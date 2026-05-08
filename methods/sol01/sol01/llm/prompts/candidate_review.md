# Candidate Review

Review executable SQL candidates and decide whether the preferred result needs repair.

First, choose the candidate that best satisfies the answer contract. Compare:
- output shape
- answer grain
- filters and time constraints
- table usage and join path
- metric source
- native value handling
- grounded versus unsupported assumptions

Then decide whether the preferred candidate should be repaired. Look for:
- ungrounded filters, current/latest rules, status rules, dedupe rules, limits, or row narrowing
- missing filters or time constraints
- wrong shape or missing grouped identifiers
- suspicious aggregations, including empty, zero-like, or tiny aggregate results
- metric-source mismatches
- native value mismatches
- unsupported assumptions that change row set, grain, or answer shape

Recommend repair only when there is a concrete reason.
