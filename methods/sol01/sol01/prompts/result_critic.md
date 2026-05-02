# Result Critic

Review the SQL result and decide whether it likely answers the question.

Look for:
- assumptions the SQL made that are not grounded in the answer contract
- filters, current/latest rules, dedupe rules, status rules, limits, or row narrowing that the task did not ask for
- mismatches between `native_value_terms` and the SQL when the contract names exact schema values
- wrong shape
- missing filters
- suspicious aggregations
- empty or obviously incorrect results

Recommend repair only when there is a concrete reason.
If an unsupported assumption changes the row set, grain, or answer shape, recommend repair.
