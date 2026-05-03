# Aggregate Verification

Review the result of a query that looks like an aggregate or count.

Check whether a zero or very small result is actually plausible.
Look for signs of over-filtering, wrong grain, or a value variant that may need to be mapped.
If the query is counting a country or similar label, consider nearby value variants before accepting the result.
If the grain looks too coarse or too fine, recommend repair.

Be concrete. Only ask for repair when the result is not trustworthy.
