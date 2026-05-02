# Intent Extraction

Rewrite the user question into a structured intent.

Focus on:
- entities
- metrics
- filters
- time constraints
- answer grain
- ordering or top-k requirements
- expected output shape
- assumptions that need to be made, with evidence

Stay close to the question. Do not invent business rules that are not implied.

Treat the output as an answer contract for later SQL review:
- Only list a filter, time constraint, dedupe rule, ordering rule, or top-k rule when it is grounded in the question or document context.
- Do not infer current/latest/snapshot/status/release rules merely because the schema has columns that could support them.
- Put useful but ungrounded ideas in unsupported_assumptions or do_not_assume instead of treating them as requirements.
- Keep evidence short and quote or paraphrase the task text that grounds each important constraint.
