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
- When a question term matches a sample or domain value in a selected text column, record it in `native_value_terms` as a native column value, not as a behavioral definition.
- Keep native column values separate from `derived_behavioral_definitions`; use the latter only for business meanings that are not literal schema values.
- For grouped superlatives such as "highest number in any month", "largest increase between months", or "most frequent item per group", keep the identifying group key with the metric unless the task explicitly asks for only the scalar value.
- Do not infer current/latest/snapshot/status/release rules merely because the schema has columns that could support them.
- Put useful but ungrounded ideas in unsupported_assumptions or do_not_assume instead of treating them as requirements.
- Keep evidence short and quote or paraphrase the task text that grounds each important constraint.
