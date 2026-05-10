# Planning

Select the needed tables and rewrite the question into one answer contract.

For table selection:
- include all tables plausibly required for final output, joins, filters, and metric source choice
- include join and bridge tables, plus lower-grain detail tables when the question may need them
- omit only tables that are clearly irrelevant
- preserve fully qualified table names exactly as shown

For the answer contract:
- stay close to the question and document context
- identify entities, metrics, filters, time constraints, answer grain, ordering, top-k behavior, and output shape
- do not invent current/latest, status, dedupe, filtering, or metric rules
- when a question term is a native schema value, keep it separate from derived business definitions
- put useful but ungrounded ideas in unsupported_assumptions or do_not_assume
