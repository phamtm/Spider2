# Schema Context Selection Verification

Last verified: 2026-05-13.

This note records verification for the schema-context selection work tracked by
`sp-tvm.14`, `sp-4rb.10`, and the later schema-context cleanup follow-up.

## Code Audit

- Runtime planning enters `_run_planning()` in `sol01/coordinator.py`, builds a
  versioned schema-context cache, selects available schema objects, calls
  `_schema_context_planning_user_prompt()`, sanitizes selected object IDs, and
  resolves selected logical objects to physical tables.
- Runtime schema context uses deterministic schema objects in
  `sol01/schema/schema_context.py`. For databases with curated summary
  coverage, the planner sees summary-backed objects only. Otherwise, it sees
  the full logical metadata object set. It does not use dense embeddings, BM25,
  or a separate model-backed search service.
- The legacy full-schema planning and schema-expansion prompt builders were
  removed from `sol01/llm/prompt_builders.py`.
- Remaining `db_schema_summary` usage is limited to offline schema-context-eval
  prompt-reduction measurement and its tests, not runtime planning or schema
  expansion.
- Runtime code imports offline gold-table labels only through the
  `schema-context-eval` CLI path. The coordinator, planner, SQL generation, repair,
  and candidate review paths do not import or load gold-table data.
- Curated large-schema summaries live in
  `methods/sol01/metadata/large_schema_summaries.json`. Edits are validated by
  `sol01/schema/large_schema_summaries.py`, and the summary registry hash is
  part of the schema-context cache key.

## Focused Tests

```bash
uv run pytest tests/test_schema_context_planning.py tests/test_coordinator.py \
  tests/test_schema_context_eval.py tests/test_schema_resolver.py \
  tests/test_schema_context.py tests/test_schema_context_cache.py \
  tests/test_schema_objects.py -q
```

Result: `48 passed`.

These tests cover schema-context planning prompts, planner sanitization, trace
`schema_context_version` and `schema_context` diagnostics, schema recovery,
offline schema-context-eval accounting, resolver expansion, large-schema
summaries, schema-context caching, and schema objects.

## Full Quality Gates

```bash
uv run pytest tests -q
uv run ruff check .
uv run ruff format --check .
```

Results:

- `just check`: passed on 2026-05-13
- `ruff check`: passed
- `ruff format`: passed
- `pytest tests -q`: passed

## Prompt Budget And Hallucinated Columns

Prompt budget diagnostics are recorded in each task trace under
`schema_context.prompt_budget`. The diagnostics include planning and resolved
context character counts and whether each value stayed within
`SOL01_SCHEMA_MAX_PROMPT_CHARS`.

`uv run sol01 schema-context-eval --output-id <id>` persists prompt-reduction and
prompt-size-win reporting under `outputs/<id>/schema_context_eval/`. Add
`--trace-run-id <run_id>` to scan saved solver traces for hallucinated-column
validation failures. Runtime validation blocks unknown columns before
Snowflake execution when selected schema context is specific enough.

## Schema-Context-Eval Smoke

Small database smoke:

```bash
uv run sol01 schema-context-eval --db BBC --limit 1
```

Result:

- evaluated 1 task at object cutoff 12
- pre-resolver any-gold recall: `100.0%`
- post-resolver all-gold recall: `100.0%`
- family expansion success: `n/a`
- average prompt reduction: `90.1%`

Large repeated-schema smoke:

```bash
uv run sol01 schema-context-eval --db GA360 --limit 1
```

Result: attempted against a 366-table repeated-schema database and stopped
after several minutes with no failure output. Large/repeated-schema behavior
remains covered by focused unit tests; the live smoke should be rerun when
longer local runtime is acceptable.
