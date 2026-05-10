# Retrieval Schema Selection Verification

Last verified: 2026-05-10.

This note records the final verification for the retrieval-first schema
selection work tracked by `sp-tvm.14`.

## Code Audit

- Runtime planning enters `_run_planning()` in `sol01/coordinator.py`, builds a
  versioned retrieval index, retrieves/reranks schema objects, calls
  `_retrieval_planning_user_prompt()`, sanitizes selected object IDs, and
  resolves selected logical objects to physical tables.
- The legacy full-schema planning and schema-expansion prompt builders were
  removed from `sol01/llm/prompt_builders.py`.
- Remaining `db_schema_summary` usage is limited to offline retrieval-eval
  prompt-reduction measurement and its tests, not runtime planning or schema
  expansion.
- Runtime code imports offline gold-table labels only through the
  `retrieval-eval` CLI path. The coordinator, planner, SQL generation, repair,
  and candidate review paths do not import or load gold-table data.

## Focused Tests

```bash
uv run pytest tests/test_retrieval_planning.py tests/test_coordinator.py \
  tests/test_retrieval_eval.py tests/test_schema_resolver.py \
  tests/test_hybrid_retrieval.py tests/test_retrieval_index.py \
  tests/test_schema_objects.py tests/test_schema_chunks.py -q
```

Result: `48 passed`.

These tests cover retrieval-scoped planning prompts, planner sanitization,
trace `schema_retrieval_version` and `schema_retrieval` diagnostics, retrieval
schema expansion, offline retrieval-eval accounting, resolver expansion,
hybrid retrieval, retrieval index caching, schema objects, and chunk rendering.

## Full Quality Gates

```bash
uv run pytest tests -q
uv run ruff check .
uv run ruff format --check .
```

Results:

- `260 passed`
- `ruff check`: passed
- `ruff format --check`: passed after formatting `sol01/llm/prompt_builders.py`

## Retrieval-Eval Smoke

Small database smoke:

```bash
SOL01_SCHEMA_RERANKER_MODEL='dengcao/Qwen3-Reranker-4B:Q8_0' \
  uv run sol01 retrieval-eval --db BBC --limit 1
```

Result:

- evaluated 1 task at object cutoff 12
- pre-resolver any-gold recall: `100.0%`
- post-resolver all-gold recall: `100.0%`
- family expansion success: `n/a`
- average prompt reduction: `90.1%`

Large repeated-schema smoke:

```bash
SOL01_SCHEMA_RERANKER_MODEL='dengcao/Qwen3-Reranker-4B:Q8_0' \
  uv run sol01 retrieval-eval --db GA360 --limit 1
```

Result: attempted against a 366-table repeated-schema database and stopped
after several minutes of local Ollama embedding/reranking work with no failure
output. Large/repeated-schema behavior remains covered by focused unit tests;
the live smoke should be rerun when longer local model runtime is acceptable.
