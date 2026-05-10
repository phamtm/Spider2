# Plan: Retrieval-Based Schema Selection for `methods/sol01`

Tracking epic: `sp-tvm`

Implementation is split into self-contained child beads. Agents should claim
the child beads, not the parent epic:

- `sp-tvm.1`: Schema retrieval config and core models
- `sp-tvm.2`: Build canonical schema objects from metadata
- `sp-tvm.3`: Detect table families and partition dimensions
- `sp-tvm.4`: Render schema chunks for retrieval and prompts
- `sp-tvm.5`: Implement local Ollama embedding and reranker providers
- `sp-tvm.6`: Build retrieval index cache and prewarm flow
- `sp-tvm.7`: Implement hybrid schema retrieval and object aggregation
- `sp-tvm.8`: Add retrieval planner prompt and structured selection
- `sp-tvm.9`: Resolve selected schema objects to allowed physical tables
- `sp-tvm.10`: Integrate retrieval path into sol01 coordinator
- `sp-tvm.11`: Make schema expansion retrieval-based
- `sp-tvm.12`: Add offline retrieval evaluation
- `sp-tvm.13`: Document retrieval-based schema selection
- `sp-tvm.14`: Final verification for retrieval schema selection

## Problem

`methods/sol01` currently relies on passing a large schema summary or DDL-like
context to an LLM so it can pick relevant tables. This does not scale:

- Some Spider2 databases have hundreds of tables.
- Some tables have thousands of columns.
- Repeated physical tables can share almost the same structure with only a
  date, year, shard, or suffix difference.
- Passing the full schema increases latency and cost, and can make the planner
  less accurate because useful tables and columns are buried in noise.
- The current approach does not use available table and column metadata under
  `spider2-snow/resource/databases` enough to guide selection.

The replacement should be generic. It must not encode dataset-specific rules
such as "COVID tables mean X" or "GA tables mean Y".

## Target Pipeline

```text
User question
  -> hybrid schema retrieval
  -> local reranking
  -> planner selects logical schema objects
  -> resolver expands families/partitions to physical tables
  -> compact schema context
  -> SQL generation, validation, execution, repair
```

## Non-Negotiable Decisions

- Remove the old `llm_only` schema-selection mode. Do not preserve a
  compatibility path that passes the full schema to the planner.
- Use one retrieval-based path and one trace/version field, for example
  `schema_retrieval_version = "hybrid_v1"`.
- Old traces without the new retrieval version should rerun.
- Use local Ollama models:
  - embedding: `qwen3-embedding:4b`
  - reranker: Qwen3-Reranker-4B through an installed Ollama model tag
- Fail clearly when Ollama is down or required local models are missing.
- Do not use gold tables at runtime.
- Do not add the Pi/LLM schema explorer in this version.
- SQL generation must receive the resolved compact schema context, not the full
  database schema summary.

## Implementation Order

### 1. Configuration

Add a schema retrieval config to `methods/sol01/sol01/infra/config.py`.

Suggested fields:

- `schema_retrieval_version: str = "hybrid_v1"`
- `ollama_base_url: str = "http://localhost:11434"`
- `embedding_model: str = "qwen3-embedding:4b"`
- `reranker_model: str`
- `top_k_sparse: int = 80`
- `top_k_dense: int = 80`
- `top_k_rerank: int = 80`
- `top_k_objects: int = 30`
- `family_similarity_threshold: float = 0.85`
- `max_doc_chars: int = 4000`
- `max_schema_prompt_chars: int = 80000`

There should be no runtime config option for `llm_only`.

### 2. Core Models

Add schema retrieval models in `methods/sol01/sol01/models/core.py` and export
them from `methods/sol01/sol01/models/__init__.py`.

Required model concepts:

- `SchemaObject`
- `RetrievalChunk`
- `RetrievedChunk`
- `RetrievedSchemaObject`
- `SelectionConstraints`
- `SelectedSchemaObject`
- `ResolvedSchemaContext`
- `HybridPlanningDecision`

Stable object id formats:

- `table:<full_table_name>`
- `column:<full_table_name>#<column_name>`
- `column_group:<full_table_name>#<group_slug>:<8char_hash>`
- `sample_value:<full_table_name>#<column_name>:<8char_hash>`
- `join_candidate:<left_table>#<left_col>-><right_table>#<right_col>:<8char_hash>`
- `family:<db>.<schema_or_none>:<stem_slug>:<8char_hash>`

`SelectionConstraints` must be valid with no arguments. The selected-object
role enum must include `unknown`, because exact-table fallback and legacy-ish
normalization should not invent a semantic role.

### 3. Deterministic Schema Object Builder

Create `methods/sol01/sol01/schema/objects.py`.

Input: the existing `load_db_index(db)` table metadata.

Output: canonical `SchemaObject` records for:

- tables
- columns
- column groups
- join candidates
- sample values, only for likely categorical columns
- table families

This layer uses fixed handwritten logic. That is an intentional design risk and
should be tested carefully. It is allowed to describe structural evidence and
compress repeated patterns, but it must not make domain-specific semantic
decisions.

Generic structural rules only:

- Key-like columns: names containing `id`, `_id`, `key`, `code`, `uuid`.
- Time-like columns: date/time types or date-like names.
- Numeric measure candidates: numeric columns excluding key-like columns.
- Repeated-prefix column groups: at least 3 columns sharing a meaningful prefix.
- Join candidates: same normalized column name and compatible primitive type
  across tables in the same database/schema. Treat these as evidence, not real
  foreign keys.

### 4. Table Family Detection

Family objects are the main context-reduction mechanism for large repeated
schemas.

Create family objects when tables show repeated physical structure:

- Exact family: same ordered `(column_name, type)` signature, at least 2 tables.
- Near family: same normalized stem pattern and column-name Jaccard similarity
  above the configured threshold, at least 3 tables.
- Partition dimensions from generic suffix patterns only:
  - `YYYYMMDD`
  - `YYYY_MM_DD`
  - `YYYY`
  - integer or version suffixes

Family object metadata should include:

- member table refs
- canonical member table ref, chosen deterministically
- common columns
- capped variant columns
- detected suffix or partition dimensions
- source paths or metadata provenance
- caveats when the family is near-match rather than exact

Wide-column patterns are not table families. Represent those as table facets or
column groups, not fake repeated tables.

### 5. Sample Value Policy

Sample values are dangerous for cardinality and retrieval quality. Index them
only for likely low-cardinality categorical columns.

Do index sample values when:

- metadata exposes low distinct cardinality, or a bounded sample strongly
  indicates low cardinality
- values are short, repeated, human-readable labels or codes
- values look like enums, statuses, categories, regions, types, or similar
- distinct count is below a fixed cap, such as 50 or 100
- sampled values cover a meaningful portion of observed rows

Do not index sample values when:

- values look like UUIDs, IDs, hashes, emails, URLs, timestamps, raw text,
  JSON, blobs, addresses, names, or free-form descriptions
- distinct count is high or unknown and cannot be safely bounded
- values are mostly unique in the sample
- values are long, sparse, binary-looking, or opaque
- the column is numeric continuous, date/time, or key-like

Implementation requirements:

- Create `sample_value` objects only for columns classified as categorical.
- Cap indexed values per column, for example top 20 frequent values.
- Store provenance: source metadata, sample size, distinct count, and reason.
- Prefer BM25/exact matching for sample values.
- Do not dense-embed arbitrary sample values by default.
- Include sample values in the planner prompt only when the parent table/column
  has been retrieved.

### 6. Chunk Rendering

Create `methods/sol01/sol01/schema/chunks.py`.

Render deterministic chunks from `SchemaObject`. The object is the source of
truth; chunks are retrieval views.

Each chunk should have:

- `embedding_text`: semantic description for dense retrieval.
- `bm25_text`: exact names, identifiers, normalized tokens, and safe literals.
- `rerank_text`: concise query-pair evidence for the reranker.
- `prompt_text`: compact planner-facing text.

Keep `source_definition` separate from `inferred_usage`. Do not present
inferred usage as if it came from source metadata.

### 7. Ollama Providers

Create provider code under `methods/sol01/sol01/schema/`.

Suggested files:

- `embedding.py`: protocol interfaces, vector helpers, fake providers for tests.
- `ollama_provider.py`: real Ollama embedding and reranker clients.

Embedding behavior:

- Call `POST /api/embed`.
- Use `model = qwen3-embedding:4b`.
- Batch inputs where possible.
- Normalize vectors before similarity search.

Reranking behavior:

- Call `POST /api/generate` or `POST /api/chat` with `stream=false`.
- Use the configured Qwen3-Reranker-4B Ollama model.
- Use the reranker in yes/no relevance mode.
- Request `logprobs=true` and `top_logprobs` when supported.
- Score from first-token `yes` versus `no` probabilities.
- If the server/model cannot return usable logprobs, fail clearly rather than
  silently falling back to weak free-form scoring.

Do not add `sentence-transformers` or `torch` as project dependencies for this
version.

### 8. Retrieval Index And Cache

Create `methods/sol01/sol01/schema/retrieval_index.py`.

Cache layout:

```text
.cache/schema_retrieval_index/<db>/versions/<cache_key>/
  manifest.json
  objects.jsonl
  chunks.jsonl
  sparse.json
  embeddings.npy
.cache/schema_retrieval_index/<db>/current.json
```

`cache_key` should include:

- source DB schema content hash
- object builder version
- render version
- embedding model name
- embedding model metadata or digest if available
- family similarity threshold

Safe build rules:

- Write to a temp version directory.
- Validate required files before publishing.
- Rename temp to final only if final does not already exist.
- If final exists, delete temp and reuse final.
- Update `current.json` via temp file plus `os.replace`.
- Use a build lock. Concurrent workers should wait bounded time and reload the
  current pointer instead of racing writes.
- Never overwrite a populated cache directory with `mv -f`.

### 9. Hybrid Retrieval Algorithm

Create `methods/sol01/sol01/schema/hybrid_retrieval.py`.

Query construction:

- raw user question
- clipped linked docs context, max `max_doc_chars`
- exact literals from quotes, dates, years, integers, dotted names,
  underscore names, and uppercase codes
- lightweight normalized query tokens

Retrieval:

- Local BM25 over `bm25_text`.
- Dense vector search over `embedding_text`.
- Exact lookup boosts for literal identifier hits.
- Do not dense-search sample values by default.

Use per-type quotas before reranking so one chunk type cannot crowd out all
others. Example quotas:

- table family: 20
- table: 30
- column group: 20
- column: 40
- join candidate: 20
- sample value: 20, BM25/exact only

Merge and deduplicate chunks while retaining evidence from sparse, dense, and
exact matches.

Rerank at most `top_k_rerank` chunks using the raw question, clipped docs, and
candidate `rerank_text`.

Aggregate chunk evidence back to schema objects:

- direct table/family hits count strongly
- column and column-group hits contribute to parent table and family
- sample-value hits contribute filter evidence to parent column/table
- join-candidate hits contribute endpoint evidence
- avoid one opaque weighted formula that cannot be tested; use clear constants
  and focused tests for ranking behavior

Return top `top_k_objects` objects plus diagnostics.

### 10. Planner Prompt And Structured Output

Replace the full-schema planning prompt with a retrieval-planning prompt.

The planner sees:

- user question
- clipped external docs context
- top retrieved schema objects
- compact object evidence
- available object ids

The planner returns `HybridPlanningDecision`:

- selected schema objects
- constraints such as date range, years, suffixes, version, or include-all
- rationale
- confidence
- intent

The planner must not be allowed to invent object ids. Sanitize selected ids
against retrieved object ids.

If the model emits exact table names for compatibility, normalize only exact
retrieved tables into `SelectedSchemaObject(kind="table", role="unknown")`.
Do not use this as a broad fallback to the full schema.

### 11. Resolver

Create `methods/sol01/sol01/schema/resolver.py`.

Resolver input:

- selected logical schema objects
- canonical schema objects
- database index
- user question and retrieval evidence

Resolver output: `ResolvedSchemaContext`:

- `allowed_tables: list[str]`
- `table_schemas: dict[str, TableSchema]`
- `prompt_context: str`
- `resolution_diagnostics: dict`

Behavior:

- Table selection resolves to that exact physical table.
- Family selection with explicit date/year/suffix/version constraints resolves
  to matching members.
- `include_all=true` resolves to all family members.
- If constraints are missing, include the canonical member plus a warning unless
  the question/evidence contains broad range terms such as all, history, every,
  daily, monthly, between, or a clear date span. In those broad cases, include
  all matching family members.
- If constraints do not match any member, include canonical member plus a
  diagnostic warning for recovery.

Prompt context for families:

- render canonical DDL once
- list member tables or compact member ranges
- show common columns
- show capped variant columns
- show matched date/suffix dimensions
- show retrieved filter/join evidence

Validation must use `allowed_tables`, not only the canonical prompt table.

### 12. Coordinator Integration

Update `methods/sol01/sol01/coordinator.py`.

Expected changes:

- Remove branches that call full `db_schema_summary` for planning.
- Build or load retrieval index before planning.
- Run hybrid retrieval.
- Call retrieval planner.
- Resolve selected objects.
- Store `ResolvedSchemaContext` in task context.
- Use `resolved.prompt_context` for SQL generation, review, repair, and schema
  expansion prompts.
- Use `resolved.allowed_tables` for SQL validation.
- Store compact selection in `SchemaSelection`.
- Store full retrieval evidence in a top-level trace field such as
  `schema_retrieval`.

`SchemaSelection` should stay small:

- selected object ids
- resolved/allowed tables
- compact diagnostics

Full retrieved chunks and scores should live in trace diagnostics, not the
planner-facing selection object.

### 13. Schema Expansion

Keep schema expansion, but make it retrieval-based.

When validation or execution fails because of missing schema context:

- Build an augmented retrieval query from the original question, failed SQL,
  validation/execution error, and current selected object ids.
- Rerun hybrid retrieval with this query.
- Ask the planner to select from retrieved expansion candidates only.
- Resolve again.
- Do not pass the full database schema as an expansion fallback.

Deterministic exact-name recovery is allowed only when the error names a table
that maps unambiguously to the database index.

### 14. Offline Retrieval Evaluation

Create `methods/sol01/sol01/schema/retrieval_eval.py` and a CLI command such as
`sol01 retrieval-eval`.

Use gold tables offline only. Do not feed gold tables into runtime retrieval.

Metrics:

- pre-resolver any-gold recall at object cutoff
- post-resolver all-gold recall at object cutoff
- missing gold tables
- family expansion success for tasks with many gold tables from one detected
  family
- prompt character reduction versus full schema summary
- top evidence chunks for failures

This should be runnable without Snowflake execution.

### 15. Tests

Add focused tests before broad integration tests.

Required test coverage:

- config defaults and env overrides
- no runtime `llm_only` mode remains
- stable and unique object ids
- exact family detection
- near family detection
- suffix/date/year partition extraction
- wide-column patterns are not table families
- join candidates are evidence only, not foreign keys
- sample values are included only for bounded categorical columns
- sample values exclude UUIDs, hashes, timestamps, IDs, raw text, JSON, blobs,
  continuous numeric values, and high/unknown-cardinality columns
- chunk rendering is deterministic
- `source_definition` and `inferred_usage` stay separate
- cache invalidates on schema/model/render changes
- cache publication is atomic and concurrency-safe
- fake embedding/reranker providers work without Ollama
- real provider failure messages are actionable
- retrieval per-type quotas prevent one object type from crowding out others
- resolver expands explicit ranges and include-all correctly
- resolver warnings appear for ambiguous family constraints
- validation accepts resolved non-canonical family members
- prompt-size regression for large repeated schemas
- coordinator does not use full `db_schema_summary` in planning or expansion

Quality gates for the implementation:

```bash
cd methods/sol01
uv run pytest tests -q
uv run ruff check .
uv run ruff format --check .
```

### 16. Documentation

Update `methods/sol01/docs/SOLUTION_OVERVIEW.md` and any relevant README notes.

Document:

- retrieval-first architecture
- Ollama model requirements
- cache layout
- index build command
- runtime command
- retrieval-eval command
- sample-value policy and why it is strict
- no `llm_only` mode

## Expected Accuracy Lever

The largest accuracy gain should come from high-recall schema retrieval plus
correct family/partition expansion. Embeddings and reranking help find the
right logical schema area, but expansion is what prevents missing many physical
tables when the correct SQL must union or scan a repeated table family.

## Main Risks To Watch

- Handwritten object-building logic may be brittle if it over-groups unrelated
  tables or misses real families.
- Sample values can bloat and pollute retrieval if cardinality is not strictly
  controlled.
- Reranker scoring through Ollama depends on usable yes/no logprobs from the
  chosen local model.
- Compact prompt context can hide columns needed for filters, joins, or
  aggregation if object aggregation is too aggressive.
- Resolver expansion can under-select physical tables if constraints are
  ambiguous or date/suffix parsing is too narrow.
