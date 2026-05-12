# sol01 Solution Overview

`sol01` is a Snowflake-only solver for Spider2-Snow.

It reads benchmark questions, asks an LLM to write Snowflake SQL, runs the SQL
against Snowflake, saves the result CSV, and scores that CSV with the official
Spider2-Snow evaluator.

The main goal is not just to get an answer. The goal is to make every run
auditable: the selected schema, prompts, candidate SQL, validation result,
execution result, final CSV, official eval output, and raw LLM calls are all
saved under `methods/sol01/outputs/`.

## What It Solves

Each Spider2-Snow task gives the solver:

- an `instance_id`
- a Snowflake database name
- a natural-language question
- sometimes, a linked markdown document with business or metric context

`sol01` turns that into:

- one final SQL file
- one final CSV prediction
- one trace explaining how the SQL was chosen
- official evaluator artifacts showing whether the CSV passed

## High-Level Flow

```text
Spider2-Snow task
  -> load database metadata
  -> retrieve relevant logical schema objects
  -> ask the LLM to select from retrieved objects and describe question intent
  -> resolve selected logical objects to physical Snowflake tables
  -> render compact schema context for SQL generation
  -> generate several SQL candidates in one batch
  -> validate each SQL candidate
  -> execute valid candidates in Snowflake
  -> attach local observations and score breakdowns
  -> ask the LLM to adjudicate executable candidates
  -> repair when needed
  -> write final SQL and CSV
  -> run official Spider2-Snow eval
  -> save run, trace, eval, and registry artifacts
```

## Main Entry Points

- `just run` runs the solver CLI with its default batch settings.
- `just run-selected <selector>...` runs selected solver tasks by exact ID,
  glob, tier, tag, or `all` selector.
- `just gold <instance_id>` runs the benchmark gold SQL for one exact question.
- `just progress` opens the local progress dashboard.
- `uv run sol01 ...` exposes lower-level commands for indexing, running, eval,
  analysis, ad hoc questions, and LLM call inspection.

Common examples:

```bash
uv run sol01 run
uv run sol01 run --instance-id sf_bq320
uv run sol01 run sf035 sf_bq135 sf_bq084
uv run sol01 run --db E_COMMERCE --question-contains revenue
just run-selected sf035 sf_bq135
just gold sf_bq320
```

## Data Sources

`sol01` reads benchmark data from the checked-in Spider2-Snow assets:

- tasks: `spider2-snow/spider2-snow.jsonl`
- database metadata: `spider2-snow/resource/databases/`
- linked documents: `spider2-snow/resource/documents/`
- official evaluator: `spider2-snow/evaluation_suite/evaluate.py`
- gold SQL: `spider2-snow/evaluation_suite/gold/sql/`

Runtime secrets stay local:

- LLM config comes from shell variables or `methods/sol01/.env`.
- Snowflake credentials come from `methods/sol01/snowflake_credential.json`.

## Schema Indexing

Before SQL generation, `sol01` builds two local caches from the Spider2-Snow
metadata.

The base metadata cache is built with:

```bash
uv run sol01 index
```

For each table, the index keeps:

- the canonical Snowflake table name
- the DDL text
- column names and types
- column descriptions
- bounded sample rows and sample values

This index is cached at `methods/sol01/.cache/snow_index.json`.

The solver preserves fully qualified Snowflake table names when they are
available, because table identity matters during validation and execution.

The retrieval cache is built per database from the base metadata cache. It
contains canonical schema objects, retrieval chunks, local lexical postings, and
a version manifest:

```text
methods/sol01/.cache/schema_retrieval_index/<DB>/
  current.json
  versions/<cache_key>/
    manifest.json
    objects.jsonl
    chunks.jsonl
    sparse.json
```

Build retrieval indexes before a batch run with either command:

```bash
uv run sol01 prewarm-schema-index E_COMMERCE
uv run sol01 run --db E_COMMERCE --prewarm-schema-index
```

The retrieval index cache key includes the source schema hash, schema-object
builder version, chunk renderer version, sparse-index version, and family
similarity threshold. If one of those inputs changes, a new version directory
is created and `current.json` is updated.

## Retrieval-First Planning

The old full-schema planner mode has been removed. `sol01` no longer has an
`llm_only` or full-schema table-selection path. Runtime schema retrieval is
local lexical/exact matching over persisted schema chunks, with no separate
model-backed retrieval service.

That path was removed because large Spider2-Snow databases can have hundreds of
tables or very wide tables. Passing a database-wide DDL summary into planning
creates huge prompts, makes table selection noisy, and can bury the few relevant
columns among irrelevant schema text.

The runtime pipeline is:

```text
question
  -> lexical/exact retrieval over schema chunks
  -> LLM planner selects retrieved logical objects
  -> resolver expands logical families to physical tables
  -> compact schema context is rendered
  -> SQL generation uses only that compact context
```

Schema objects are logical, not just physical tables. They include:

- table objects
- column objects
- column-group objects for keys, time columns, measures, and repeated prefixes
- inferred join candidates
- bounded categorical sample-value objects
- table-family objects for partitioned or suffixed physical tables

Lexical retrieval combines local term scoring with exact literal, code, and date
matches. Linked markdown documents are clipped to passages that overlap with
the question before retrieval, rather than being sent wholesale.

For each question, the planner sees only retrieved object evidence. It returns
selected object IDs, object roles, constraints, and the answer contract in one
structured response.

The solver sanitizes the planner output:

- object IDs outside the retrieved candidate set are dropped
- table names are normalized only when they match retrieved evidence
- if no valid retrieved schema object survives, confidence is lowered
- planner diagnostics record ignored or missing selections

The same planning call also returns an answer contract capturing:

- entities
- metrics
- filters
- time constraints
- answer grain
- requested ordering or top-k behavior
- expected output shape
- assumptions and unsupported assumptions

This step is important because later stages use the contract to reject SQL that
quietly adds extra filters, drops grouping keys, uses the wrong metric source,
or returns the wrong shape.

The resolver turns selected logical objects into the allowed physical table set.
For table-family objects, explicit date, year, suffix, version, or include-all
constraints select matching members. Broad range questions can include all
members. Ambiguous families fall back to a canonical member and record a
warning. The resolver then renders the compact schema context used by SQL
generation.

## Sample-Value Policy

Sample values are indexed only when they are bounded, low-cardinality
categorical evidence. The retrieval index excludes high-cardinality,
opaque, free-text, and sensitive-looking values because they can create noisy
retrieval hits, leak irrelevant literals into prompts, and encourage the model
to treat arbitrary examples as business rules.

Sample-value objects are excluded for:

- key-like columns such as IDs, UUIDs, hashes, and foreign keys
- temporal columns and date-like values
- numeric, temporal, and semi-structured column types
- raw text, description, body, email, URL, JSON, payload, and similar columns
- values longer than the short label/code threshold
- columns with too many distinct sample values
- columns where every sampled value is distinct unless the column name is
  explicitly categorical

Included sample values are sparse/exact-match evidence. If a question mentions
an included sample value, the solver can keep that value tied to its native
column instead of converting it into an invented rule.

## Curated Large-Schema Summaries

Curated summaries live in
`methods/sol01/metadata/large_schema_summaries.json`. They compact repeated
table families and wide repeated column groups before retrieval chunks are
rendered.

Add or edit a summary when a schema family has a stable shape that should be
retrieved as one logical object. Each entry defines:

- `summary_id`: lower_snake_case stable ID
- `schema_copies`: database/schema locations covered by the same shape
- `match`: either exact `table_names` or a regex `table_pattern`, optionally
  with an inclusive suffix range
- `purpose` and `grain`
- stable columns, repeated-column rules, inclusive ranges, quoting rules,
  examples, and aliases

Summaries must describe schema shape only. Do not include benchmark answers,
gold SQL, instance IDs, or question-specific hints. The registry validator
rejects those tokens, and the retrieval cache key includes the summary registry
hash, so a summary edit creates a fresh cache version.

Relevant runtime settings:

- `SOL01_SCHEMA_CHUNK_TOP_K`, default `80`
- `SOL01_SCHEMA_OBJECT_TOP_K`, default `12`
- `SOL01_SCHEMA_FAMILY_TOP_K`, default `8`
- `SOL01_SCHEMA_FAMILY_SIMILARITY_THRESHOLD`, default `0.82`
- `SOL01_SCHEMA_MAX_LINKED_DOC_CHARS`, default `6000`
- `SOL01_SCHEMA_MAX_PROMPT_CHARS`, default `24000`
- `SOL01_SCHEMA_RETRIEVAL_VERSION`, default `lexical_v1`

These can be set in the shell or in `methods/sol01/.env`.

## Retrieval Evaluation

Offline retrieval coverage can be checked with:

```bash
uv run sol01 retrieval-eval
uv run sol01 retrieval-eval --db E_COMMERCE --json
```

The command reports pre-resolver any-gold recall, post-resolver all-gold
recall, family expansion success, and prompt reduction. By default it reads
offline labels from `methods/gold-tables/spider2-snow-gold-tables.jsonl`;
`--gold-path <path>` is only for alternate local label files. Gold tables are
offline-only labels for measuring retrieval coverage. They are not available to
runtime planning, SQL generation, repair, or candidate review.

Use `--covered-only` to focus on tasks whose gold tables touch curated
large-schema summaries. Use `--baseline-path <report.json|tasks.jsonl>` to
check for recall regressions, `--trace-run-id <run_id>` to scan saved solver
traces for hallucinated-column validation failures, and `--output-id <id>` to
persist `summary.json`, `tasks.jsonl`, `failures.json`, `report.json`, and
`summary.md` under `outputs/<id>/retrieval_eval/`.

## SQL Generation

For each task, the coordinator asks for the initial SQL candidates in a single
batch call.

Each SQL prompt includes:

- the compact resolved schema context
- DDL and column details
- bounded sample rows and categorical sample values
- linked document text
- the extracted answer contract
- guidance about aggregate grain and metric source when available

The prompt asks for independently executable read-only Snowflake queries. It
also asks the model to record assumptions, unsupported assumptions, and the
constraint ledger behind row narrowing, dedupe, ordering, and top-k choices for
each candidate.

## Validation

Before a query touches Snowflake, `sol01` validates it locally.

Validation checks that the SQL:

- parses as Snowflake SQL
- contains exactly one statement
- is read-only
- does not load extensions
- only references selected tables
- uses known columns when the selected schema is clear enough

Invalid SQL is not executed. It can still be used as repair feedback.

Unknown-column validation is the runtime guard for hallucinated columns. The
offline retrieval-eval command can also scan saved traces with
`--trace-run-id <run_id>` and report those validation failures.

## Execution

Valid candidates run against Snowflake.

For each executed candidate, the solver records:

- whether execution succeeded
- row count
- output columns
- sample rows
- a small result profile
- validation warnings or errors
- elapsed time

The candidate CSV is not written immediately. Only the final selected candidate
becomes the saved prediction CSV.

## Candidate Observations

`sol01` records local observations for each candidate.

The strongest signal is execution:

- executable SQL gets a large positive score
- non-executable SQL gets a large negative score

The observation payload also includes:

- validation quality
- expected output shape
- filter grounding
- aggregate grain
- result size plausibility
- model confidence as a small tie-breaker

The numeric score is retained as diagnostic evidence. It is separate from the
official benchmark score, which runs later on the final CSV, and it is not the
semantic authority for choosing the final answer.

## Candidate Review

After executable candidates are observed, the solver asks the LLM for one
combined candidate review. Local scores, validation output, execution profiles,
shape reports, and grounding reports are evidence for that review rather than
hardcoded semantic policy.

The review chooses the candidate that best matches:

- output shape
- grain
- filters
- table usage
- metric source
- grounded assumptions

The same review also decides whether the preferred candidate needs repair. This
keeps semantic judgment in the model while preserving deterministic safety checks
in local code.

## Repairs

`sol01` has three repair paths.

### Execution Repair

If the best candidate does not execute, the solver asks for a SQL repair using
the validation and execution feedback.

### Semantic Repair

When candidate review finds a concrete semantic issue, the solver asks for one
semantic repair.

The review looks for concrete issues such as:

- ungrounded filters
- wrong shape
- missing filters
- suspicious aggregates
- wrong metric source
- literal values treated as invented business definitions

By default, each task has a small attempt budget: three initial candidates and
up to four total attempts.

## Final Output Selection

Only one candidate becomes the final answer.

The final candidate must execute successfully. The solver writes:

- `sql/<instance_id>.sql`
- `csv/<instance_id>.csv`
- `traces/<instance_id>.json`
- `llm_calls/<instance_id>.jsonl`

The trace contains the full local decision path: schema selection, intent,
prompt hashes, attempts, scores, candidate review output, final SQL, and final
execution summary.

Retrieval traces also include:

- `schema_retrieval_version`
- effective `schema_retrieval_config`
- retrieval diagnostics for sparse and exact stages
- retrieved object IDs, scores, evidence, and chunk snippets
- planner sanitization diagnostics
- resolver entries, allowed tables, and resolver warnings
- schema expansion retrieval diagnostics when repair expands schema context

Prompt budget diagnostics live in `schema_retrieval.prompt_budget` inside each
task trace. They record planning and resolved-context character counts against
`SOL01_SCHEMA_MAX_PROMPT_CHARS`. `uv run sol01 retrieval-eval --output-id <id>`
also writes prompt reduction and prompt-size wins to
`outputs/<id>/retrieval_eval/`.

## Official Evaluation

After the solver writes CSVs, it runs the official Spider2-Snow evaluator in
`exec_result` mode.

The evaluator is run inside the persisted run folder. Its command, stdout,
stderr, staged input CSVs, temp workspace, metadata copy, credential copy, and
summary files are saved.

The local registry records the difference between:

- `solver_failed`: the solver did not produce a usable CSV
- `missing_csv`: no CSV existed for an expected task
- `official_fail`: the official evaluator scored the CSV as wrong
- `eval_failed`: the evaluator failed before scoring finished
- `pass`: the official evaluator scored the CSV as correct

## Output Layout

Each persisted run writes to:

```text
methods/sol01/outputs/<run_id>/
  manifest.json
  logs/
    stdout.txt
    stderr.txt
    run.jsonl
  sql/
  csv/
  traces/
  llm_calls/
  eval/
    scored_csv/
    summary.json
    per_instance.jsonl
    runs/<eval_id>/
      command.json
      stdout.txt
      stderr.txt
      summary.json
      per_instance.jsonl
      input_csv/
      workspace/
  analysis/
```

The registry lives at:

```text
methods/sol01/outputs/registry/
  runs.jsonl
  task_results.jsonl
  latest.json
```

## Debugging A Run

Start with the run folder.

1. Read `logs/run.jsonl` to see the run sequence.
2. Read `logs/stdout.txt` and `logs/stderr.txt`.
3. Open `traces/<instance_id>.json` for the per-question decision path.
4. Open `llm_calls/<instance_id>.jsonl` to inspect raw prompts and responses.
5. Check `sql/<instance_id>.sql` and `csv/<instance_id>.csv`.
6. Check `eval/summary.json` and `eval/per_instance.jsonl`.
7. If official eval failed, inspect `eval/runs/<eval_id>/`.
8. Check `outputs/registry/latest.json` for the latest status view.

Useful commands:

```bash
uv run sol01 analyze --run-id <run_id>
uv run sol01 llm-calls --run-id <run_id> --instance-id <instance_id>
uv run sol01 llm-calls --run-id <run_id> --instance-id <instance_id> --call-id <call_id>
uv run sol01 retrieval-eval --limit 20
just progress
```

## Concurrency

Batch runs use task-level concurrency.

The default worker count is `4`. It can be set with:

- `SOL01_CONCURRENCY` in `methods/sol01/.env`
- `--concurrency` on `sol01 run`

Each worker handles a whole question. The solver does not split one question
across workers. This keeps output artifacts coherent and avoids shared CSV or
trace races.

## Main Files

- `sol01/tasks.py`: loads and filters Spider2-Snow tasks.
- `sol01/schema/index.py`: builds the Snowflake metadata cache.
- `sol01/schema/objects.py`: builds canonical logical schema objects.
- `sol01/schema/chunks.py`: renders retrieval chunks from schema objects.
- `sol01/schema/retrieval_index.py`: builds versioned lexical retrieval indexes.
- `sol01/schema/hybrid_retrieval.py`: runs lexical and exact retrieval.
- `sol01/schema/resolver.py`: resolves selected logical objects to physical table context.
- `sol01/schema/retrieval_eval.py`: evaluates offline retrieval coverage.
- `sol01/docs.py`: loads linked markdown documents.
- `sol01/llm/client.py`: runs structured LLM calls and logs raw call data.
- `sol01/prompt_builders.py`: builds the prompts for each pipeline stage.
- `sol01/coordinator.py`: runs the per-task solver pipeline.
- `sol01/candidate_evaluator.py`: validates, executes, profiles, and scores a candidate.
- `sol01/candidate_scoring.py`: produces local score breakdowns for review evidence.
- `sol01/candidate_verification.py`: adds shape, filter, aggregate, and metric-source checks.
- `sol01/validation.py`: blocks unsafe or out-of-scope SQL before execution.
- `sol01/snowflake_runner.py`: executes SQL against Snowflake.
- `sol01/output.py`: creates run folders and writes artifacts.
- `sol01/eval_runner.py`: runs and persists official evaluator output.
- `sol01/registry.py`: records local run history.
- `sol01/cli.py`: wires the user-facing commands to the solver pipeline.
- `sol01/gold_run.py`: runs one official gold SQL file.
- `progress_ui.py`: shows run progress, failures, and LLM call details.

## Boundaries

`sol01` is intentionally narrow:

- It targets Spider2-Snow, not every Spider2 dialect.
- It writes local, gitignored run artifacts.
- It uses Snowflake for execution.
- It uses the official evaluator only after choosing one final CSV per task.
- It does not push code, beads data, or run artifacts anywhere.
