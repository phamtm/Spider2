# sol01 Solution Overview

`sol01` is a Snowflake-only solver for Spider2-Snow.

It reads benchmark questions, asks an LLM to write Snowflake SQL, runs the SQL
against Snowflake, saves the result CSV, and scores that CSV with the official
Spider2-Snow evaluator.

The main goal is not just to get an answer. The goal is to make every run
auditable: the selected schema, prompts, candidate SQL, validation result,
execution result, final CSV, official eval output, and raw LLM calls are all
saved under `methods/sol01/outputs/`.

Default runtime choices, solver budgets, prompt/schema caps, filter-grounding
caps, and eval metadata now live in `sol01/infra/policy.py`. Runtime modules
use those defaults, but they no longer hide the policy inside local constants.

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
  -> prepare schema context objects
  -> ask the LLM to select from available objects and describe question intent
  -> resolve selected logical objects to physical Snowflake tables
  -> render compact schema context for SQL generation
  -> generate several SQL candidates in one batch
  -> validate each SQL candidate
  -> execute valid candidates in Snowflake
  -> attach local observations and score breakdowns
  -> run one recovery stage when needed
  -> write final SQL and CSV
  -> run official Spider2-Snow eval
  -> save run, trace, eval, and registry artifacts
```

## Main Moving Parts

- `sol01/infra/policy.py`: default runtime profile and heuristic caps
- `sol01/infra/config.py`: env loading and the fixed runtime contract
- `sol01/pipeline.py`: high-level per-task stage flow
- `sol01/pipeline_recovery.py`: recovery orchestration
- `sol01/pipeline_output.py`: final SQL / CSV / trace writing
- `sol01/pipeline_support.py`: shared prompt, budget, and candidate-recording helpers
- `sol01/llm/planning_prompts.py`: planner prompt assembly and planner-output cleanup
- `sol01/llm/sql_prompts.py`: SQL generation, repair, and review prompts
- `sol01/recovery_signals.py`: schema-expansion trigger detection
- `sol01/analysis/eval_runner.py`: official evaluator wrapper and persisted summaries

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
  The public runtime knobs are the API key, base URL, model, and concurrency.
  Provider routing stays pinned to DeepSeek on OpenRouter by design.
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

The schema-context cache is built per database from the base metadata cache. It
contains canonical schema objects and a version manifest:

```text
methods/sol01/.cache/schema_context_cache/<DB>/
  current.json
  versions/<cache_key>/
    manifest.json
    objects.jsonl
```

Build schema-context caches before a batch run with either command:

```bash
uv run sol01 prewarm-schema-context E_COMMERCE
uv run sol01 run --db E_COMMERCE
```

The schema-context cache key includes the source schema hash, schema-object
builder version, curated summary registry hash/version,
and family similarity threshold. If one of those inputs changes, a new version
directory is created and `current.json` is updated.

## Schema Context Planning

Runtime schema context uses deterministic schema objects with no question-time
retrieval or ranking. Curated large-schema summaries switch the planner into a
summary-only mode, so repeated table families and very wide schemas stay
compact. For databases without summary coverage, the planner sees the full
built logical object set. There is no dense embedding, BM25, or separate
model-backed search service in the runtime path.

This keeps planning grounded while removing retrieval-era complexity. Large
Spider2-Snow databases can still stay compact through curated summaries, while
ordinary databases no longer depend on a ranked shortlist.

The runtime pipeline is:

```text
question
  -> deterministic schema metadata context
  -> LLM planner selects available logical objects
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

Linked markdown documents are clipped to passages that overlap with the
question before planning, rather than being sent wholesale.

For each question, the planner sees planner-visible schema metadata evidence.
That evidence is either curated summaries only or the full logical object set
for the database. It returns selected object IDs, object roles, constraints,
and the answer contract in one structured response.

The solver sanitizes the planner output:

- object IDs outside the available schema metadata are dropped
- table names are normalized only when they match available metadata
- if no valid schema object survives, confidence is lowered
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

Sample values are included only when they are bounded, low-cardinality
categorical evidence. Schema context excludes high-cardinality, opaque,
free-text, and sensitive-looking values because they can leak irrelevant
literals into prompts and encourage the model to treat arbitrary examples as
business rules.

Sample-value objects are excluded for:

- key-like columns such as IDs, UUIDs, hashes, and foreign keys
- temporal columns and date-like values
- numeric, temporal, and semi-structured column types
- raw text, description, body, email, URL, JSON, payload, and similar columns
- values longer than the short label/code threshold
- columns with too many distinct sample values
- columns where every sampled value is distinct unless the column name is
  explicitly categorical

Included sample values are schema metadata evidence. If a question mentions an
included sample value, the solver can keep that value tied to its native column
instead of converting it into an invented rule.

## Curated Large-Schema Summaries

Curated summaries live in
`methods/sol01/metadata/large_schema_summaries.json`. They compact repeated
table families and wide repeated column groups before planner selection and
resolver expansion.

Add or edit a summary when a schema family has a stable shape that should be
represented as one logical object. Each entry defines:

- `summary_id`: lower_snake_case stable ID
- `schema_copies`: database/schema locations covered by the same shape
- `match`: either exact `table_names` or a regex `table_pattern`, optionally
  with an inclusive suffix range
- `purpose` and `grain`
- stable columns, repeated-column rules, inclusive ranges, quoting rules,
  examples, and aliases

Summaries must describe schema shape only. Do not include benchmark answers,
gold SQL, instance IDs, or question-specific hints. The registry validator
rejects those tokens, and the schema-context cache key includes the summary registry
hash, so a summary edit creates a fresh cache version.

Relevant runtime settings:

- `SOL01_SCHEMA_FAMILY_SIMILARITY_THRESHOLD`, default `0.82`
- `SOL01_SCHEMA_MAX_LINKED_DOC_CHARS`, default `6000`
- `SOL01_SCHEMA_MAX_PROMPT_CHARS`, default `24000`

These can be set in the shell or in `methods/sol01/.env`. They override the
defaults from `infra/policy.py`; the remaining heuristic caps stay code-defaulted
so the method has one clear baseline policy surface.

`SOL01_SCHEMA_CONTEXT_OBJECT_CUTOFF` is eval-only. Use
`uv run sol01 schema-context-eval --object-cutoff <n>` when you want to change
how many planner-visible objects the offline coverage report measures.

## Schema Context Evaluation

Offline schema-context coverage can be checked with:

```bash
uv run sol01 schema-context-eval
uv run sol01 schema-context-eval --db E_COMMERCE --json
```

The command reports pre-resolver any-gold recall, post-resolver all-gold
recall, family expansion success, and prompt reduction. By default it reads
offline labels from `methods/gold-tables/spider2-snow-gold-tables.jsonl`;
`--gold-path <path>` is only for alternate local label files. Gold tables are
offline-only labels for measuring schema-context coverage. They are not available to
runtime planning, SQL generation, or repair.

Use `--covered-only` to focus on tasks whose gold tables touch curated
large-schema summaries. Use `--baseline-path <report.json|tasks.jsonl>` to
check for recall regressions, `--trace-run-id <run_id>` to scan saved solver
traces for hallucinated-column validation failures, and `--output-id <id>` to
persist `summary.json`, `tasks.jsonl`, `failures.json`, `report.json`, and
`summary.md` under `outputs/<id>/schema_context_eval/`.

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
offline schema-context-eval command can also scan saved traces with
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
official benchmark score, which runs later on the final CSV, and it is the
runtime tie-breaker for choosing the final answer among executable candidates.

## Repairs

`sol01` has one recovery stage after the initial candidate batch.

The recovery stage uses a simple priority order:

- schema recovery first when the trace shows the selected schema is incomplete
- SQL recovery when the best candidate still does not execute

All recovery attempts share the same task-level attempt budget.

### SQL Recovery

If the best candidate does not execute, the solver asks for a SQL repair using
the validation and execution feedback.

### Schema Recovery

When validation or execution evidence shows that the selected schema is
incomplete, the solver expands the schema context first and retries SQL
generation against that broader context.

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
prompt hashes, attempts, recovery actions, final SQL, and final execution
summary.

Schema-context traces also include:

- `schema_context.cache.cache_key`
- context mode and object counts
- available object IDs and ranked planning evidence
- planner sanitization diagnostics
- resolver entries, allowed tables, and resolver warnings
- recovery diagnostics when schema recovery expands schema context

Prompt budget diagnostics live in `schema_context.prompt_budget` inside each
task trace. They record planning and resolved-context character counts against
`SOL01_SCHEMA_MAX_PROMPT_CHARS`. `uv run sol01 schema-context-eval --output-id <id>`
also writes prompt reduction and prompt-size wins to
`outputs/<id>/schema_context_eval/`.

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
uv run sol01 schema-context-eval --limit 20
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

- `sol01/loading/tasks.py`: loads and filters Spider2-Snow tasks.
- `sol01/loading/docs.py`: loads linked markdown documents.
- `sol01/schema/index.py`: builds the Snowflake metadata cache.
- `sol01/schema/objects.py`: assembles canonical logical schema objects.
- `sol01/schema/object_text.py`: builds planner/search text for schema objects.
- `sol01/schema/reference_context.py`: renders compact schema context for SQL generation.
- `sol01/schema/schema_context_cache.py`: builds versioned schema-context caches.
- `sol01/schema/schema_context.py`: ranks planner-visible schema objects.
- `sol01/schema/resolver.py`: resolves selected logical objects to physical table context.
- `sol01/analysis/schema_context_eval.py`: evaluates offline schema-context coverage.
- `sol01/llm/client.py`: runs structured LLM calls and logs raw call data.
- `sol01/llm/prompt_builders.py`: builds the prompts for each pipeline stage.
- `sol01/llm/llm_call_logs.py`: reads persisted LLM call logs for CLI/UI inspection.
- `sol01/coordinator.py`: runs batches and keeps `run_task()` as the workflow shell.
- `sol01/pipeline.py`: owns per-task planning, candidate, repair, and output stages.
- `sol01/candidates/evaluator.py`: validates, executes, profiles, and scores a candidate.
- `sol01/candidates/scoring.py`: produces local score breakdowns for review evidence.
- `sol01/candidates/verification.py`: adds shape, filter, aggregate, and metric-source checks.
- `sol01/execution/validation.py`: blocks unsafe or out-of-scope SQL before execution.
- `sol01/execution/snowflake_runner.py`: executes SQL against Snowflake.
- `sol01/output/output.py`: creates run folders and writes artifacts.
- `sol01/output/registry.py`: records local run history.
- `sol01/analysis/eval_runner.py`: runs and persists official evaluator output.
- `sol01/analysis/gold_run.py`: runs one official gold SQL file.
- `sol01/analysis/analysis.py`: summarizes persisted solver runs.
- `sol01/cli.py`: wires the user-facing commands to the solver pipeline.
- `sol01/progress_ui/app.py`: shows run progress, failures, and LLM call details.

## Boundaries

`sol01` is intentionally narrow:

- It targets Spider2-Snow, not every Spider2 dialect.
- It writes local, gitignored run artifacts.
- It uses Snowflake for execution.
- It uses the official evaluator only after choosing one final CSV per task.
- It does not push code, beads data, or run artifacts anywhere.
