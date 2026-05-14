# sol01

Snowflake-only Spider2-snow solver.

This method loads Spider2-snow tasks, generates read-only Snowflake SQL,
executes through Snowflake, writes CSV predictions, and records traces.

The default runtime profile and heuristic caps live in
`methods/sol01/sol01/infra/policy.py`. Use that file when you want to change
the method's default behavior. Use shell variables or CLI flags when you only
want a local override for one run or one machine.

## Setup

```bash
uv lock
uv run python -c "import sol01"
```

For local development, copy [`.env.example`](./.env.example) to `.env`.
The `sol01` CLI loads `methods/sol01/.env` automatically, but real shell
variables still win.
Set `SOL01_CONCURRENCY` to control batch worker count. The default is `4`.
Use `--concurrency` on `sol01 run` to override the environment for one run.
Concurrency is task-level per question batch, not per database.

Create `methods/sol01/snowflake_credential.json` locally with a programmatic
access token:

```json
{
  "username": "<your_username>",
  "password": "<your_generated_token>",
  "account": "RSRSBDK-YDB67606",
  "role": "PARTICIPANT",
  "warehouse": "COMPUTE_WH_PARTICIPANT"
}
```

Pydantic Logfire is enabled by default for CLI runs.

## Policy Surface

`sol01` keeps the main default policy in one place:

- runtime profile defaults: base URL, model, fixed OpenRouter routing, concurrency
- solver budgets: initial candidates, max attempts, semantic repairs
- schema prompt defaults: family threshold, linked-doc budget, total prompt budget
- prompt shrink strategy: what gets cut first when planning prompts are too large
- schema render caps: family previews, evidence lines, column previews, sample literal length
- filter-grounding caps: probe target count, fallback columns, probe row limit
- eval metadata: default dataset size and schema-context-eval report bounds

That keeps implementation modules focused on behavior instead of hiding policy
inside local constants.

## Quality

```bash
uv run --group dev ruff format .
uv run --group dev ruff check .
```

With `just`:

```bash
just fmt
just lint
just test
just check
```

## Command Line

```bash
uv run sol01 index
uv run sol01 prewarm-schema-context E_COMMERCE
uv run sol01 run --concurrency 4
uv run sol01 eval --run-id <run_id>
uv run sol01 schema-context-eval
uv run sol01 analyze --run-id <run_id>
uv run sol01 ask --db E_COMMERCE "Which customers have the highest AOV?"
uv run sol01 run --instance-id sf_bq320
uv run sol01 run sf035 sf_bq135 sf_bq084
uv run sol01 run --db E_COMMERCE --question-contains revenue
just run-selected sf035 sf_bq135
just gold sf_bq320
```

`sol01 run` accepts `--concurrency <n>`. If omitted, the CLI uses
`SOL01_CONCURRENCY` or the default value of `4`.
`sol01 run <selector>...` accepts exact IDs, globs, `tier:<n>`,
`tag:<name>`, or `all`.
`sol01 run` builds schema metadata caches for selected databases before task
workers start. `uv run sol01 prewarm-schema-context <DB>...` builds the same cache
artifacts without running solver tasks.

`sol01 schema-context-eval` measures deterministic schema-context coverage against
the offline gold-table JSONL file at
`methods/gold-tables/spider2-snow-gold-tables.jsonl` by default. Use
`--gold-path <path>` only when evaluating another local label file. Gold tables
are evaluation labels only; runtime planning, SQL generation, repair, and
candidate review do not receive gold tables.
Use `--covered-only` to limit the report to gold tables covered by curated
large-schema summaries. `--baseline-path <report.json|tasks.jsonl>` compares
recall to a previous report, `--trace-run-id <run_id>` scans saved traces for
hallucinated-column validation failures, and `--output-id <id>` writes
`summary.json`, `tasks.jsonl`, `failures.json`, `report.json`, and `summary.md`
under `outputs/<id>/schema_context_eval/`.

`just run` runs the default solver CLI.
`just run-selected` runs one or more selected solver tasks.
`just gold` is only for exact instance IDs and runs the persisted gold SQL path.
Gold runs reuse the same outputs root, but they do not populate
`eval/scored_csv/` because the gold CSV is already the scored input.

## Output Layout

`methods/sol01/outputs/` is gitignored.

Each persisted solver run writes to `methods/sol01/outputs/<run_id>/`:

- `logs/stdout.txt`
- `logs/stderr.txt`
- `logs/run.jsonl`
- `sql/`
- `csv/`
- `traces/`
- `eval/scored_csv/`
- `eval/summary.json`
- `eval/per_instance.jsonl`
- `eval/runs/default/command.json`
- `eval/runs/default/stdout.txt`
- `eval/runs/default/stderr.txt`
- `eval/runs/default/summary.json`
- `eval/runs/default/per_instance.jsonl`
- `eval/runs/default/input_csv/`
- `eval/runs/default/input_csv.csv`
- `eval/runs/default/workspace/temp/`
- `eval/runs/default/workspace/spider2-snow/evaluation_suite/log.txt`
- `eval/runs/<filtered-tag>/` for filtered official eval invocations, with the same layout

The top-level `eval/summary.json` and `eval/per_instance.jsonl` files are
current summaries for the latest official eval state. The `eval/runs/*/`
folders are the audit trail.
The evaluator workspace is refreshed on each invocation, including
`eval/runs/<eval_id>/workspace/temp/`.

The local registry lives in `methods/sol01/outputs/registry/` with:

- `runs.jsonl`
- `task_results.jsonl`
- `latest.json`

## Schema Context

`sol01` uses deterministic schema objects with no question-time retrieval or
ranking. If a database has curated large-schema summary coverage, planning sees
only those summary-backed logical objects. Otherwise, planning sees the full
logical metadata object set for the database.

The runtime path is:

```text
question -> schema metadata context
  -> planner object selection
  -> resolver expansion -> compact schema context -> SQL generation
```

The runtime code follows the same split:

- `sol01/pipeline.py`: high-level per-task stage flow
- `sol01/pipeline_recovery.py`: recovery orchestration
- `sol01/pipeline_output.py`: final SQL / CSV / trace writing
- `sol01/pipeline_support.py`: shared prompt, budget, and candidate-recording helpers
- `sol01/llm/planning_prompts.py`: planner prompt assembly and planner-output cleanup
- `sol01/llm/sql_prompts.py`: SQL generation, repair, and review prompts
- `sol01/recovery_signals.py`: schema-expansion trigger detection

Large-schema summaries keep very wide or repeated schemas compact. For ordinary
schemas, the planner works from the full logical metadata object set instead of
from a ranked shortlist.

Cache layout:

- `methods/sol01/.cache/snow_index.json`: base metadata cache from
  `uv run sol01 index`
- `methods/sol01/.cache/schema_context_cache/<DB>/current.json`: pointer to
  the active schema-context version
- `methods/sol01/.cache/schema_context_cache/<DB>/versions/<cache_key>/`:
  `manifest.json` and `objects.jsonl`

Large-schema summaries live in
`methods/sol01/metadata/large_schema_summaries.json`. Add or edit one summary
when a schema has repeated table families or very wide repeated column groups
that should be represented compactly. Each summary must define:

- `summary_id`: lower_snake_case stable ID
- `schema_copies`: database/schema locations covered by the same shape
- `match`: exact `table_names` or a regex `table_pattern`, optionally with an
  inclusive suffix range
- `purpose`, `grain`, stable columns, repeated-column rules, quoting rules,
  examples, and aliases

Do not include benchmark answers, gold SQL, instance IDs, or question-specific
hints. Registry validation rejects those tokens, and the schema-context cache
key includes the summary registry hash, so summary edits naturally create a new
cache version.

Runtime schema-context settings can be set in the shell or `methods/sol01/.env`:

- `SOL01_SCHEMA_FAMILY_SIMILARITY_THRESHOLD`, default `0.82`
- `SOL01_SCHEMA_MAX_LINKED_DOC_CHARS`, default `6000`
- `SOL01_SCHEMA_MAX_PROMPT_CHARS`, default `24000`

Those env-configured values override the defaults from `infra/policy.py`. The
remaining prompt and render heuristics stay code-defaulted on purpose so the
runtime path has one explicit source of truth.

`SOL01_SCHEMA_CONTEXT_OBJECT_CUTOFF` is not a runtime setting. It is an
offline `sol01 schema-context-eval` option, exposed as `--object-cutoff`.

Sample-value objects are built only for bounded, low-cardinality categorical
evidence. High-cardinality, opaque, free-text, numeric, temporal,
semi-structured, URL, email, UUID, hash, key-like, and raw text values are
excluded so arbitrary literals are not promoted into SQL prompts.

Each task trace records cache-backed schema-context provenance, including
`schema_context.cache.cache_key`, planner-visible schema-object evidence,
context diagnostics, planner sanitization diagnostics, resolver entries,
allowed tables, and recovery diagnostics when schema recovery adds schema
context.

To verify prompt size, inspect `schema_context.prompt_budget` in
`traces/<instance_id>.json`, or run:

```bash
uv run sol01 schema-context-eval --db <DB> --limit <n> --output-id <id>
```

The persisted report under `outputs/<id>/schema_context_eval/` includes prompt
reduction and prompt-size wins. Add `--trace-run-id <run_id>` to scan existing
solver traces for hallucinated-column validation failures. Runtime validation
also blocks SQL that references unknown columns when selected schema context is
specific enough, so those failures are caught before Snowflake execution.

## Statuses

Pass/fail comes from the official evaluator:

- `pass`: CSV scored 1 by the official evaluator
- `official_fail`: CSV was present, but the official evaluator scored it 0
- `solver_failed`: the solver task itself failed
- `missing_csv`: the solver never produced a CSV
- `eval_failed`: the evaluator failed before scoring finished

## Debugging

1. Read `logs/run.jsonl` for the run sequence.
2. Check `logs/stdout.txt` and `logs/stderr.txt`.
3. Inspect `sql/`, `csv/`, `traces/`, and `llm_calls/` for the task-level artifacts.
4. Open `eval/runs/default/command.json`, `stdout.txt`, `stderr.txt`,
   `summary.json`, `per_instance.jsonl`, `input_csv/`, `input_csv.csv`,
   `workspace/temp/`, and `workspace/spider2-snow/evaluation_suite/log.txt`.
5. Open `eval/summary.json` and `eval/per_instance.jsonl`.
6. Check `methods/sol01/outputs/registry/latest.json`.
7. Inspect `schema_context` in `traces/<instance_id>.json` for selection,
   planner, resolver, and expansion diagnostics.
8. Use the Streamlit `LLM calls` view in `sol01/progress_ui/app.py` to pick a question
   and inspect the call timeline.
9. Use `uv run sol01 llm-calls --run-id <run_id> --instance-id <instance_id>`
   for a terminal summary, or add `--call-id <call_id>` for one full call.
