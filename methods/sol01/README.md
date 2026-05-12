# sol01

Snowflake-only Spider2-snow solver.

This method loads Spider2-snow tasks, generates read-only Snowflake SQL,
executes through Snowflake, writes CSV predictions, and records traces.

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
uv run sol01 prewarm-schema-index E_COMMERCE
uv run sol01 run --concurrency 4
uv run sol01 eval --run-id <run_id>
uv run sol01 retrieval-eval
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
`sol01 run` builds retrieval caches for selected databases before task workers
start. `uv run sol01 prewarm-schema-index <DB>...` builds the same cache
artifacts without running solver tasks.

`sol01 retrieval-eval` measures lexical/exact schema-retrieval coverage against
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
under `outputs/<id>/retrieval_eval/`.

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

## Schema Retrieval

`sol01` uses retrieval-first schema selection. The old `llm_only` and
full-schema planner modes no longer exist. Schema retrieval is local
lexical/exact matching over persisted schema chunks, with no separate
model-backed retrieval service.

The runtime path is:

```text
question -> lexical/exact retrieval -> planner object selection
  -> resolver expansion -> compact schema context -> SQL generation
```

This replaced full-schema selection because large Spider2-Snow databases can
have hundreds of tables or very wide tables. Sending the whole schema to the
planner produces large prompts and makes relevant tables harder to select.

Cache layout:

- `methods/sol01/.cache/snow_index.json`: base metadata cache from
  `uv run sol01 index`
- `methods/sol01/.cache/schema_retrieval_index/<DB>/current.json`: pointer to
  the active retrieval index version
- `methods/sol01/.cache/schema_retrieval_index/<DB>/versions/<cache_key>/`:
  `manifest.json`, `objects.jsonl`, `chunks.jsonl`, and `sparse.json`

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
hints. Registry validation rejects those tokens, and the retrieval cache key
includes the summary registry hash, so summary edits naturally create a new
cache version.

Runtime config can be set in the shell or `methods/sol01/.env`:

- `SOL01_SCHEMA_CHUNK_TOP_K`, default `80`
- `SOL01_SCHEMA_OBJECT_TOP_K`, default `12`
- `SOL01_SCHEMA_FAMILY_TOP_K`, default `8`
- `SOL01_SCHEMA_FAMILY_SIMILARITY_THRESHOLD`, default `0.82`
- `SOL01_SCHEMA_MAX_LINKED_DOC_CHARS`, default `6000`
- `SOL01_SCHEMA_MAX_PROMPT_CHARS`, default `24000`

Sample values are indexed only for bounded, low-cardinality categorical
evidence. High-cardinality, opaque, free-text, numeric, temporal,
semi-structured, URL, email, UUID, hash, key-like, and raw text values are
excluded so retrieval does not promote arbitrary literals into SQL prompts.

Each task trace records `schema_retrieval_version`, effective retrieval config,
retrieved object evidence, sparse/exact diagnostics, planner
sanitization diagnostics, resolver entries, allowed tables, and schema
expansion diagnostics when repair adds schema context.

To verify prompt size, inspect `schema_retrieval.prompt_budget` in
`traces/<instance_id>.json`, or run:

```bash
uv run sol01 retrieval-eval --db <DB> --limit <n> --output-id <id>
```

The persisted report under `outputs/<id>/retrieval_eval/` includes prompt
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
7. Inspect `schema_retrieval` in `traces/<instance_id>.json` for retrieval,
   planner, resolver, and expansion diagnostics.
8. Use the Streamlit `LLM calls` view in `progress_ui.py` to pick a question
   and inspect the call timeline.
9. Use `uv run sol01 llm-calls --run-id <run_id> --instance-id <instance_id>`
   for a terminal summary, or add `--call-id <call_id>` for one full call.
