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
  -> ask the LLM which tables matter
  -> extract the question intent
  -> generate several SQL candidates
  -> validate each SQL candidate
  -> execute valid candidates in Snowflake
  -> score and compare candidates
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

Before SQL generation, `sol01` builds a compact schema index from the
Spider2-Snow metadata.

For each table, the index keeps:

- the canonical Snowflake table name
- the DDL text
- column names and types
- column descriptions
- sample rows and sample values

This index is cached at `methods/sol01/.cache/snow_index.json`.

The solver preserves fully qualified Snowflake table names when they are
available, because table identity matters during validation and execution.

## Table Selection

The default retrieval path is `llm_only`.

For each question, the solver gives the LLM a compact summary of every table in
the target database. The LLM returns the smallest useful table set for the
question.

The selector is asked to include:

- tables needed for final output
- tables needed for joins
- tables needed for filters
- metric tables at the right grain

The solver then sanitizes the answer:

- unknown table names are dropped
- duplicate tables are dropped
- unambiguous suffix matches are accepted
- if no valid table survives, the selection confidence becomes `0`

## Intent Extraction

After table selection, the solver asks the LLM to rewrite the question into an
answer contract.

The contract captures:

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

If a question mentions a value that appears in selected table samples, the
solver records it as a native column value. This helps avoid turning literal
database values into invented business rules.

## SQL Generation

For each task, the coordinator usually asks for three initial SQL candidates.

Each SQL prompt includes:

- the selected table context
- DDL and column details
- sample rows
- linked document text
- the extracted answer contract
- guidance about aggregate grain and metric source when available

The prompt asks for one read-only Snowflake query. It also asks the model to
record assumptions, unsupported assumptions, and the constraint ledger behind
row narrowing, dedupe, ordering, and top-k choices.

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

## Candidate Scoring

`sol01` scores each candidate using local evidence.

The strongest signal is execution:

- executable SQL gets a large positive score
- non-executable SQL gets a large negative score

Then it adjusts the score for:

- validation quality
- expected output shape
- filter grounding
- aggregate grain
- result size plausibility
- model confidence as a small tie-breaker

This score is local solver logic. It is separate from the official benchmark
score, which runs later on the final CSV.

## Candidate Comparison

If more than one candidate executes, the solver asks the LLM to compare the
executable candidates against the answer contract.

The comparison can override the current best candidate when another executable
candidate better matches:

- output shape
- grain
- filters
- table usage
- metric source
- grounded assumptions

This keeps candidate choice from being purely numeric when two queries both run.

## Repairs

`sol01` has three repair paths.

### Execution Repair

If the best candidate does not execute, the solver asks for a SQL repair using
the validation and execution feedback.

### Aggregate Repair

If a candidate returns a suspicious aggregate result, such as zero or a tiny
count, the solver asks a verifier whether the result is trustworthy.

If the verifier says repair is needed, the solver asks for an aggregate-focused
repair.

### Semantic Repair

After a best executable candidate is chosen, a critic reviews whether the SQL
answers the question.

The critic looks for concrete issues such as:

- ungrounded filters
- wrong shape
- missing filters
- suspicious aggregates
- wrong metric source
- literal values treated as invented business definitions

If the critic recommends repair, the solver asks for one semantic repair.

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
prompt hashes, attempts, scores, comparison output, verifier output, critic
output, final SQL, and final execution summary.

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
- `sol01/index.py`: builds the Snowflake schema index.
- `sol01/retrieval.py`: selects tables for a question.
- `sol01/docs.py`: loads linked markdown documents.
- `sol01/llm.py`: runs structured LLM calls and logs raw call data.
- `sol01/prompt_builders.py`: builds the prompts for each pipeline stage.
- `sol01/coordinator.py`: runs the per-task solver pipeline.
- `sol01/candidate_evaluator.py`: validates, executes, profiles, and scores a candidate.
- `sol01/candidate_scoring.py`: chooses the best local attempt.
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
