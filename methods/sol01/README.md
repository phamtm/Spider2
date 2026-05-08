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
uv run sol01 run --concurrency 4
uv run sol01 eval --run-id <run_id>
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
7. Use the Streamlit `LLM calls` view in `progress_ui.py` to pick a question and inspect the call timeline.
8. Use `uv run sol01 llm-calls --run-id <run_id> --instance-id <instance_id>` for a terminal summary, or add `--call-id <call_id>` for one full call.
