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
uv run sol01 run
uv run sol01 eval --run-id <run_id>
uv run sol01 analyze --run-id <run_id>
uv run sol01 ask --db E_COMMERCE "Which customers have the highest AOV?"
```

Persisted run mode uses `just`:

```bash
just run sf_bq320
just run 'sf_bq3*' 'sf_bq4*'
just all
just smoke sf_bq320
```

Use quotes around patterns so the shell does not expand them first.
Bare `*` is rejected by the helper on purpose.
`just smoke` is only for exact instance IDs and runs the gold SQL smoke path.

## Output Layout

`methods/sol01/outputs/` is gitignored.

Each persisted run writes to `methods/sol01/outputs/<run_id>/`:

- `logs/stdout.txt`
- `logs/stderr.txt`
- `logs/run.jsonl`
- `sql/`
- `csv/`
- `traces/`
- `eval/scored_csv/`
- `eval/summary.json`
- `eval/per_instance.jsonl`

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
3. Inspect `sql/`, `csv/`, and `traces/` for the task-level artifacts.
4. Open `eval/summary.json` and `eval/per_instance.jsonl`.
5. Check `methods/sol01/outputs/registry/latest.json`.

The implementation plan is tracked in `PLAN.md`.
