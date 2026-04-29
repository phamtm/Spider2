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
Run outputs live under `methods/sol01/outputs/<run_id>/`.
The durable logs are in `logs/`, scored CSVs in `eval/scored_csv/`,
per-instance eval rows in `eval/per_instance.jsonl`, and the local registry in
`methods/sol01/outputs/registry/`.

The implementation plan is tracked in `PLAN.md`.
