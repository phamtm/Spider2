# sol01

SQLite-only Spider2-Lite solver for local tasks.

This method is scoped to Spider2-Lite instances whose `instance_id` starts with
`local`. It generates read-only SQLite SQL, executes against local SQLite
databases, writes CSV predictions, and evaluates the local subset.

## Setup

```bash
uv lock
uv run python -c "import sol01"
```

For local development, copy [`.env.example`](./.env.example) to `.env`.
The `sol01` CLI loads `methods/sol01/.env` automatically, but real shell
variables still win.

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

## Planned CLI

```bash
uv run sol01 index
uv run sol01 run --local-only
uv run sol01 eval --run-id <run_id>
uv run sol01 analyze --run-id <run_id>
uv run sol01 ask --db E_commerce "Which customers have the highest AOV?"
```

The implementation plan is tracked in `PLAN.md`.
