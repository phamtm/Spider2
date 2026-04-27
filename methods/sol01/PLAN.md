# sol01 Implementation Plan

**Goal:** Build `methods/sol01`, a SQLite-only Spider2-Lite solver that generates SQL with Pydantic AI and DeepSeek via OpenRouter, executes read-only SQLite queries, writes CSV outputs, and evaluates the local subset.

**Architecture:** A deterministic Python coordinator owns task loading, retrieval, validation, execution, retries, traces, resume behavior, and output writing. Pydantic AI is used only for bounded structured LLM calls: intent, schema selection, SQL generation, SQL repair, and result critique.

**Tech Stack:** Python 3.11+, uv, Pydantic, Pydantic AI, OpenRouter OpenAI-compatible API, DeepSeek model, SQLite, pandas, sqlglot, pytest, local JSONL/JSON traces.

---

## Scope

`sol01` targets only Spider2-Lite local SQLite tasks.

- Include tasks where `instance_id` starts with `local`.
- Ignore BigQuery and Snowflake tasks.
- Report local subset score as `correct / 135`.
- Also report full benchmark equivalent as `correct / 547`.
- Do not use `methods/gold-tables`, gold SQL, gold execution results, or evaluator metadata inside generation.
- Use gold outputs only through offline evaluation after predictions are written.

## Core Decisions

- Method folder: `methods/sol01`.
- Package style: self-contained Python package with `pyproject.toml` and committed `uv.lock`.
- Runtime: `uv`.
- Default model provider: OpenRouter.
- Default model: DeepSeek through OpenRouter, pinned to provider `deepseek`.
- Provider fallback: fail fast.
- Tracing: local JSON and JSONL only.
- Prompt storage: versioned markdown files with prompt hashes recorded in traces.
- Submission artifact: CSV execution results as primary output, SQL files as secondary debug artifacts.
- Validation: strict read-only SQLite validation before every execution.
- SQL form: single `SELECT` or `WITH ... SELECT`; no temp tables or views.
- Retry budget: 3 initial candidates, 4 attempts max, repair only the best candidate.
- Critic: deterministic checks plus one DeepSeek semantic critic repair max.
- CSV normalization: minimal, `pandas.to_csv(index=False)`.
- Resume: on by default.
- Default concurrency: 2.
- Smoke set: `local003`, `local004`, `local007`.

## Runtime Config

Environment variables:

```bash
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=deepseek/deepseek-v4-pro
OPENROUTER_PROVIDER_ONLY=deepseek
OPENROUTER_ALLOW_FALLBACKS=false
```

Generic aliases should also work:

```bash
LLM_API_KEY=...
LLM_BASE_URL=...
LLM_MODEL=...
```

OpenRouter request policy:

```json
{
  "provider": {
    "only": ["deepseek"],
    "allow_fallbacks": false
  }
}
```

## Output Layout

```text
methods/sol01/outputs/
  <run_id>/
    manifest.json
    sql/
      local003.sql
    csv/
      local003.csv
    traces/
      local003.json
    eval/
      official_stdout.txt
      summary.json
    analysis/
      failures.json
      summary.md
  ask/
    <timestamp>/
      sql.sql
      result.csv
      trace.json
```

## CLI

```bash
uv run sol01 index

uv run sol01 run --local-only
uv run sol01 run --instance-id local003
uv run sol01 run --instance-id local003 --run-id smoke-local003
uv run sol01 run --db E_commerce
uv run sol01 run --question-contains "retention"
uv run sol01 run --limit 10

uv run sol01 eval --run-id <run_id>
uv run sol01 eval --run-id <run_id> --instance-id local003
uv run sol01 eval --run-id <run_id> --db E_commerce

uv run sol01 analyze --run-id <run_id>

uv run sol01 ask --db E_commerce "Which customers have the highest AOV?"
```

`ask` writes to `outputs/ask/<timestamp>` by default. It should only write into a benchmark run folder when `--run-id` is explicitly provided.

## Architecture Diagram

```text
                         +----------------------+
                         | CLI                  |
                         | sol01/cli.py, run.py |
                         +----------+-----------+
                                    |
          +-------------------------+-------------------------+
          |                         |                         |
          v                         v                         v
+-------------------+     +-------------------+     +-------------------+
| Index command     |     | Coordinator       |     | Eval / analysis   |
| sol01/index.py    |     | sol01/coordinator |     | eval_runner.py    |
+---------+---------+     +---------+---------+     | analysis.py       |
          |                         |               +---------+---------+
          v                         |                         ^
+-------------------+               |                         |
| Metadata cache    |               |                         |
| .cache/index.json |               |                         |
+---------+---------+               |                         |
          |                         |                         |
          v                         v                         |
+-------------------+     +-------------------+               |
| Schema retrieval  |---->| Task execution    |               |
| sol01/retrieval.py|     | loop              |               |
+-------------------+     +---------+---------+               |
                                    |                         |
      +-----------------------------+------------------+      |
      |                             |                  |      |
      v                             v                  v      |
+------------+              +--------------+     +-----------+|
| Task loader|              | Document     |     | LLM client||
| tasks.py   |              | retrieval    |     | llm.py    ||
+------------+              | docs.py      |     +-----+-----+|
                            +--------------+           ^      |
                                                       |      |
                                                +------+-----+|
                                                | Prompt     ||
                                                | files      ||
                                                +------------+|
                                    |
                                    v
                            +---------------+
                            | SQL validation|
                            | validation.py |
                            +-------+-------+
                                    |
                                    v
                            +---------------+
                            | SQLite runner |
                            | sqlite_runner |
                            +-------+-------+
                                    |
                                    v
                            +---------------+
                            | Profiling     |
                            | profiling.py  |
                            +-------+-------+
                                    |
                                    v
                            +---------------+
                            | Output/resume |
                            | output.py     |
                            +-------+-------+
                                    |
                                    v
                            +---------------+
                            | Run files     |
                            | outputs/...   |
                            +-------+-------+
                                    |
                                    +-------------------------+

Shared inputs:

  config.py  -> CLI and coordinator
  models.py  -> all module boundaries
```

## Component Responsibilities

### CLI (`sol01/cli.py`, `run.py`)

Owns user-facing commands and argument parsing. It should stay thin: resolve command options, load config, call the right module, and print concise results.

- `index`: builds or refreshes the metadata index.
- `run`: executes benchmark tasks through the coordinator.
- `eval`: evaluates saved CSV predictions with the official evaluator wrapper.
- `analyze`: summarizes traces and evaluation results.
- `ask`: runs one ad hoc database question outside the benchmark flow unless `--run-id` is explicitly provided.

### Config (`sol01/config.py`)

Loads runtime settings from environment variables and CLI overrides. It should normalize OpenRouter and generic `LLM_*` aliases into one config object, enforce DeepSeek-only provider routing by default, and fail fast when required secrets are missing for live runs.

### Typed Models (`sol01/models.py`)

Defines the shared Pydantic objects passed between modules. These models are the contract for tasks, retrieved schemas, metric definitions, SQL candidates, validation reports, execution results, critic output, and final answers.

### Task Loader (`sol01/tasks.py`)

Reads `spider2-lite/spider2-lite.jsonl`, keeps only local SQLite tasks, and applies debug filters such as `--instance-id`, `--db`, `--question-contains`, and `--limit`. It must not read gold SQL, gold execution results, or evaluator metadata.

### Metadata Index (`sol01/index.py`)

Parses SQLite metadata from `DDL.csv` and per-table JSON files into `TableSchema` objects. It builds a cached index under `methods/sol01/.cache/index.json` so retrieval and prompts can use compact, structured schema context.

### Document Retrieval (`sol01/docs.py`)

Loads allowed markdown documents from `spider2-lite/resource/documents`. It chunks documents by heading, table, and paragraph blocks, force-loads task `external_knowledge` when present, and returns structured metric definitions instead of dumping full documents into prompts.

### Schema Retrieval (`sol01/retrieval.py`)

Ranks tables inside the task database using lexical overlap across table names, column names, descriptions, sample values, and the question. It then asks the LLM to select from compact candidates and expands selected tables with conservative join-neighbor rules.

This is the v1 path. After the main solver is working end to end, add one follow-up experiment that skips lexical ranking and asks the LLM to choose tables directly from the DB schema summary. Keep that experiment separate so we can compare recall, prompt size, and SQL quality against the lexical-first path.

### Prompt Files (`prompts/*.md`)

Stores versioned prompt text for intent extraction, schema selection, SQL generation, SQL repair, and result critique. Prompt SHA-256 hashes must be recorded in traces for reproducibility.

### LLM Client (`sol01/llm.py`)

Wraps Pydantic AI calls to OpenRouter. It owns prompt loading, prompt hashing, provider routing, structured response parsing, and fake-client seams for tests. Unit tests should not call the live API.

### Coordinator (`sol01/coordinator.py`)

Owns the per-task execution loop. It loads task context, retrieves schema and documents, calls the LLM for intent and SQL candidates, validates and executes candidates, chooses the best result, runs bounded repair, and records all attempts in the trace.

### SQLite Validation (`sol01/validation.py`)

Uses `sqlglot` with SQLite dialect to enforce read-only SQL before execution. It allows one `SELECT` or `WITH ... SELECT`, rejects mutation, DDL, `ATTACH`, `DETACH`, `PRAGMA`, extension loading, and chained statements, and checks referenced tables against local metadata.

### SQLite Runner (`sol01/sqlite_runner.py`)

Executes validated SQL against an in-memory backup of the target SQLite file. It writes CSV output with `pandas.to_csv(index=False)` and returns structured execution results.

### Profiling (`sol01/profiling.py`)

Creates cheap result profiles for candidate scoring and critique: row count, columns, sample rows, null counts, distinct counts, min/max, and top values where bounded and useful.

### Output and Resume (`sol01/output.py`)

Creates run directories, writes manifests, SQL files, CSV files, and traces, and decides whether a task should be skipped or rerun. Resume behavior is based on both a successful final trace and an existing CSV.

### Eval Wrapper (`sol01/eval_runner.py`)

Runs the official evaluator in `exec_result` mode against saved CSV predictions. It must use the active `sol01` Python interpreter, save stdout, parse correct and total counts when possible, and write `outputs/<run_id>/eval/summary.json`.

### Analysis (`sol01/analysis.py`)

Reads traces and evaluation summaries after a run. It groups failures by validation, execution, empty result, critic, missing CSV, retrieval miss, aggregation issue, date-filter issue, and database, then writes `failures.json` and `summary.md`.

## Data Sources

Allowed at generation time:

- `spider2-lite/spider2-lite.jsonl`
- `spider2-lite/resource/databases/sqlite/<db>/DDL.csv`
- `spider2-lite/resource/databases/sqlite/<db>/*.json`
- `spider2-lite/resource/databases/spider2-localdb/<db>.sqlite`
- `spider2-lite/resource/documents/*.md`, only when task-aware retrieval selects them

Disallowed at generation time:

- `methods/gold-tables/*`
- `spider2-lite/evaluation_suite/gold/sql/*`
- `spider2-lite/evaluation_suite/gold/exec_result/*`
- `spider2-lite/evaluation_suite/gold/spider2lite_eval.jsonl`
- evaluator fields such as `condition_cols`, `ignore_order`, or temporal flags

## Typed Objects

Implement these as Pydantic models.

```python
class Task(BaseModel):
    instance_id: str
    db: str
    question: str
    external_knowledge: str | None = None

class Intent(BaseModel):
    summary: str
    entities: list[str]
    metrics: list[str]
    filters: list[str]
    time_constraints: list[str]
    output_expectation: str
    assumptions: list[str]

class ColumnSchema(BaseModel):
    name: str
    type: str | None = None
    description: str | None = None
    sample_values: list[str] = []

class TableSchema(BaseModel):
    name: str
    ddl: str
    columns: list[ColumnSchema]
    sample_rows: list[dict[str, object]] = []
    searchable_text: str

class SchemaSelection(BaseModel):
    db: str
    selected_tables: list[str]
    expanded_tables: list[str]
    rationale: str
    confidence: float

class MetricDefinition(BaseModel):
    metric_name: str
    source_file: str | None = None
    heading: str | None = None
    definition: str
    formula: str | None = None
    sql_notes: str | None = None
    confidence: float

class SQLCandidate(BaseModel):
    sql: str
    explanation: str
    assumptions: list[str]
    confidence: float

class ValidationReport(BaseModel):
    ok: bool
    errors: list[str]
    warnings: list[str]
    referenced_tables: list[str]

class ExecutionResult(BaseModel):
    ok: bool
    row_count: int
    columns: list[str]
    sample_rows: list[dict[str, object]]
    csv_path: str | None = None
    error: str | None = None

class ConfidenceReport(BaseModel):
    confidence: float
    issues: list[str]
    should_repair: bool
    repair_focus: str | None = None

class FinalAnswer(BaseModel):
    instance_id: str
    status: Literal["success", "failed", "skipped"]
    sql: str | None
    csv_path: str | None
    trace_path: str
```

## Retrieval Policy

### Schema Retrieval

Use DB-constrained hybrid retrieval:

1. Resolve local SQLite DB from task `db`.
2. Load only `spider2-lite/resource/databases/sqlite/<db>`.
3. Parse `DDL.csv` and per-table JSON metadata.
4. Rank tables by lexical overlap over table name, column names, descriptions, sample values, and question text.
5. Ask DeepSeek to choose from compact candidates.
6. Expand selected tables with conservative join-neighbor rules.
7. Cap schema context with `--max-schema-tables`, default `12`.

Join-neighbor expansion:

- Exact column name match, such as `customer_id`.
- Table key pattern, such as `customers.id` to `orders.customer_id`.
- Bridge table hints, such as `_map`, `_relationship`, `_details`, `_items`.

### Document and Metric Retrieval

Use task-aware deterministic retrieval:

1. If `external_knowledge` is present, force-load that document.
2. Search inside that document first.
3. Search the full documents corpus only if no task doc exists or the metric is ambiguous.
4. Do not dump docs into every prompt.
5. Return structured `MetricDefinition` with confidence.

## SQLite Validation

Before execution:

- Parse with `sqlglot` using SQLite dialect.
- Require exactly one statement.
- Allow only `SELECT` or `WITH ... SELECT`.
- Reject mutation or DDL: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `CREATE`, `ALTER`.
- Reject `ATTACH`, `DETACH`, `PRAGMA`, extension loading, and chained statements.
- Check referenced tables against local metadata.
- Warn on unknown columns instead of failing initially, because aliases and expressions can confuse static checks.
- Execute with timeout.

## Execution Loop

Per task:

1. Load task.
2. Retrieve schema and task-aware docs.
3. Generate `Intent`.
4. Generate 3 initial `SQLCandidate` objects.
5. Validate each candidate.
6. Execute valid candidates against an in-memory SQLite backup.
7. Score candidates by validation, execution success, result shape, and critic confidence.
8. Repair the best candidate on validation or execution error.
9. Run at most one semantic repair when critic confidence is low.
10. Write SQL, CSV, and trace on success.
11. Write failure trace and skip CSV on failure.

Defaults:

```text
initial_candidates = 3
max_attempts = 4
semantic_repairs = 1
concurrency = 2
```

## Result Critique

Deterministic checks:

- SQL executed successfully.
- Row count is reasonable.
- Columns are non-empty.
- Output columns roughly match requested entities or metrics.
- Aggregation questions return aggregate-looking columns.
- Superlative or rank questions use `ORDER BY`, `LIMIT`, or ranking logic.
- Date-range questions include date filters.

LLM critic:

- Input: question, schema context, SQL, result profile.
- Output: `ConfidenceReport`.
- No gold data.

## Resume Behavior

- If final trace has `status=success` and CSV exists, skip task.
- `--force` reruns successful tasks.
- Failed traces rerun by default.
- `--skip-failed` skips failed traces.
- Run manifest records config, model, provider routing, prompt hashes, task IDs, and git commit when available.

## Evaluation

`sol01 eval` calls the official evaluator:

```bash
cd spider2-lite/evaluation_suite
/abs/path/to/methods/sol01/.venv/bin/python evaluate.py --result_dir /abs/path/to/methods/sol01/outputs/<run_id>/csv --mode exec_result
```

The wrapper should call the evaluator with the active `sol01` Python interpreter, not bare `python`, so evaluator dependencies come from the `methods/sol01` environment.

Wrapper output:

```text
outputs/<run_id>/eval/
  official_stdout.txt
  summary.json
```

Summary should include:

- correct local tasks
- attempted local tasks
- local subset score
- full benchmark equivalent score
- missing CSV count
- failed instance IDs

## Analysis

`sol01 analyze` reads traces and eval output. It should summarize:

- syntax errors
- missing table or column errors
- empty results
- suspicious row counts
- likely wrong aggregation
- likely wrong date filters
- retrieval misses
- per-database success rate

It must not feed gold data back into generation automatically.

## File Plan

Create:

```text
methods/sol01/
  PLAN.md
  README.md
  pyproject.toml
  uv.lock
  run.py
  prompts/
    intent.md
    schema_selection.md
    sql_generation.md
    sql_repair.md
    result_critic.md
  sol01/
    __init__.py
    cli.py
    config.py
    models.py
    coordinator.py
    llm.py
    tasks.py
    index.py
    retrieval.py
    docs.py
    validation.py
    sqlite_runner.py
    profiling.py
    output.py
    eval_runner.py
    analysis.py
  tests/
    test_config.py
    test_models.py
    test_tasks.py
    test_index.py
    test_docs.py
    test_validation.py
    test_sqlite_runner.py
    test_output.py
    test_llm.py
    test_coordinator.py
    test_eval_runner.py
    test_analysis.py
    test_cli.py
```

## Tasks

Each task should end with a targeted verification command before moving to the next task. Prefer narrow tests first, then run the full suite at the end. For LLM-dependent code, use fake clients in tests and reserve live OpenRouter calls for smoke runs.

### Task 1: Package Skeleton

**Files:**
- Create: `methods/sol01/pyproject.toml`
- Create: `methods/sol01/run.py`
- Create: `methods/sol01/README.md`
- Create: `methods/sol01/sol01/__init__.py`

- [ ] Create package metadata with Python `>=3.11`.
- [ ] Add dependencies: `pydantic`, `pydantic-ai`, `pandas`, `sqlglot`, `typer`, `rich`, `pytest`.
- [ ] Add CLI entrypoint `sol01 = "sol01.cli:app"`.
- [ ] Run `uv lock`.
- [ ] Run `uv run python -c "import sol01"`.

### Task 2: Config and Models

**Files:**
- Create: `methods/sol01/sol01/config.py`
- Create: `methods/sol01/sol01/models.py`
- Test: `methods/sol01/tests/test_config.py`
- Test: `methods/sol01/tests/test_models.py`

- [ ] Implement environment loading for OpenRouter and generic `LLM_*` aliases.
- [ ] Enforce provider-only `deepseek` and fallback disabled by default.
- [ ] Implement Pydantic models listed in this plan.
- [ ] Test default config and env override behavior.
- [ ] Test model construction and validation for representative typed objects.
- [ ] Run `uv run pytest tests/test_config.py tests/test_models.py -q`.

### Task 3: Task Loading and Filtering

**Files:**
- Create: `methods/sol01/sol01/tasks.py`
- Test: `methods/sol01/tests/test_tasks.py`

- [ ] Load `../../spider2-lite/spider2-lite.jsonl`.
- [ ] Filter local tasks by `instance_id.startswith("local")`.
- [ ] Support filters: `--instance-id`, `--db`, `--question-contains`, `--limit`.
- [ ] Test that local count is 135.
- [ ] Test `local003` can be selected by instance ID.
- [ ] Run `uv run pytest tests/test_tasks.py -q`.

### Task 4: Metadata Index

**Files:**
- Create: `methods/sol01/sol01/index.py`
- Test: `methods/sol01/tests/test_index.py`

- [ ] Parse SQLite `DDL.csv`.
- [ ] Parse table JSON metadata.
- [ ] Build `TableSchema` objects.
- [ ] Cache index to `methods/sol01/.cache/index.json`.
- [ ] Test index creation for `E_commerce`.
- [ ] Run `uv run pytest tests/test_index.py -q`.

### Task 5: Document Retrieval

**Files:**
- Create: `methods/sol01/sol01/docs.py`
- Test: `methods/sol01/tests/test_docs.py`

- [ ] Load markdown docs from `spider2-lite/resource/documents`.
- [ ] Chunk by headings, tables, and paragraph blocks.
- [ ] Implement `get_metric_definition(metric_name, instance_id=None, db=None)`.
- [ ] Force-load task `external_knowledge` when present.
- [ ] Add tests for `retention rate`, `RFM`, and `tip_rate`.
- [ ] Do not test `ST_DWITHIN` unless syntax-document retrieval is explicitly added; it is not part of the allowed SQLite-local document corpus.
- [ ] Run `uv run pytest tests/test_docs.py -q`.

### Task 6: Schema Retrieval

**Files:**
- Create: `methods/sol01/sol01/retrieval.py`
- Test: `methods/sol01/tests/test_index.py`

- [ ] Implement lexical ranking over table names, columns, descriptions, and sample values.
- [ ] Implement cap-limited join-neighbor expansion.
- [ ] Return `SchemaSelection`.
- [ ] Test selected tables stay within the task DB.
- [ ] Run `uv run pytest tests/test_index.py -q`.

### Task 7: SQLite Validation

**Files:**
- Create: `methods/sol01/sol01/validation.py`
- Test: `methods/sol01/tests/test_validation.py`

- [ ] Validate single statement only.
- [ ] Allow `SELECT` and `WITH ... SELECT`.
- [ ] Reject mutation, DDL, attach, detach, pragma, extension loading, and chained statements.
- [ ] Check referenced table names.
- [ ] Test valid CTE query passes.
- [ ] Test `DROP TABLE`, `PRAGMA`, and `SELECT 1; SELECT 2` fail.
- [ ] Run `uv run pytest tests/test_validation.py -q`.

### Task 8: SQLite Runner and Profiling

**Files:**
- Create: `methods/sol01/sol01/sqlite_runner.py`
- Create: `methods/sol01/sol01/profiling.py`
- Test: `methods/sol01/tests/test_sqlite_runner.py`

- [ ] Execute against an in-memory backup of the target SQLite file.
- [ ] Write CSV with `index=False`.
- [ ] Return `ExecutionResult`.
- [ ] Implement cheap automatic profiles: row count, columns, up to 3 sample rows.
- [ ] Implement bounded deeper profiles: null counts, distinct counts, min/max, top values.
- [ ] Test execution writes a CSV for a simple `SELECT`.
- [ ] Run `uv run pytest tests/test_sqlite_runner.py -q`.

### Task 9: Prompt Files and LLM Client

**Files:**
- Create: `methods/sol01/prompts/intent.md`
- Create: `methods/sol01/prompts/schema_selection.md`
- Create: `methods/sol01/prompts/sql_generation.md`
- Create: `methods/sol01/prompts/sql_repair.md`
- Create: `methods/sol01/prompts/result_critic.md`
- Create: `methods/sol01/sol01/llm.py`
- Test: `methods/sol01/tests/test_llm.py`

- [ ] Store prompts as markdown files.
- [ ] Compute SHA-256 prompt hashes.
- [ ] Configure Pydantic AI with OpenRouter-compatible model config.
- [ ] Include OpenRouter provider routing in each request.
- [ ] Return structured Pydantic outputs.
- [ ] Test prompt hash calculation from fixture prompt text.
- [ ] Test OpenRouter provider routing payload uses `only=["deepseek"]` and `allow_fallbacks=False`.
- [ ] Test structured output parsing with a fake LLM client; do not call the live API in unit tests.
- [ ] Run `uv run pytest tests/test_llm.py -q`.

### Task 10: Output and Resume

**Files:**
- Create: `methods/sol01/sol01/output.py`
- Test: `methods/sol01/tests/test_output.py`

- [ ] Create output directories for each run.
- [ ] Write `manifest.json`.
- [ ] Write SQL, CSV, and trace files.
- [ ] Implement resume behavior.
- [ ] Test successful task is skipped when CSV and success trace exist.
- [ ] Test failed traces rerun by default and `--skip-failed` skips them.
- [ ] Run `uv run pytest tests/test_output.py -q`.

### Task 11: Coordinator

**Files:**
- Create: `methods/sol01/sol01/coordinator.py`
- Test: `methods/sol01/tests/test_coordinator.py`

- [ ] Implement per-task execution loop.
- [ ] Generate 3 initial SQL candidates.
- [ ] Validate and execute candidates.
- [ ] Repair best candidate on validation or execution error.
- [ ] Run one critic-triggered semantic repair max.
- [ ] Skip CSV on failure by default.
- [ ] Record assumptions and all attempts in trace.
- [ ] Test a success path with fake LLM outputs and a temp SQLite DB.
- [ ] Test validation repair path with one invalid candidate followed by a repaired valid candidate.
- [ ] Test failure writes a failed trace and no CSV.
- [ ] Run `uv run pytest tests/test_coordinator.py -q`.

### Task 12: Official Eval Wrapper

**Files:**
- Create: `methods/sol01/sol01/eval_runner.py`
- Test: `methods/sol01/tests/test_eval_runner.py`

- [ ] Call `spider2-lite/evaluation_suite/evaluate.py` in `exec_result` mode.
- [ ] Use the current `sol01` Python interpreter path when spawning the evaluator.
- [ ] Save official stdout.
- [ ] Parse correct count and total count when possible.
- [ ] Write wrapper `summary.json`.
- [ ] Test subprocess command construction with a monkeypatched runner.
- [ ] Test parsing sample official stdout into local subset and full benchmark summary fields.
- [ ] Run `uv run pytest tests/test_eval_runner.py -q`.

### Task 13: Analysis Command

**Files:**
- Create: `methods/sol01/sol01/analysis.py`
- Test: `methods/sol01/tests/test_analysis.py`

- [ ] Read traces.
- [ ] Read eval summary.
- [ ] Group failures by validation, execution, empty result, critic, and missing CSV.
- [ ] Write `failures.json`.
- [ ] Write concise `summary.md`.
- [ ] Test analysis output from synthetic success, validation failure, execution failure, empty result, and missing CSV traces.
- [ ] Run `uv run pytest tests/test_analysis.py -q`.

### Task 14: CLI

**Files:**
- Create: `methods/sol01/sol01/cli.py`
- Modify: `methods/sol01/run.py`
- Test: `methods/sol01/tests/test_cli.py`

- [ ] Implement `index`.
- [ ] Implement `run`.
- [ ] Implement `eval`.
- [ ] Implement `analyze`.
- [ ] Implement `ask`.
- [ ] Support debug filters on `run` and `eval`.
- [ ] Support `--run-id` for `run`.
- [ ] Default `run` to local-only.
- [ ] Test CLI argument parsing and command dispatch with `typer.testing.CliRunner` and mocked handlers.
- [ ] Test `run --instance-id local003 --run-id smoke-local003` dispatches expected filters.
- [ ] Run `uv run pytest tests/test_cli.py -q`.

### Task 15: Smoke Run

**Files:**
- No new files expected.

- [ ] Run `uv run sol01 index`.
- [ ] Run `uv run sol01 run --instance-id local003 --run-id smoke-local003`.
- [ ] Run `uv run sol01 run --instance-id local004 --run-id smoke-local004`.
- [ ] Run `uv run sol01 run --instance-id local007 --run-id smoke-local007`.
- [ ] Run `uv run sol01 eval --run-id smoke-local003`.
- [ ] Inspect traces for prompt, schema retrieval, SQL, result profile, and provider routing.

## Verification

Before calling v1 complete:

```bash
cd methods/sol01
uv run pytest
uv run sol01 index
uv run sol01 run --instance-id local003 --run-id smoke-local003
uv run sol01 eval --run-id smoke-local003
```

Expected:

- Tests pass.
- Index builds.
- `outputs/smoke-local003/sql/local003.sql` exists.
- `outputs/smoke-local003/csv/local003.csv` exists on success.
- `outputs/smoke-local003/traces/local003.json` records prompt hashes and all attempts.
- Eval wrapper writes `outputs/smoke-local003/eval/official_stdout.txt`.

## Reporting Rules

When reporting results:

- Say this is SQLite/local subset only.
- Report `correct / 135` for local subset.
- Report `correct / 547` as full benchmark equivalent.
- Do not call it a full Spider2-Lite leaderboard score until BigQuery and Snowflake are supported.

## Post-v1 Experiments

These are follow-up tasks. Do not start them until the main v1 path above is complete and smoke runs are working.

### Task 16: LLM-Only Table Selection Experiment

**Depends on:** Task 15

**Files:**
- Modify: `methods/sol01/sol01/retrieval.py`
- Modify: `methods/sol01/sol01/llm.py`
- Modify: `methods/sol01/sol01/coordinator.py`
- Create: `methods/sol01/tests/fixtures/retrieval_cases.json`
- Test: `methods/sol01/tests/test_retrieval.py`

- [ ] Add an alternate retrieval mode that skips lexical ranking and asks the LLM to choose relevant tables directly from one DB schema summary.
- [ ] Keep the current lexical-first retrieval as the default path so we can compare both approaches.
- [ ] Record which retrieval mode was used in traces.
- [ ] Add a named fixture set in `tests/fixtures/retrieval_cases.json` with at least 5 vague local questions where lexical matching is weak.
- [ ] Write a comparison artifact at `outputs/<run_id>/analysis/retrieval_compare.json` with per-case selected tables, prompt size, and final SQL outcome for both modes.
- [ ] Compare lexical-first versus LLM-only selection on the fixture set and summarize misses and tradeoffs in `outputs/<run_id>/analysis/retrieval_compare.md`.
- [ ] Run `uv run pytest tests/test_retrieval.py -q`.
