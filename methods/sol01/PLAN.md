# sol01 Plan

`sol01` is a Snowflake-only Spider2-snow solver.

## Current Contract

- Load tasks from `spider2-snow/spider2-snow.jsonl`.
- Load documents from `spider2-snow/resource/documents`.
- Build schema indexes from `spider2-snow/resource/databases`.
- Preserve fully qualified Snowflake table names for retrieval and validation.
- Generate one read-only Snowflake query per task.
- Execute through Snowflake using a local ignored credential file.
- Write outputs under `methods/sol01/outputs/<run_id>/{sql,csv,traces,eval,analysis}`.
- Evaluate generated CSVs through the official Spider2-snow evaluator in `exec_result` mode.

## Credential Contract

Runtime credentials live in `methods/sol01/snowflake_credential.json` and are ignored by git.

The local JSON should match:

```json
{
  "username": "<your_username>",
  "password": "<your_generated_token>",
  "account": "RSRSBDK-YDB67606",
  "role": "PARTICIPANT",
  "warehouse": "COMPUTE_WH_PARTICIPANT"
}
```

Only `username` and `password` are user-specific. `password` is the generated
programmatic access token, not the normal Snowsight password.

## Remaining Work

- Improve prompt quality and repair strategy from real Spider2-snow failures.
- Improve LLM schema-selection prompts from real Spider2-snow failures.
- Add optional gold scripts for live Snowflake connectivity when credentials are present.
