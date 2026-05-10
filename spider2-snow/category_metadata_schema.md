# Spider2-Snow Category Metadata Schema

This file defines the hand-editable JSONL contract used to categorize Spider2-Snow questions.

Each line is one JSON object. Each object describes one `instance_id`. Batch files can be edited in parallel and merged later by `instance_id`.

## Row Shape

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `instance_id` | string | yes | Stable question id, for example `sf_bq011` or `sf_local067`. Must be unique. |
| `tier` | integer | yes | Complexity tier from 1 to 4. |
| `primary_category` | string | yes | One of the allowed primary categories listed below. |
| `tags` | array[string] | yes | Non-empty, unique list of allowed tags. Use the smallest set that fully describes the question. |
| `batch_id` | string | no | Optional batch label such as `batch-01`. Useful when slices are produced in parallel. |
| `source_line_start` | integer | no | Optional 1-based start line for the source slice. |
| `source_line_end` | integer | no | Optional 1-based end line for the source slice. Must be at least `source_line_start` when both are present. |
| `notes` | string | no | Short free-text note for edge cases or review comments. |

## Tier Definitions

- `tier = 1`: Simple lookup or single-step aggregate. Usually one table and one obvious filter.
- `tier = 2`: Straightforward multi-step query. Usually one join, modest filtering, or a simple grouped result.
- `tier = 3`: Multi-step reasoning. Common examples are ranking, window functions, temporal rollups, cohort logic, or a query that depends on external notes.
- `tier = 4`: Hard query. Usually mixes several advanced patterns, such as nested aggregation, geospatial logic, multi-hop joins, or multiple time-based transformations.

## Primary Categories

- `lookup`
- `aggregate`
- `comparison`
- `ranking`
- `trend`
- `multi_step`
- `spatial`
- `text`
- `path`
- `other`

## Allowed Tags

Use tags to describe the main techniques or reasoning patterns involved in the question.

- `aggregation`
- `comparison`
- `count`
- `distinct`
- `filter`
- `group_by`
- `join`
- `multi_join`
- `ranking`
- `top_k`
- `sort`
- `temporal`
- `date_range`
- `time_series`
- `window`
- `cumulative`
- `cohort`
- `ratio`
- `percent_change`
- `string_match`
- `text_search`
- `external_knowledge`
- `geospatial`
- `distance`
- `intersects`
- `contains`
- `subquery`
- `nested_aggregation`
- `pivot`
- `other`

## Validation Rules

- Every row must parse as a single JSON object.
- `instance_id`, `tier`, `primary_category`, and `tags` are required.
- `instance_id` must be unique across all batch files.
- `tier` must be an integer from 1 through 4.
- `primary_category` must be one of the allowed primary categories.
- `tags` must contain at least one value, must not repeat values, and every value must be in the allowed tag list.
- If both `source_line_start` and `source_line_end` are present, they must be integers and `source_line_end` must be greater than or equal to `source_line_start`.
- Use `notes` only for short human guidance. Do not store structured data there.
- If a question does not fit any listed category or tag, use `other`.

## Example

```json
{"instance_id":"sf_bq011","tier":3,"primary_category":"trend","tags":["temporal","cumulative","aggregation"],"batch_id":"batch-01","source_line_start":1,"source_line_end":69,"notes":"Requires date windowing and a cumulative rollup."}
```
