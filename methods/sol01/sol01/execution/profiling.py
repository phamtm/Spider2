"""Build bounded profiles from query results for later scoring and debugging."""

from __future__ import annotations

import pandas as pd

from sol01.execution.snowflake_runner import _clean_value, _dataframe_records, _record_keys


def profile_dataframe(
    dataframe: pd.DataFrame,
    *,
    sample_limit: int = 3,
    max_profile_rows: int = 1000,
    top_k: int = 5,
) -> dict[str, object]:
    """Return cheap and bounded summary stats for one query result DataFrame."""

    bounded = dataframe.head(max_profile_rows)
    null_counts: dict[str, int] = {}
    distinct_counts: dict[str, int] = {}
    min_values: dict[str, object] = {}
    max_values: dict[str, object] = {}
    top_values: dict[str, list[dict[str, object]]] = {}
    record_keys = _record_keys(bounded.columns)

    for column_index, _column_name in enumerate(bounded.columns):
        column_key = record_keys[column_index]
        series = bounded.iloc[:, column_index]
        null_counts[column_key] = int(series.isna().sum())
        distinct_counts[column_key] = int(series.nunique(dropna=False))

        non_null = series.dropna()
        if not non_null.empty:
            try:
                min_values[column_key] = _clean_value(non_null.min())
                max_values[column_key] = _clean_value(non_null.max())
            except TypeError:
                pass

            counts = non_null.value_counts().head(top_k)
            top_values[column_key] = [
                {"value": _clean_value(value), "count": int(count)}
                for value, count in counts.items()
            ]
        else:
            top_values[column_key] = []

    return {
        "row_count": len(dataframe),
        "columns": [str(column) for column in dataframe.columns],
        "sample_rows": _dataframe_records(dataframe.head(sample_limit)),
        "null_counts": null_counts,
        "distinct_counts": distinct_counts,
        "min_values": min_values,
        "max_values": max_values,
        "top_values": top_values,
        "profile_row_count": len(bounded),
    }
