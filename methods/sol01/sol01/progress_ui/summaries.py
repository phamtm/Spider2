from __future__ import annotations

from typing import Any

import pandas as pd

from sol01.progress_ui.utils import normalize_tag_values


def _status_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = {"correct": 0, "incorrect": 0, "answered": 0, "unanswered": 0}
    if "status" not in frame.columns:
        return counts

    value_counts = frame["status"].astype(str).str.lower().value_counts(dropna=False)
    for status in counts:
        counts[status] = int(value_counts.get(status, 0))
    return counts


def _summary_rates(
    total: int, answered: int, correct: int, incorrect: int, unanswered: int
) -> dict[str, float]:
    coverage_pct = answered / total * 100 if total else 0.0
    accuracy_pct = correct / answered * 100 if answered else 0.0
    return {
        "coverage_pct": coverage_pct,
        "accuracy_pct": accuracy_pct,
        "answered_pct": answered / total * 100 if total else 0.0,
        "correct_pct": correct / total * 100 if total else 0.0,
        "incorrect_pct": incorrect / total * 100 if total else 0.0,
        "unanswered_pct": unanswered / total * 100 if total else 0.0,
    }


def _summary_columns(key_name: str, value_name: str) -> list[str]:
    return [
        key_name,
        value_name,
        "total",
        "answered",
        "correct",
        "incorrect",
        "unanswered",
        "coverage_pct",
        "accuracy_pct",
        "answered_pct",
        "correct_pct",
        "incorrect_pct",
        "unanswered_pct",
    ]


def _build_summary_row(label: Any, value_name: str, group: pd.DataFrame) -> dict[str, Any]:
    counts = _status_counts(group)
    total = int(len(group))
    answered = total - counts["unanswered"]
    rates = _summary_rates(
        total,
        answered,
        counts["correct"],
        counts["incorrect"],
        counts["unanswered"],
    )
    return {
        value_name: str(label),
        "total": total,
        "answered": answered,
        "correct": counts["correct"],
        "incorrect": counts["incorrect"],
        "unanswered": counts["unanswered"],
        **rates,
    }


def compute_overall_summary(frame: pd.DataFrame) -> dict[str, Any]:
    total = int(len(frame))
    counts = _status_counts(frame)
    answered = total - counts["unanswered"]
    summary = {
        "total": total,
        "answered": answered,
        "correct": counts["correct"],
        "incorrect": counts["incorrect"],
        "unanswered": counts["unanswered"],
    }
    return summary | _summary_rates(
        total,
        answered,
        counts["correct"],
        counts["incorrect"],
        counts["unanswered"],
    )


def compute_tier_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_summary_columns("primary_tier", "tier_label"))

    rows: list[dict[str, Any]] = []
    tier_series = (
        frame["primary_tier"]
        if "primary_tier" in frame.columns
        else pd.Series([pd.NA] * len(frame), index=frame.index)
    )
    for tier_value, group in frame.assign(_tier=tier_series).groupby("_tier", dropna=False):
        if pd.isna(tier_value):
            rows.append(
                {"primary_tier": pd.NA, **_build_summary_row("Uncategorized", "tier_label", group)}
            )
            continue

        try:
            tier_number = int(tier_value)
        except (TypeError, ValueError):
            tier_number = tier_value
        rows.append(
            {
                "primary_tier": tier_number,
                **_build_summary_row(f"Tier {tier_number}", "tier_label", group),
            }
        )

    result = pd.DataFrame(rows, columns=_summary_columns("primary_tier", "tier_label"))
    if result.empty:
        return result

    sort_key = result["primary_tier"].apply(
        lambda value: (
            (1, 0, "")
            if pd.isna(value)
            else (0, 0, int(value))
            if str(value).isdigit()
            else (0, 1, str(value))
        )
    )
    result = result.assign(_tier_sort=sort_key).sort_values(
        by=["_tier_sort", "tier_label"],
        kind="stable",
    )
    return result.drop(columns="_tier_sort").reset_index(drop=True)


def compute_tag_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_summary_columns("tag", "tag_label"))

    tag_frame = frame.copy()
    if "tags" not in tag_frame.columns:
        tag_frame["tags"] = [[] for _ in range(len(tag_frame))]

    tag_frame["tag"] = tag_frame["tags"].apply(normalize_tag_values)
    tag_frame["tag"] = tag_frame["tag"].apply(lambda tags: tags or ["(no tags)"])
    tag_frame = tag_frame.explode("tag", ignore_index=True)

    rows = []
    for tag_value, group in tag_frame.groupby("tag", dropna=False):
        rows.append({"tag": str(tag_value), **_build_summary_row(tag_value, "tag_label", group)})

    result = pd.DataFrame(rows, columns=_summary_columns("tag", "tag_label"))
    if result.empty:
        return result

    return result.sort_values(
        by=["unanswered", "incorrect", "total", "tag_label"],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def recommend_focus(frame: pd.DataFrame) -> dict[str, Any]:
    summary = compute_overall_summary(frame)
    if summary["total"] == 0:
        return {
            "kind": "empty",
            "title": "No questions loaded",
            "detail": "Load results to get a focus recommendation.",
            "count": 0,
            "primary_tier": None,
            "tag": None,
            "coverage_pct": 0.0,
            "accuracy_pct": 0.0,
        }

    if summary["answered"] < 10:
        tier_summary = compute_tier_summary(frame)
        low_tier_unanswered = tier_summary[
            tier_summary["primary_tier"].isin([1, 2, 3]) & (tier_summary["unanswered"] > 0)
        ]
        if not low_tier_unanswered.empty:
            row = low_tier_unanswered.sort_values(
                by=["primary_tier", "unanswered", "total"],
                ascending=[True, False, False],
                kind="stable",
            ).iloc[0]
            tier_value = None if pd.isna(row["primary_tier"]) else int(row["primary_tier"])
            return {
                "kind": "unanswered",
                "title": f"Clear tier {row['tier_label']}",
                "detail": (
                    f"{int(row['unanswered'])} questions in {row['tier_label']} "
                    "are still unanswered."
                ),
                "count": int(row["unanswered"]),
                "primary_tier": tier_value,
                "tag": None,
                "coverage_pct": float(row["coverage_pct"]),
                "accuracy_pct": float(row["accuracy_pct"]),
            }

        return {
            "kind": "baseline",
            "title": "Build a baseline",
            "detail": (
                f"Only {summary['answered']} of {summary['total']} questions are answered. "
                "Get more coverage before tuning accuracy."
            ),
            "count": summary["answered"],
            "primary_tier": None,
            "tag": None,
            "coverage_pct": summary["coverage_pct"],
            "accuracy_pct": summary["accuracy_pct"],
        }

    tier_summary = compute_tier_summary(frame)
    incorrect_tiers = tier_summary[tier_summary["incorrect"] > 0]
    if not incorrect_tiers.empty:
        row = incorrect_tiers.sort_values(
            by=["incorrect", "unanswered", "primary_tier"],
            ascending=[False, False, True],
            kind="stable",
        ).iloc[0]
        tier_value = None if pd.isna(row["primary_tier"]) else int(row["primary_tier"])
        return {
            "kind": "incorrect",
            "title": f"Fix tier {row['tier_label']} answers",
            "detail": (
                f"{int(row['incorrect'])} incorrect answers remain in {row['tier_label']}. "
                "Start there."
            ),
            "count": int(row["incorrect"]),
            "primary_tier": tier_value,
            "tag": None,
            "coverage_pct": float(row["coverage_pct"]),
            "accuracy_pct": float(row["accuracy_pct"]),
        }

    unanswered_tiers = tier_summary[tier_summary["unanswered"] > 0]
    if not unanswered_tiers.empty:
        row = unanswered_tiers.sort_values(
            by=["primary_tier", "unanswered", "total"],
            ascending=[True, False, False],
            kind="stable",
        ).iloc[0]
        tier_value = None if pd.isna(row["primary_tier"]) else int(row["primary_tier"])
        return {
            "kind": "unanswered",
            "title": f"Clear tier {row['tier_label']}",
            "detail": (
                f"{int(row['unanswered'])} questions in {row['tier_label']} are still unanswered."
            ),
            "count": int(row["unanswered"]),
            "primary_tier": tier_value,
            "tag": None,
            "coverage_pct": float(row["coverage_pct"]),
            "accuracy_pct": float(row["accuracy_pct"]),
        }

    return {
        "kind": "complete",
        "title": "No urgent work left",
        "detail": "All questions in the current slice are answered and correct.",
        "count": 0,
        "primary_tier": None,
        "tag": None,
        "coverage_pct": summary["coverage_pct"],
        "accuracy_pct": summary["accuracy_pct"],
    }
