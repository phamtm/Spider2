"""Local Streamlit dashboard for Spider progress.

Run from methods/sol01:
    uv run streamlit run sol01/progress_ui/app.py

Optional CLI defaults:
    uv run streamlit run sol01/progress_ui/app.py -- --source outputs/registry/latest.json
"""

from __future__ import annotations

import streamlit as st

from sol01.progress_ui.constants import STATUS_LABELS, STATUS_ORDER, TABLE_HEIGHT
from sol01.progress_ui.display import (
    _status_dot_label,
    _status_dot_style,
    format_question_option,
    select_question_row,
)
from sol01.progress_ui.filters import apply_frame_filters, should_show_all_questions
from sol01.progress_ui.loading import (
    load_category_metadata_rows,
    load_records,
    read_dataset,
)
from sol01.progress_ui.parsing import parse_args, resolve_path
from sol01.progress_ui.render import (
    apply_page_style,
    render_debug_tab,
    render_question_detail,
    render_status_legend,
)
from sol01.progress_ui.summaries import (
    compute_overall_summary,
    compute_tag_summary,
    compute_tier_summary,
    recommend_focus,
)
from sol01.progress_ui.transforms import (
    build_status_frame,
    dataframe_height,
    prepare_question_table,
)


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="Spider Progress", layout="wide")
    apply_page_style()

    dataset_path = resolve_path(args.dataset)
    source_path = resolve_path(args.source)
    with st.sidebar:
        st.title("Progress UI")
        metadata_rows = load_category_metadata_rows(str(dataset_path))
        available_tiers = sorted({row["primary_tier"] for row in metadata_rows.values()})
        available_tags = sorted({tag for row in metadata_rows.values() for tag in row["tags"]})
        search = st.text_input("Search", value="")
        selected = st.multiselect(
            "Status",
            options=list(STATUS_ORDER),
            default=list(STATUS_ORDER),
            format_func=lambda value: STATUS_LABELS[value],
        )
        selected_tiers = st.multiselect(
            "Tier",
            options=available_tiers,
            default=[],
            help="Leave empty to include every tier.",
        )
        selected_tags = st.multiselect(
            "Tags",
            options=available_tags,
            default=[],
            help="Multiple tags use AND semantics.",
        )

    dataset = read_dataset(str(dataset_path))
    records = load_records(str(source_path))
    full_frame = build_status_frame(dataset, records, metadata_rows)

    frame = apply_frame_filters(
        full_frame,
        search=search,
        selected_status=selected,
        selected_tiers=selected_tiers,
        selected_tags=selected_tags,
    )

    summary = compute_overall_summary(frame)
    total = summary["total"]
    answered = summary["answered"]
    correct = summary["correct"]
    incorrect = summary["incorrect"]
    answered_score = summary["coverage_pct"]
    correct_score = summary["accuracy_pct"]

    overview_tab, questions_tab, debug_tab = st.tabs(["Overview", "Questions", "Debug"])

    with overview_tab:
        st.header("Spider2-Snowflake Progress")
        st.caption(f"Dataset: `{dataset_path}`  |  Results: `{source_path}`")

        cols = st.columns(4)
        cols[0].metric("Coverage", f"{answered_score:.1f}%", f"{answered:,} answered")
        cols[1].metric("Accuracy", f"{correct_score:.1f}%", f"{correct:,} correct")
        cols[2].metric("Total", f"{total:,}")
        cols[3].metric("Answered", f"{answered:,}")

        cols = st.columns(3)
        cols[0].metric("Correct", f"{correct:,}")
        cols[1].metric("Incorrect", f"{incorrect:,}")
        cols[2].metric("Unanswered", f"{summary['unanswered']:,}")

        focus = recommend_focus(frame)
        focus_cols = st.columns([2, 1])
        with focus_cols[0]:
            st.subheader("Recommended focus")
            st.info(f"{focus['title']}\n\n{focus['detail']}")
        with focus_cols[1]:
            st.metric("Focus count", f"{focus['count']:,}")
            st.metric("Coverage", f"{focus['coverage_pct']:.1f}%")
            st.metric("Accuracy", f"{focus['accuracy_pct']:.1f}%")

        tier_summary = compute_tier_summary(frame)
        tag_summary = compute_tag_summary(frame)

        st.subheader("Tier progress")
        if tier_summary.empty:
            st.info("No tier data available for the current slice.")
        else:
            st.dataframe(
                tier_summary[
                    [
                        "tier_label",
                        "total",
                        "answered",
                        "correct",
                        "incorrect",
                        "unanswered",
                        "coverage_pct",
                        "accuracy_pct",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

        st.subheader("Tag progress")
        if tag_summary.empty:
            st.info("No tag data available for the current slice.")
        else:
            st.dataframe(
                tag_summary[
                    [
                        "tag_label",
                        "total",
                        "answered",
                        "correct",
                        "incorrect",
                        "unanswered",
                        "coverage_pct",
                        "accuracy_pct",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )

    with questions_tab:
        render_status_legend()
        question_columns = [
            "instance_id",
            "status",
            "primary_tier",
            "tags",
            "db",
            "instruction",
            "note",
            "diagnostics",
        ]

        st.subheader("Filtered questions")
        if frame.empty:
            st.info("No questions match the current filters.")
        else:
            question_frame = prepare_question_table(frame)
            question_display = question_frame[question_columns].copy()
            question_display["status"] = question_display["status"].apply(_status_dot_label)
            st.dataframe(
                question_display.style.map(_status_dot_style, subset=["status"]),
                width="stretch",
                height=dataframe_height(len(question_display)),
                row_height=24,
                hide_index=True,
            )

            st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

            selected_instance_id = st.selectbox(
                "Selected question",
                options=list(question_frame["instance_id"]),
                format_func=lambda instance_id: format_question_option(
                    question_frame.loc[question_frame["instance_id"] == instance_id].iloc[0]
                ),
                label_visibility="visible",
            )
            st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
            render_question_detail(select_question_row(frame, selected_instance_id))

        if should_show_all_questions(selected_tiers, selected_tags):
            st.subheader("All questions")
            full_question_frame = prepare_question_table(full_frame)
            if full_question_frame.empty:
                st.info("No questions are available for the current dataset.")
            else:
                st.dataframe(
                    full_question_frame[question_columns],
                    width="stretch",
                    height=TABLE_HEIGHT,
                    row_height=24,
                    hide_index=True,
                )

    with debug_tab:
        render_debug_tab(
            dataset_path=dataset_path,
            source_path=source_path,
            records=records,
            full_frame=full_frame,
            available_tiers=available_tiers,
            available_tags=available_tags,
            metadata_rows=metadata_rows,
        )


if __name__ == "__main__":
    main()
