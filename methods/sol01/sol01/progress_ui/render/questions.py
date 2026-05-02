from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from sol01.llm_call_logs import (
    build_llm_call_detail_sections,
    build_llm_call_summary_rows,
    format_llm_call_value,
    load_llm_call_log,
)
from sol01.progress_ui.display import (
    _format_llm_call_option,
    resolve_selected_llm_call_log_path,
)
from sol01.progress_ui.utils import is_missing_value, normalize_tag_values


def render_llm_call_log_panel(row: dict[str, Any]) -> None:
    st.subheader("LLM calls")
    log_path = resolve_selected_llm_call_log_path(row)
    if log_path is None:
        st.info("No LLM call log is available for this question.")
        return

    log = load_llm_call_log(log_path)
    if log.errors:
        st.warning(f"Skipped {len(log.errors)} corrupted LLM call row(s) while loading this log.")
        with st.expander("Load warnings", expanded=False):
            for error in log.errors:
                st.code(
                    f"{error.path}:{error.line_number}\n{error.message}\n{error.raw_line or ''}",
                    language="text",
                )

    if not log.records:
        st.info(f"No usable LLM call rows were found in `{log_path}`.")
        return

    summary_rows = build_llm_call_summary_rows(log)
    st.caption(f"Log file: `{log_path}`")
    st.dataframe(
        pd.DataFrame(summary_rows)[
            [
                "sequence",
                "call_id",
                "prompt_name",
                "status",
                "duration",
                "model",
                "attempts",
                "error_state",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    selected_call_index = st.selectbox(
        "Selected call",
        options=list(range(len(summary_rows))),
        format_func=lambda index: _format_llm_call_option(summary_rows[index]),
        key=f"llm-call-{row['instance_id']}",
    )
    selected_record = log.records[selected_call_index]
    sections = build_llm_call_detail_sections(selected_record)

    st.markdown("**System prompt**")
    st.code(format_llm_call_value(sections["system_prompt"]), language="text")
    st.markdown("**User prompt**")
    st.code(format_llm_call_value(sections["user_prompt"]), language="text")
    st.markdown("**Output schema**")
    st.code(format_llm_call_value(sections["output_schema"]), language="text")
    st.markdown("**Validated response**")
    st.code(format_llm_call_value(sections["validated_output"]), language="json")
    st.markdown("**Attempts**")
    st.code(format_llm_call_value(sections["attempts"]), language="json")
    st.markdown("**Error**")
    st.code(format_llm_call_value(sections["error"]), language="json")


def render_question_detail(row: dict[str, Any] | None) -> None:
    if not row:
        st.info("Select a question to see its details.")
        return

    st.subheader("Selected question")
    question_text = (
        html.escape(str(row.get("instruction")))
        if not is_missing_value(row.get("instruction"))
        else "\u2014"
    )
    st.markdown(f"**Question**\n\n{question_text}")
    st.markdown(
        """
        <div class="question-summary">
          <div class="question-summary-item">
            <span class="question-summary-label">Status</span>
            <span class="question-summary-value">{status}</span>
          </div>
          <div class="question-summary-item">
            <span class="question-summary-label">Tier</span>
            <span class="question-summary-value">{tier}</span>
          </div>
          <div class="question-summary-item">
            <span class="question-summary-label">DB</span>
            <span class="question-summary-value">{db}</span>
          </div>
          <div class="question-summary-item">
            <span class="question-summary-label">Score</span>
            <span class="question-summary-value">{score}</span>
          </div>
        </div>
        """.format(
            status=html.escape(str(row["status_label"])),
            tier=html.escape(str(row["primary_tier_label"])),
            db=html.escape(str(row["db"] if not is_missing_value(row["db"]) else "\u2014")),
            score=html.escape("" if pd.isna(row.get("score")) else str(row["score"])),
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="question-tags">
          <div class="question-tags-label">Tags</div>
          <div class="question-tags-row">{tags}</div>
        </div>
        """.format(tags=_render_tag_chips(row.get("tags"))),
        unsafe_allow_html=True,
    )

    with st.expander("More details", expanded=False):
        _render_question_field(
            "Note",
            row["note"] if not is_missing_value(row["note"]) else "\u2014",
        )
        _render_question_field(
            "Difficulty notes",
            row["difficulty_notes"] if not is_missing_value(row["difficulty_notes"]) else "\u2014",
        )

        if not is_missing_value(row.get("diagnostics")):
            _render_question_field("Diagnostics", row["diagnostics"])

        source_path = row["source_path"] if not is_missing_value(row["source_path"]) else "\u2014"
        st.markdown(
            f"""
            <div class="question-source">
              <span class="question-source-label">Source</span>
              <span class="question-source-value">{html.escape(str(source_path))}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    render_llm_call_log_panel(row)


def _render_question_field(label: str, value: Any) -> None:
    st.markdown(f"**{label}**\n\n{value}")


def _render_tag_chips(value: Any) -> str:
    tags = normalize_tag_values(value)
    if not tags:
        return "<span class='question-tag question-tag-empty'>\u2014</span>"
    return "".join(f"<span class='question-tag'>{html.escape(tag)}</span>" for tag in tags)
