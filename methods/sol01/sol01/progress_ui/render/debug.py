from __future__ import annotations

from typing import Any

import streamlit as st

from sol01.progress_ui.display import build_run_command
from sol01.progress_ui.transforms import prepare_debug_frame


def render_debug_tab(
    dataset_path: Any,
    source_path: Any,
    records: list[Any],
    full_frame: Any,
    available_tiers: list[int],
    available_tags: list[str],
    metadata_rows: dict[str, dict[str, Any]],
) -> None:
    from sol01.progress_ui.constants import TABLE_HEIGHT

    st.header("Debug")
    st.caption("Operational details for triage and empty-state debugging.")

    run_command = build_run_command(dataset_path, source_path)
    debug_cols = st.columns(4)
    debug_cols[0].metric("Loaded records", f"{len(records):,}")
    debug_cols[1].metric("Raw rows", f"{len(full_frame):,}")
    debug_cols[2].metric("Metadata tiers", f"{len(available_tiers):,}")
    debug_cols[3].metric("Metadata tags", f"{len(available_tags):,}")

    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.markdown("**Dataset path**")
        st.code(str(dataset_path))
        st.markdown("**Results source**")
        st.code(str(source_path))
    with detail_cols[1]:
        st.markdown("**Run command**")
        st.code(run_command, language="bash")
        st.markdown("**Metadata state**")
        if metadata_rows:
            st.success(f"Category metadata available for {len(metadata_rows):,} questions.")
        else:
            st.warning("Category metadata is missing for this dataset.")

    st.subheader("Raw rows")
    debug_frame = prepare_debug_frame(full_frame)
    st.dataframe(debug_frame, width="stretch", height=TABLE_HEIGHT, hide_index=True)
