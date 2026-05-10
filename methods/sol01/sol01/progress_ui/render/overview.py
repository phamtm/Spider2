from __future__ import annotations

import streamlit as st

from sol01.loading.category_metadata import TIER_COMPLEXITY, tier_complexity_summary
from sol01.progress_ui.constants import STATUS_COLORS, STATUS_LABELS, STATUS_ORDER


def render_status_legend() -> None:
    items = "".join(
        f"""
        <span class="status-chip">
            <span class="status-swatch" style="background:{STATUS_COLORS[status]}"></span>
            {STATUS_LABELS[status]}
        </span>
        """
        for status in STATUS_ORDER
    )
    st.markdown(
        f"""
        <style>
        .status-legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 0 0 12px 0;
        }}
        .status-chip {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.9rem;
        }}
        .status-swatch {{
            width: 10px;
            height: 10px;
            border-radius: 999px;
            display: inline-block;
        }}
        </style>
        <div class="status-legend">{items}</div>
        """,
        unsafe_allow_html=True,
    )


def render_tier_guide(selected_tiers: list[int]) -> None:
    st.caption(tier_complexity_summary(selected_tiers))
    with st.expander("Tier guide", expanded=bool(selected_tiers)):
        for tier in sorted(TIER_COMPLEXITY):
            st.markdown(f"- **Tier {tier}**: {TIER_COMPLEXITY[tier]}")
