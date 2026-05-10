from __future__ import annotations

import html
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from sol01.progress_ui.constants import (
    ANSWERED_COLOR,
    CHART_HEIGHT,
    CORRECT_COLOR,
    STATUS_COLORS,
    STATUS_LABELS,
)
from sol01.progress_ui.utils import is_missing_value


def render_chart(progress: Any, *, empty_message: str) -> None:
    if progress.empty:
        st.info(empty_message)
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=progress["x"],
            y=progress["answered_pct"],
            name="Answered",
            mode="lines+markers",
            line={"color": ANSWERED_COLOR, "width": 3, "shape": "hv"},
            fill="tozeroy",
            fillcolor="rgba(100, 116, 139, 0.18)",
            hovertemplate="Answered: %{customdata[0]}<br>%{y:.1f}%<extra></extra>",
            customdata=progress[["answered"]],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=progress["x"],
            y=progress["correct_pct"],
            name="Correct",
            mode="lines+markers",
            line={"color": CORRECT_COLOR, "width": 3, "shape": "hv"},
            fill="tozeroy",
            fillcolor="rgba(34, 197, 94, 0.18)",
            hovertemplate="Correct: %{customdata[0]}<br>%{y:.1f}%<extra></extra>",
            customdata=progress[["correct"]],
        )
    )
    fig.update_layout(
        height=CHART_HEIGHT,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        paper_bgcolor="#080808",
        plot_bgcolor="#080808",
        font={"color": "#d7d7d7"},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.05, "x": 0},
        yaxis={
            "range": [0, 100],
            "ticksuffix": "%",
            "gridcolor": "rgba(255,255,255,0.08)",
            "zeroline": False,
        },
        xaxis={"gridcolor": "rgba(255,255,255,0.04)", "zeroline": False},
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_grid(frame: Any) -> None:
    tiles = []
    for row in frame.to_dict("records"):
        status = row["status"]
        title = f"{row['instance_id']} | {STATUS_LABELS[status]}"
        if not is_missing_value(row.get("db")):
            title += f" | {row['db']}"
        if not is_missing_value(row.get("note")):
            title += f" | {row['note']}"
        if not is_missing_value(row.get("instruction")):
            title += f" | {row['instruction']}"
        tiles.append(
            '<div class="tile" '
            f'title="{html.escape(title)}" '
            f'style="background:{STATUS_COLORS[status]}"></div>'
        )

    st.markdown(
        f"""
        <style>
        .status-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(10px, 1fr));
            gap: 3px;
            width: 100%;
            align-items: center;
        }}
        .status-grid-wrap {{
            padding-bottom: 16px;
        }}
        .tile {{
            aspect-ratio: 1 / 1;
            border: 1px solid rgba(0, 0, 0, 0.65);
            min-width: 10px;
        }}
        .tile:hover {{
            transform: scale(1.8);
            outline: 1px solid rgba(255, 255, 255, 0.75);
            z-index: 2;
        }}
        </style>
        <div class="status-grid-wrap">
            <div class="status-grid">{"".join(tiles)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
