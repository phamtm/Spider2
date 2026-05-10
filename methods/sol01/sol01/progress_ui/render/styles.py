from __future__ import annotations

import streamlit as st


def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #080808;
            color: #f3f4f6;
            overflow-x: hidden;
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stMainBlockContainer"],
        [data-testid="stVerticalBlock"] {
            overflow-x: hidden;
        }
        [data-testid="stHeader"] {
            background: rgba(8, 8, 8, 0.85);
        }
        [data-testid="stMetricValue"] {
            color: #f8fafc;
        }
        [data-testid="stSidebar"] {
            background: #101010;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(255,255,255,0.08);
        }
        .section-spacer {
            height: 24px;
        }
        .question-summary {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 8px 0 12px 0;
        }
        .question-summary-item {
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 10px 12px;
            background: rgba(255, 255, 255, 0.03);
            min-width: 0;
        }
        .question-summary-label {
            display: block;
            font-size: 0.78rem;
            color: rgba(255, 255, 255, 0.65);
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0;
        }
        .question-summary-value {
            display: block;
            font-size: 1rem;
            color: #f8fafc;
            overflow-wrap: anywhere;
        }
        .question-tags {
            margin: 0 0 12px 0;
        }
        .question-tags-label {
            font-size: 0.78rem;
            color: rgba(255, 255, 255, 0.65);
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0;
        }
        .question-tags-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .question-tag {
            display: inline-flex;
            align-items: center;
            min-height: 28px;
            padding: 0 10px;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(96, 165, 250, 0.14);
            color: #dbeafe;
            font-size: 0.85rem;
            white-space: nowrap;
        }
        .question-tag-empty {
            background: rgba(255, 255, 255, 0.04);
            color: rgba(255, 255, 255, 0.6);
        }
        .question-source {
            margin: 8px 0 0 0;
        }
        .question-source-label {
            display: block;
            font-size: 0.78rem;
            color: rgba(255, 255, 255, 0.65);
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0;
        }
        .question-source-value {
            display: block;
            font-family: monospace;
            font-size: 0.9rem;
            color: #dbeafe;
            overflow-wrap: anywhere;
            word-break: break-word;
            white-space: normal;
        }
        @media (max-width: 1100px) {
            .question-summary {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 700px) {
            .question-summary {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
