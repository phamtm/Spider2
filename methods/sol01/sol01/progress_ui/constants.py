from __future__ import annotations

from sol01.infra.paths import REPO_ROOT

ROOT = REPO_ROOT
DEFAULT_DATASET = REPO_ROOT / "spider2-snow" / "spider2-snow.jsonl"
DEFAULT_SOURCE = REPO_ROOT / "methods" / "sol01" / "outputs" / "registry" / "latest.json"
OUTPUTS_ROOT = REPO_ROOT / "methods" / "sol01" / "outputs"

STATUS_ORDER = ("correct", "incorrect", "answered", "unanswered")
STATUS_LABELS = {
    "correct": "Correct",
    "incorrect": "Incorrect",
    "answered": "Answered",
    "unanswered": "Unanswered",
}
STATUS_COLORS = {
    "correct": "#22c55e",
    "incorrect": "#ef4444",
    "answered": "#64748b",
    "unanswered": "#1f2937",
}

CORRECT_COLOR = STATUS_COLORS["correct"]
INCORRECT_COLOR = STATUS_COLORS["incorrect"]
ANSWERED_COLOR = STATUS_COLORS["answered"]

CHART_HEIGHT = 440
TABLE_ROW_HEIGHT = 24
TABLE_VISIBLE_ROWS = 50
TABLE_HEIGHT = TABLE_VISIBLE_ROWS * TABLE_ROW_HEIGHT + 48
SECTION_GAP = 24
QUESTION_STATUS_ORDER = ("unanswered", "incorrect", "answered", "correct")
