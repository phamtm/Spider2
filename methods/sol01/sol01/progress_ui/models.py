from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Record:
    instance_id: str
    status: str
    score: float | None
    timestamp: datetime | None
    run_id: str | None
    db: str | None
    note: str | None
    source_path: str | None
    diagnostics: str | None = None
