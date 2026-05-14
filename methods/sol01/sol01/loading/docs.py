"""Load allowed Spider2-snow markdown documents used as task-linked context."""

from __future__ import annotations

from functools import cache

from sol01.loading.tasks import REPO_ROOT

DOCUMENTS_ROOT = REPO_ROOT / "spider2-snow" / "resource" / "documents"


@cache
def load_document_text(file_name: str) -> str:
    """Return one allowed markdown document as plain text."""

    return (DOCUMENTS_ROOT / file_name).read_text(encoding="utf-8").strip()
