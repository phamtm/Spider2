"""Load allowed Spider2-snow markdown documents used as task-linked context."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache

from sol01.loading.tasks import REPO_ROOT

DOCUMENTS_ROOT = REPO_ROOT / "spider2-snow" / "resource" / "documents"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class DocumentChunk:
    """One searchable block from a markdown document."""

    source_file: str
    heading: str | None
    kind: str
    text: str


@cache
def load_document_chunks(file_name: str) -> list[DocumentChunk]:
    """Split one markdown document into heading, table, and paragraph chunks."""

    path = DOCUMENTS_ROOT / file_name
    lines = path.read_text(encoding="utf-8").splitlines()
    chunks: list[DocumentChunk] = []
    current_heading: str | None = None
    paragraph_lines: list[str] = []
    table_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        text = "\n".join(line.strip() for line in paragraph_lines if line.strip()).strip()
        if text:
            chunks.append(
                DocumentChunk(
                    source_file=file_name,
                    heading=current_heading,
                    kind="paragraph",
                    text=text,
                )
            )
        paragraph_lines = []

    def flush_table() -> None:
        nonlocal table_lines
        text = "\n".join(line.rstrip() for line in table_lines if line.strip()).strip()
        if text:
            chunks.append(
                DocumentChunk(
                    source_file=file_name,
                    heading=current_heading,
                    kind="table",
                    text=text,
                )
            )
        table_lines = []

    for raw_line in lines:
        line = raw_line.rstrip()
        heading_match = HEADING_RE.match(line.strip())

        if heading_match:
            flush_paragraph()
            flush_table()
            current_heading = heading_match.group(2).strip()
            chunks.append(
                DocumentChunk(
                    source_file=file_name,
                    heading=current_heading,
                    kind="heading",
                    text=current_heading,
                )
            )
            continue

        if _is_table_line(line):
            flush_paragraph()
            table_lines.append(line)
            continue

        if not line.strip():
            flush_paragraph()
            flush_table()
            continue

        flush_table()
        paragraph_lines.append(line)

    flush_paragraph()
    flush_table()
    return chunks


@cache
def load_document_text(file_name: str) -> str:
    """Return one allowed markdown document as plain text."""

    return (DOCUMENTS_ROOT / file_name).read_text(encoding="utf-8").strip()


def _is_table_line(line: str) -> bool:
    """Treat pipe-delimited markdown rows as table content."""

    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")
