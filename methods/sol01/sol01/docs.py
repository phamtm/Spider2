"""Load allowed markdown documents and pull task-aware metric definitions from them."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from sol01.models import MetricDefinition, Task
from sol01.tasks import REPO_ROOT, SPIDER2_LITE_PATH

DOCUMENTS_ROOT = REPO_ROOT / "spider2-lite" / "resource" / "documents"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class DocumentChunk:
    """One searchable block from a markdown document."""

    source_file: str
    heading: str | None
    kind: str
    text: str


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


def get_metric_definition(
    metric_name: str,
    *,
    instance_id: str | None = None,
    db: str | None = None,
) -> MetricDefinition:
    """Find the most relevant metric definition, preferring task-linked documents when present."""

    task_doc = _task_external_knowledge(instance_id)
    scored_chunks = sorted(
        (
            (
                _score_chunk(chunk, metric_name, task_doc=task_doc, db=db),
                chunk,
            )
            for chunk in _all_chunks()
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    best_score, best_chunk = scored_chunks[0]
    section_text = _section_text(best_chunk)

    return MetricDefinition(
        metric_name=metric_name,
        source_file=best_chunk.source_file,
        heading=best_chunk.heading,
        definition=section_text,
        formula=_extract_formula(section_text),
        sql_notes=_extract_sql_notes(section_text),
        confidence=min(1.0, best_score / 20.0),
    )


def _all_chunks() -> Iterable[DocumentChunk]:
    """Yield chunks from every allowed markdown document."""

    for path in sorted(DOCUMENTS_ROOT.glob("*.md")):
        yield from load_document_chunks(path.name)


def _task_external_knowledge(instance_id: str | None) -> str | None:
    """Return the task-linked document for an instance when one exists."""

    if not instance_id:
        return None

    with SPIDER2_LITE_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("instance_id") == instance_id:
                task = Task.model_validate(payload)
                return task.external_knowledge
    return None


def _score_chunk(
    chunk: DocumentChunk,
    metric_name: str,
    *,
    task_doc: str | None,
    db: str | None,
) -> float:
    """Score a chunk using file, heading, and text matches."""

    metric_text = _normalize(metric_name)
    metric_tokens = set(metric_text.split())
    file_text = _normalize(Path(chunk.source_file).stem)
    heading_text = _normalize(chunk.heading or "")
    chunk_text = _normalize(chunk.text)
    score = 0.0

    if task_doc and chunk.source_file == task_doc:
        score += 8.0
    if metric_text and metric_text == file_text:
        score += 12.0
    elif metric_text and metric_text in file_text:
        score += 9.0
    if metric_text and metric_text == heading_text:
        score += 10.0
    elif metric_text and metric_text in heading_text:
        score += 8.0
    if metric_text and metric_text in chunk_text:
        score += 7.0

    score += 1.5 * len(metric_tokens & set(file_text.split()))
    score += 1.0 * len(metric_tokens & set(heading_text.split()))
    score += 0.5 * len(metric_tokens & set(chunk_text.split()))

    if db:
        db_tokens = set(_normalize(db).split())
        score += 0.25 * len(db_tokens & set(chunk_text.split()))

    if chunk.kind == "heading":
        score += 0.25
    return score


def _section_text(best_chunk: DocumentChunk) -> str:
    """Return the best chunk plus nearby chunks from the same heading section."""

    chunks = load_document_chunks(best_chunk.source_file)
    if best_chunk.heading is None:
        whole_document = [chunk.text for chunk in chunks if chunk.kind != "heading"]
        return "\n\n".join(whole_document) if whole_document else best_chunk.text

    section_parts: list[str] = []
    in_section = False
    for chunk in chunks:
        if chunk.kind == "heading":
            if in_section and chunk.heading != best_chunk.heading:
                break
            in_section = chunk.heading == best_chunk.heading
            continue
        if in_section:
            section_parts.append(chunk.text)

    if section_parts:
        return "\n\n".join(section_parts)
    return best_chunk.text


def _extract_formula(section_text: str) -> str | None:
    """Pull out a formula hint when the document mentions one explicitly."""

    for paragraph in section_text.split("\n\n"):
        if "formula" in paragraph.casefold():
            return paragraph.strip()
    return None


def _extract_sql_notes(section_text: str) -> str | None:
    """Return criteria or category notes that are useful when writing SQL."""

    notes: list[str] = []
    for paragraph in section_text.split("\n\n"):
        lowered = paragraph.casefold()
        if "criteria:" in lowered or "categorized" in lowered or "segment" in lowered:
            notes.append(paragraph.strip())
    if notes:
        return "\n\n".join(notes)
    return None


def _normalize(text: str) -> str:
    """Normalize text for deterministic matching."""

    lowered = text.replace("_", " ").casefold()
    return " ".join(part for part in NON_ALNUM_RE.sub(" ", lowered).split() if part)


def _is_table_line(line: str) -> bool:
    """Treat pipe-delimited markdown rows as table content."""

    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")
