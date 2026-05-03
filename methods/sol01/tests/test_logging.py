"""Tests for structured console logging."""

from __future__ import annotations

from io import StringIO

from sol01.infra.logging import configure_logging, get_logger


def test_configure_logging_renders_structured_fields():
    """Structured logs should include the event name and key/value context."""

    stream = StringIO()
    configure_logging("INFO", stream=stream, use_colors=False)

    logger = get_logger("sol01.test")
    logger.info("task start", run_id="run-1", task_count=2)

    output = stream.getvalue()
    assert "task start" in output
    assert "run_id" in output
    assert "task_count=2" in output
