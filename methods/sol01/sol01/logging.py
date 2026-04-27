"""Structured console logging for sol01."""

from __future__ import annotations

import logging as std_logging
import os
import sys
from typing import TextIO

import structlog

DEFAULT_LOG_LEVEL = "INFO"


def configure_logging(
    level: str | int = DEFAULT_LOG_LEVEL,
    *,
    stream: TextIO | None = None,
    use_colors: bool | None = None,
) -> None:
    """Configure colorized structured logs for the command line entrypoints."""

    numeric_level = _resolve_level(level)
    output_stream = stream or sys.stderr
    std_logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        stream=output_stream,
        force=True,
    )
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=_resolve_colors(use_colors, output_stream)),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a logger that emits structured key/value events."""

    return structlog.get_logger(name)


def _resolve_level(level: str | int) -> int:
    """Normalize string or numeric levels to a logging level constant."""

    if isinstance(level, int):
        return level

    normalized = level.strip().upper()
    if not normalized:
        return std_logging.INFO

    resolved = std_logging.getLevelName(normalized)
    if isinstance(resolved, int):
        return resolved
    raise ValueError(f"Unknown log level: {level}")


def _resolve_colors(use_colors: bool | None, stream: TextIO) -> bool:
    """Pick a color policy that respects terminals and explicit overrides."""

    if use_colors is not None:
        return use_colors
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    isatty = getattr(stream, "isatty", None)
    if callable(isatty):
        return bool(isatty())
    return sys.stderr.isatty()
