"""Observability hooks for sol01."""

from __future__ import annotations

import atexit
import sys

import logfire

from sol01 import __version__

_LOGFIRE_CONFIGURED = False
_LOGFIRE_SHUTDOWN_REGISTERED = False


def configure_logfire() -> bool:
    """Enable Logfire spans for Pydantic AI calls."""

    global _LOGFIRE_CONFIGURED, _LOGFIRE_SHUTDOWN_REGISTERED

    if _LOGFIRE_CONFIGURED:
        return True

    logfire.configure(
        send_to_logfire="if-token-present",
        console=logfire.ConsoleOptions(
            output=sys.stderr,
            min_log_level="debug",
            include_timestamps=True,
            include_tags=True,
            show_project_link=False,
        ),
        service_name="sol01",
        service_version=__version__,
    )
    logfire.instrument_pydantic_ai(include_content=True, version=3)
    if not _LOGFIRE_SHUTDOWN_REGISTERED:
        atexit.register(logfire.shutdown)
        _LOGFIRE_SHUTDOWN_REGISTERED = True
    _LOGFIRE_CONFIGURED = True
    return True
