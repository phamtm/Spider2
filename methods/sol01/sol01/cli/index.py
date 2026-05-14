"""CLI command for building the local schema index."""

from __future__ import annotations

from typing import Any

import typer

from sol01.infra.logging import get_logger
from sol01.schema.index import CACHE_PATH, build_index_cache

logger = get_logger(__name__)


def register(app: typer.Typer) -> None:
    app.command()(index)


def index() -> None:
    """Build the local schema index cache."""

    logger.info("index start")
    payload = handle_index()
    typer.echo(f"Indexed {len(payload)} databases into {CACHE_PATH}")
    logger.info("index complete", database_count=len(payload), cache_path=str(CACHE_PATH))


def handle_index() -> dict[str, Any]:
    """Build the schema cache and return the in-memory payload."""

    return build_index_cache()
