"""Public CLI package for sol01."""

from sol01.cli.app import app
from sol01.cli.ask import handle_ask
from sol01.cli.eval import handle_eval
from sol01.cli.index import handle_index
from sol01.cli.run import handle_run
from sol01.cli.schema_context import handle_schema_context_eval

__all__ = [
    "app",
    "handle_ask",
    "handle_eval",
    "handle_index",
    "handle_run",
    "handle_schema_context_eval",
]
