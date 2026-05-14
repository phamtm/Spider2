"""Create the Typer app and register the command modules."""

from __future__ import annotations

from typing import Annotated

import typer

from sol01.cli.analysis import register as register_analysis
from sol01.cli.ask import register as register_ask
from sol01.cli.eval import register as register_eval
from sol01.cli.index import register as register_index
from sol01.cli.inspect import register as register_inspect
from sol01.cli.run import register as register_run
from sol01.cli.schema_context import register as register_schema_context
from sol01.infra.logging import configure_logging, get_logger
from sol01.infra.observability import configure_logfire

app = typer.Typer(help="Snowflake Spider2-snow solver.", no_args_is_help=True)
logger = get_logger(__name__)


@app.callback()
def main(
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            envvar="SOL01_LOG_LEVEL",
            help="Logging level for structured console output.",
        ),
    ] = "INFO",
) -> None:
    """Keep the root command focused on subcommands."""

    configure_logging(log_level)
    configure_logfire()
    logger.info("logging configured", log_level=log_level)


def _register_commands() -> None:
    register_index(app)
    register_run(app)
    register_eval(app)
    register_analysis(app)
    register_inspect(app)
    register_schema_context(app)
    register_ask(app)


_register_commands()
