"""Command line entrypoint for the sol01 solver."""

import typer

app = typer.Typer(
    help="SQLite-local Spider2-Lite solver.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Show root help until concrete subcommands are implemented."""
