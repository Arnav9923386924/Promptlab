"""PromptLab CLI - Main entry point.

This file creates the Typer app and sub-apps, then delegates all command
registration to the modules in ``cli/commands/``.
"""

import typer

from promptlab import __version__
from promptlab.cli.helpers import console


# ── App / sub-app creation ──────────────────────────────────────────────

app = typer.Typer(
    name="promptlab",
    help="CI/CD for LLM Applications - Test prompts with LLM Council evaluation",
    add_completion=False,
)

bsp_app = typer.Typer(
    name="bsp",
    help="BSP validation, linting, optimization & test generation.",
    add_completion=False,
    invoke_without_command=True,
)

ft_app = typer.Typer(
    name="ft",
    help="Fine-tuning hyperparameter governance council.",
    add_completion=False,
    invoke_without_command=True,
)

app.add_typer(bsp_app, name="bsp")
app.add_typer(ft_app, name="ft")


# ── Version callback ────────────────────────────────────────────────────

def version_callback(value: bool):
    """Show version and exit."""
    if value:
        console.print(f"[bold blue]promptlab[/bold blue] v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """PromptLab - CI/CD for LLM Applications."""
    pass


# ── Register all commands from sub-modules ──────────────────────────────

from promptlab.cli.commands import register_all  # noqa: E402

register_all(app, bsp_app, ft_app)


if __name__ == "__main__":
    app()
