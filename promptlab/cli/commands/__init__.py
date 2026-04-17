"""CLI command modules.

Each module exposes a ``register(...)`` function that attaches its commands
to the appropriate Typer app or sub-app.
"""

from promptlab.cli.commands import init as _init_mod
from promptlab.cli.commands import bsp as _bsp_mod
from promptlab.cli.commands import general as _general_mod
from promptlab.cli.commands import ft as _ft_mod


def register_all(app, bsp_app, ft_app):
    """Wire every command module to the correct app/sub-app."""
    _init_mod.register(app)
    _bsp_mod.register(bsp_app, app)
    _general_mod.register(app)
    _ft_mod.register(ft_app)
