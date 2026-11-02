"""backends/ — one module per world the launcher can boot.

The registry maps the `--sim` value → backend module. Add a world = add a module here
exposing NAME/add_args/stages (+ optional reap), then list it below.
"""

from . import isaac, mujoco, real

REGISTRY = {m.NAME: m for m in (isaac, mujoco, real)}

__all__ = ["REGISTRY", "isaac", "mujoco", "real"]
