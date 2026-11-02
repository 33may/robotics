"""launch/ — the Oli launcher's generic supervisor + per-world backend plugins.

`launcher.py` (one level up) is the CLI; it picks a backend by `--sim`, asks it for an
ordered list of `Stage`s, and hands them to the `Supervisor`, which boots them, tees all
logs into one live stream, and tears the stack down cleanly. The supervisor knows nothing
about isaac/mujoco/real — adding a world = adding one module under `backends/`.
"""
