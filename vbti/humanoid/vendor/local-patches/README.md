# Vendor submodule local patches

Uncommitted local edits to the LimX vendor submodules (which point at upstream
`limxdynamics` remotes we cannot push to), exported as patches so they survive
a fresh clone.

| Patch | Applies to |
|---|---|
| `humanoid-mujoco-sim.patch` | `vendor/humanoid-mujoco-sim` (simulator.py) |
| `humanoid-rl-deploy-python.patch` | `vendor/humanoid-rl-deploy-python` (controllers.yaml) |

Restore with:

```bash
git -C vendor/humanoid-mujoco-sim apply ../local-patches/humanoid-mujoco-sim.patch
git -C vendor/humanoid-rl-deploy-python apply ../local-patches/humanoid-rl-deploy-python.patch
```

Regenerate after changing vendor files:

```bash
git -C vendor/<submodule> diff -- . ':(exclude)*.pyc' > vendor/local-patches/<submodule>.patch
```
