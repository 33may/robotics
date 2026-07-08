---
name: feedback-shell-p-and-humanoid-cwd
description: Anton's shell ergonomics — `p` is an alias for `python`, and he runs commands from the `humanoid/` folder; write run commands accordingly.
metadata:
  type: feedback
---

When handing Anton shell commands, write them for **his** shell, not from repo root:

- Use **`p`** — his alias for `python` (e.g. `p logic/simulation/isaacsim/run_oli_sim.py`,
  `p -m pytest ...`).
- Write paths **relative to `humanoid/`**, his usual cwd
  (`~/projects/ml_portfolio/robotics/humanoid`) — so `logic/simulation/isaacsim/...`,
  NOT `humanoid/logic/simulation/isaacsim/...`.

**Why:** stated 2026-06-25 — he has `p` shortcutted and mostly works from the humanoid
folder already, so repo-root `python humanoid/...` commands are extra friction to edit.

**How to apply:** default run/test commands to `p <humanoid-relative-path>`. `conda run`
invocations still need the full `conda run -n <env> ...` form (`p` only replaces the bare
`python`). Absolute paths are still fine when cwd is ambiguous. The `run_oli_sim.py`
launcher already resolves its own paths from `__file__`, so running it from `humanoid/`
works regardless of where the entry path points.
