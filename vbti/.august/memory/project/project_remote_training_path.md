---
name: project_remote_training_path
description: vbti.logic.train.remote ships lerobot-train CLI to remote — bypasses our SmolVLABackend; backend code only runs via local engine.train()
type: project
originSessionId: 672e223a-9b37-4a53-84d3-29a68457f36d
---
The remote training entrypoint `python -m vbti.logic.train.remote train` builds a `lerobot-train` CLI command (via `engine._build_lerobot_command`) and ships it to the remote 5090 inside a tmux session. This is **vanilla LeRobot** — it does NOT call `SmolVLABackend.load_model` / `make_dataloaders` / `train_step` etc. The custom backend code only runs when training is launched via our local `engine.train()` (which is rarely used in practice).

**Why:** the remote machine has zero `vbti` dependency — only lerobot's editable install. Keeps the remote setup minimal and survives vbti package changes without re-syncing code.

**How to apply:**
- Any feature that requires custom dataloader behavior (e.g. v018's depth decoder wrap) cannot be added to `SmolVLABackend` — that path doesn't run on remote. Either:
  1. Bake the transform into the dataset itself (the v016/v018 path: pre-render turbo PNGs into `_depth_turbo`)
  2. Patch `lerobot.scripts.lerobot_train` / `lerobot.datasets.factory.make_dataset` directly (touches lerobot's editable install)
  3. Write a thin python entrypoint that monkey-patches `make_dataset` and shells out to `lerobot.scripts.lerobot_train.train()` — then update `remote.py` to ship that entrypoint instead of `lerobot-train`
- Don't waste time editing `vbti/logic/train/backends/smolvla.py` expecting it to affect remote training. It's only exercised by `engine.train()` — which itself is mostly dead code given the remote-first workflow.
- If you need to verify whether a code path runs: search for `_build_lerobot_command` (remote) vs `backend.make_dataloaders` (local engine).
