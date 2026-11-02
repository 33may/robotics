# Known issues

Found while writing the tutorials (2026-11). Not fixed.

**Training**
- `logic/train/remote.py` `status` / `logs` read `<remote_run_dir>/train.log`; the launcher writes `<remote_version_dir>/train_<run>.log`. Monitor with `ssh <host> tail -f <remote_version_dir>/train_<run>.log` or W&B.
- `logic/train/remote.py` checks `Path(cfg.model.pretrained).exists()` locally; a pretrained path that only exists on the GPU box raises `PermissionError` (`chain_remote.py` handles this with `os.path.isdir`).
- `logic/train/remote.py`, `chain_remote.py`: remote env and CUDA 12.8 library path are hardcoded; the `python:` key in `remote.yaml` is unused.
- `logic/train/chain.py`: experiment hardcoded to `duck_cup_smolvla`; versions launch without `--experiment`, so the right experiment must be active.
- `logic/train/chain_remote.py`: the printed pull hint omits `--experiment`, which `remote pull` rejects.
- `experiments/state.json` is tracked and points at `enea/v004`, which is not in the repo. Run `experiment_utils use <exp> <vNNN>` first; `use` rewrites this tracked file.

**Evaluation / inference**
- `logic/inference/eval_engine.py`: a short `--checkpoint` (`30k`, `last`) resolves against the active version, not `--version`, while the session is saved under `--version`. Run `use` first.
- `logic/inference/eval_engine.py`: version-based checkpoint lookup takes the first `*/checkpoints` dir alphabetically (`lerobot_output_r1` before `r12`).
- `logic/inference/eval_render.py`: `heatmap` / `grid` only work on `dual_cup_*` sessions; crash on `id_scale_60` and `checkpoint_sweep`.
- `logic/inference/protocols/protocols.py`: `render` / `verify` raise `KeyError: 'cup_positions'` on the newer protocol format; use `render_protocol`.
- `logic/inference/run_real_inference.py`: `run()` connects to the robot twice without disconnecting; the first connection ignores `--max_relative_target` (move to rest always uses 10.0). An unknown `--cameras` preset silently falls back to `realsense`.

**Repo**
