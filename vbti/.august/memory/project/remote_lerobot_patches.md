---
name: lerobot fork — our patches live in a branch, not site-packages
description: The vbti patches to LeRobot live as commits on a fork branch (29 as of 2026-05-18, incl. the SmolVLA-UVA stack). Editable installs on both machines = no more hot-patching, no more re-applying after upgrades.
type: reference
originSessionId: 5831fc1f-f3a8-4242-8572-532eea191b91
---
## State as of 2026-05-18

The "must reapply patches after every pip upgrade" pain is gone. All vbti patches are commits on a fork branch. Both machines run editable installs of that fork. A pip/uv reinstall just resyncs the editable link — the patches stay.

## The fork

- **GitHub**: `github.com/33may/lerobot`
- **Branch**: `vbti/main`
- **Base**: tag `v0.4.4` (pinned — v0.5.0 raised the Python floor to 3.12, incompatible with remote's Python 3.10)
- **Commits above v0.4.4**: 29 (10 base + 19 SmolVLA-UVA stack)
- **HEAD as of 2026-05-18**: `f8ac3149`

Remotes inside `lerobot/` clones:
- `origin` → `github.com/33may/lerobot` (our fork — push here)
- `upstream` → `github.com/huggingface/lerobot` (HF — fetch only)

## Where the fork lives

| Machine | Path | Env | Install mode |
|---|---|---|---|
| Laptop | `/home/may33/projects/ml_portfolio/robotics/lerobot/` | conda `lerobot` (Py 3.12) | `pip install -e . --no-deps` |
| Remote | `/home/vbti/anton/lerobot/` | uv `/home/vbti/anton/env` (Py 3.10) | `uv pip install -e . --no-deps` |

## The patches (oldest first)

**Base 10** (`ad80fddc` … `fc385747`):

| Commit | Subject |
|---|---|
| `ad80fddc` | vbti: bump OpenCV threads to 4 |
| `ee130763` | vbti: depth-as-packed-PNG pipeline (camera_realsense + datasets/utils + so_follower) |
| `5c9229ee` | vbti: guard GR00T config import for Python 3.12+ |
| `a4c99cbc` | vbti: register 'internvla' as alias of SmolVLAConfig |
| `3f9728fb` | vbti: dataset viz all-episodes mode |
| `5c59f3f2` | vbti: streaming_dataset skips image_transforms on depth cams |
| `f4d1b27a` | vbti: smolvla-adamw per-group-LR optimizer (optimizers.py + factory.py) |
| `2f34b9bd` | vbti: smolvla vision-only unfreeze (vision trainable, text frozen) |
| `405f96ef` | vbti: log per-param-group LRs to wandb |
| `fc385747` | vbti: zero saturation+hue weights in ImageTransformsConfig defaults |

**SmolVLA-UVA stack** — 19 commits `cdba7406` … `f8ac3149` (added 2026-05-18). New
package `src/lerobot/policies/smolvla_uva/` (config / video_head / modeling) +
`tests/policies/smolvla_uva/` (18 tests). UVA = auxiliary future-feature-prediction
loss on SmolVLA: `loss = action_loss + 0.3·video_loss`; video head dropped at
inference. Policy type registered as `smolvla_uva`. v0 validated: 18 unit tests +
end-to-end overfit smoke pass. See `docs/superpowers/specs/2026-05-13-smolvla-uva-design.md`.

Run `git log v0.4.4..vbti/main` inside the lerobot dir for the live, exhaustive list.

## Day-to-day

- **Quick sync check** (both machines should match):
  ```bash
  git log --format="%s" v0.4.4..vbti/main | sort | md5sum
  # both machines should return: d97130aa8b43faade296ed107a18b101  (29 patches, 2026-05-18)
  ```
- **Pulling new vbti commits to remote**: `cd /home/vbti/anton/lerobot && git pull origin vbti/main`. No reinstall needed (editable).
- **Pulling new vbti commits to laptop**: same. No reinstall needed.

## Adding a new patch

1. Edit files in `lerobot/` on the laptop
2. `git add` + `git commit -m "vbti: ..."` on `vbti/main`
3. `git push origin vbti/main`
4. On remote: `git pull origin vbti/main`
5. Both sides instantly see the change (editable installs)

## Upgrading the upstream base (when ready)

Rehearsed once already (v0.4.4 → v0.5.1, rolled back due to Py 3.12 floor + active sweep). Known conflicts:

- `src/lerobot/datasets/utils.py` ↔ `src/lerobot/datasets/feature_utils.py` (function moved between files in v0.5.0). When rebasing forward, port the dtype-override patch in `hw_to_dataset_features` to the new location.
- `src/lerobot/policies/__init__.py` GR00T guard collides with v0.5.0's added `multi_task_dit` import. Merge: keep the try/except wrap on groot, keep the new import.

Other 8 patches replay cleanly with small line offsets.

**Don't upgrade base in the middle of an active experiment sweep.** v0.5.x also requires Python 3.12, which forces a remote env rebuild (flash-attn cross-device dance + torch nightly for sm_120 + gr00t reinstall).

## What was lost (and recovered)

Earlier this session the editable install accidentally replaced site-packages, wiping 5 hot-patches that weren't yet in the fork. They were recovered from the **uv cache** at `/home/vbti/.cache/uv/archive-v0/MNGaMKmLE8LtKuDthlV9d/` and from `.orig` backups at `/home/vbti/anton/env/lib/python3.10/site-packages/lerobot/**/*.orig`. The `.orig` backups can be deleted now — the patches are in git.
