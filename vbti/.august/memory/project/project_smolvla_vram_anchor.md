---
name: project_smolvla_vram_anchor
description: SmolVLA VRAM anchor on remote 5090 (32 GB) — BS=32 + 4 cams ≈ 15.6 GB; +1 cam ≈ +2 GB; scales linear-ish in BS
type: project
originSessionId: 672e223a-9b37-4a53-84d3-29a68457f36d
---
Empirical VRAM measurement on the remote 5090 (32 GB) during v017 training, mid-step:

  | config | VRAM | source |
  |---|---|---|
  | SmolVLA, BS=32, 4 cams (top/left/right/gripper), bf16, frozen vision | **15.6 GB / 32 GB** | nvidia-smi during v017 step ~500 |

Scaling rules-of-thumb derived from this anchor:
- ~+2 GB per added camera (vision tokens scale with cam count)
- Roughly linear in batch size (activations dominate)
- bf16 already on; further reduction would need gradient checkpointing or QLoRA

**Why kept:** answers "can two SmolVLA jobs fit on one 5090?" decisions. v017+v018 parallel at BS=32 = ~33 GB → OOM; at BS=24 = ~26 GB → fits with 6 GB headroom.

**How to apply:**
- For "can I run X in parallel?" questions, use this anchor + scaling rules first; only ssh to nvidia-smi if the prediction is borderline.
- If the model architecture changes (LoRA, unfrozen ViT, larger chunk), this anchor is stale — re-measure.
- For the local 4070 Ti SUPER (16 GB), halve the batch size relative to remote.
