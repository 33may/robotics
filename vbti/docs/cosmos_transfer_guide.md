# Cosmos Transfer 2.5 — Data Augmentation Pipeline

Converts Isaac Sim synthetic renders into photorealistic video while preserving robot poses and object positions.

**NVIDIA benchmark:** 54% → 91% success rate (+68.5%) when mixing original + Cosmos-augmented data.

---

## Model Specs

| Property | Value |
|----------|-------|
| Model | Cosmos-Transfer2.5-2B (2.36B params) |
| HuggingFace | `nvidia/Cosmos-Transfer2.5-2B` |
| Precision | BF16 only (FP16/FP32 not supported) |
| VRAM | 65.4 GB (720p), ~24 GB (480p) |
| Speed | ~5 min per 93-frame chunk on A40 |
| Python | 3.10 (enforced) |
| Package manager | `uv` (not pip/conda) |

---

## Our Pipeline (`cosmos_transfer.py`)

### Commands

```bash
# 1. Extract frames from HDF5
python vbti/logic/reconstruct/cosmos_transfer.py extract --episode=33

# 2. Compose frames into MP4 (RGB + depth + edge)
python vbti/logic/reconstruct/cosmos_transfer.py process --episode=33

# 3. Generate Cosmos spec JSON
python vbti/logic/reconstruct/cosmos_transfer.py config --episode=33

# 4. Run Cosmos inference
python vbti/logic/reconstruct/cosmos_transfer.py transfer --episode=33

# 5. Write augmented frames back to HDF5
python vbti/logic/reconstruct/cosmos_transfer.py reassemble --episode=33

# End-to-end for all episodes:
python vbti/logic/reconstruct/cosmos_transfer.py prepare
```

### Data Flow

```
HDF5 episode → per-camera PNG frames → MP4 videos (RGB, depth, edge)
                                          ↓
                                    Cosmos spec JSON
                                          ↓
                                    Cosmos inference (GPU)
                                          ↓
                                    Augmented MP4
                                          ↓
                                    Decode → frames → HDF5
```

### Spec JSON Format

```json
{
  "prompt": "photorealistic office desk with SO101 arm...",
  "input_video_path": "rgb.mp4",
  "vis": { "control_weight": 0.7 },
  "edge": { "input_control": "edge.mp4", "control_weight": 0.3 },
  "depth": { "input_control": "depth.mp4", "control_weight": 0.5 },
  "resolution": "480",
  "num_video_frames_per_chunk": 93,
  "num_steps": 35,
  "seed": 2025
}
```

**Control weights:** 0.7+ = strict geometry adherence; 0.2–0.3 = more photorealistic freedom.

---

## Control Types

| Control | Weight | Purpose |
|---------|--------|---------|
| Depth | 0.5 | Preserves 3D structure (recommended for robotics) |
| Edge | 0.3–1.0 | Preserves object boundaries |
| Segmentation | — | **Renders black** — known limitation, do not use |
| Vis (visibility) | 0.7 | Auto-computed from input video |

**Best combo for robotics:** depth (0.5) + edge (1.0) — preserves structure while transforming appearance.

---

## RunPod Deployment

| Setting | Value |
|---------|-------|
| GPU | A40 (48 GB VRAM for 480p) |
| Cost | ~$0.76/hr |
| Full dataset | ~136 hours for 109 episodes |
| Volume | 100 GB network volume |
| Access | SSH + rsync for data transfer |

### Setup on RunPod

```bash
# Clone repo
git clone git@github.com:nvidia-cosmos/cosmos-transfer2.5.git && cd cosmos-transfer2.5
git lfs pull

# Install deps (uv, not pip)
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv sync --extra=cu128

# Auth for gated model
uv tool install -U "huggingface_hub[cli]" && hf auth login
# Must accept NVIDIA Open Model License on HF page first!
```

---

## Known Issues

1. **Segmentation renders black** — Use only depth + edge controls
2. **Model license gating** — Must accept on TWO HuggingFace pages
3. **PhysX replay nondeterminism** — Some episodes drift on replay (affects depth/seg capture)
4. **Python version** — Cosmos enforces 3.10; RunPod may have 3.11 (use Docker or override `.python-version`)

---

## Input/Output Format

- **Input:** MP4 video (optimal: multiples of 93 frames at 16 FPS), 720p or 480p
- **Control types:** depth, edge, vis (auto), multi-control
- **Output:** MP4 video, same frame count as input
- **Text prompt:** <300 words, descriptive scene content
