"""Karpathy overfit sanity check for distilled detector pipeline.

Grabs one batch of 32 frames (top cam, both trusts=1), trains 200 steps,
checks loss < 0.01 AND (duck_px < 5 OR cup_px < 5). Saves plot + log.
"""
from __future__ import annotations

import os
import sys
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---- output dir ----
OUT_DIR = Path(
    "/home/may33/Documents/Obsidian Vault/vbti/researches/engineering tricks/"
    "detection/distillation/training/01_overfit_sanity"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- tee stdout to log ----
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_fp = open(OUT_DIR / "sanity.log", "w")
sys.stdout = Tee(sys.__stdout__, log_fp)
sys.stderr = Tee(sys.__stderr__, log_fp)

print("=" * 60)
print("Distill overfit sanity — top cam, batch=32, 200 steps")
print(f"Output dir: {OUT_DIR}")
print("=" * 60)

# ---- imports from pipeline ----
from vbti.logic.detection.distill_model import DistilledDetector, count_params
from vbti.logic.detection.distill import (
    DistillDataset,
    load_all_labels,
    CACHE_ROOT,
    CACHE_ROOT_NO_OBJECTS,
    DATASET_PATH,
    DATASET_PATH_NO_OBJECTS,
    distill_loss,
    NATIVE_W,
    NATIVE_H,
    _cache_paths,
    _count_total_frames,
)
from vbti.logic.detection.process_dataset import load_dataset_meta

# ---- build dataset ----
print("\n[1] Loading labels...")
labels_df = load_all_labels(seed=42, val_frac=0.15)

print("[2] Loading dataset meta...")
_, main_eps = load_dataset_meta(DATASET_PATH)
total_main = _count_total_frames(main_eps)
_, no_obj_eps = load_dataset_meta(DATASET_PATH_NO_OBJECTS)
total_no_obj = _count_total_frames(no_obj_eps)

cam = "top"
mm_main_path, _ = _cache_paths(cam, cache_root=CACHE_ROOT)
mm_no_obj_path, _ = _cache_paths(cam, cache_root=CACHE_ROOT_NO_OBJECTS)

print("[3] Building DistillDataset (train split)...")
train_ds = DistillDataset(
    cam, labels_df, "train",
    mm_main_path, mm_no_obj_path,
    total_main, total_no_obj,
)

# ---- find 32 consecutive indices where BOTH duck_trust AND cup_trust == 1 ----
print("[4] Selecting batch of 32 (both trusts=1)...")
# coord_masks shape: (N, 2) = [duck_trust, cup_trust]
both_trusted = (train_ds.coord_masks[:, 0] > 0.5) & (train_ds.coord_masks[:, 1] > 0.5)
trusted_indices = np.where(both_trusted)[0]
print(f"    Frames with both trusts=1: {len(trusted_indices)}")

if len(trusted_indices) < 32:
    raise RuntimeError(
        f"Not enough frames with both trusts=1 (found {len(trusted_indices)}, need 32). "
        "Check the labels parquet or trust columns."
    )

# Take first 32 consecutive (by position in trusted list, not guaranteed consecutive in raw array)
batch_indices = trusted_indices[:32].tolist()
print(f"    Using dataset indices {batch_indices[0]}..{batch_indices[-1]}")

# ---- collect batch manually (no DataLoader, no shuffle) ----
print("[5] Loading batch onto GPU...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"    Device: {device}")

imgs_list, tgts_list, cms_list = [], [], []
for i in batch_indices:
    img, tgt, cm, _src = train_ds[i]
    imgs_list.append(img)
    tgts_list.append(tgt)
    cms_list.append(cm)

imgs = torch.stack(imgs_list).to(device)    # (32, 3, 224, 224)
tgts = torch.stack(tgts_list).to(device)   # (32, 6)
cms  = torch.stack(cms_list).to(device)    # (32, 2)

print(f"    imgs: {imgs.shape}  tgts: {tgts.shape}  cms: {cms.shape}")

# ---- fresh model, AdamW ----
print("[6] Building fresh MobileNetV3-Small model...")
torch.manual_seed(0)
model = DistilledDetector(backbone="mobilenet_v3_small", pretrained=True).to(device)
print(f"    Params: {count_params(model):,}")

optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# ---- 200 steps on fixed batch ----
print("\n[7] Training 200 steps on fixed batch...")
STEPS = 200
LOG_EVERY = 10

log_steps   = []
log_total   = []
log_coord   = []
log_conf    = []
log_duck_px = []
log_cup_px  = []

model.train()

def px_error(pred_coords, tgt_coords, W, H):
    """(B,2) normalized -> median hypot pixel error."""
    diff = (pred_coords - tgt_coords).abs()
    diff[:, 0] *= W
    diff[:, 1] *= H
    err = torch.hypot(diff[:, 0], diff[:, 1])
    return err.median().item()

for step in range(1, STEPS + 1):
    optim.zero_grad(set_to_none=True)
    pred = model(imgs)                      # (32, 6)
    losses = distill_loss(pred, tgts, coord_mask=cms)
    losses["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optim.step()

    if step % LOG_EVERY == 0 or step == 1:
        with torch.no_grad():
            duck_px = px_error(pred[:, 0:2], tgts[:, 0:2], NATIVE_W, NATIVE_H)
            cup_px  = px_error(pred[:, 3:5], tgts[:, 3:5], NATIVE_W, NATIVE_H)
        log_steps.append(step)
        log_total.append(losses["total"].item())
        log_coord.append(losses["coord"].item())
        log_conf.append(losses["conf"].item())
        log_duck_px.append(duck_px)
        log_cup_px.append(cup_px)
        print(
            f"  step {step:3d}/{STEPS}  "
            f"total={losses['total'].item():.5f}  "
            f"coord={losses['coord'].item():.5f}  "
            f"conf={losses['conf'].item():.5f}  "
            f"duck_px={duck_px:.2f}  cup_px={cup_px:.2f}"
        )

# ---- final eval pass ----
model.eval()
with torch.no_grad():
    pred_final = model(imgs)
    losses_final = distill_loss(pred_final, tgts, coord_mask=cms)
    final_loss   = losses_final["total"].item()
    final_duck_px = px_error(pred_final[:, 0:2], tgts[:, 0:2], NATIVE_W, NATIVE_H)
    final_cup_px  = px_error(pred_final[:, 3:5], tgts[:, 3:5], NATIVE_W, NATIVE_H)

# ---- plot ----
print("\n[8] Saving plot...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Overfit sanity — top cam, batch=32", fontsize=13)

ax = axes[0]
ax.plot(log_steps, log_total,  label="total loss", color="C0", linewidth=2)
ax.plot(log_steps, log_coord,  label="coord loss", color="C1", ls="--")
ax.plot(log_steps, log_conf,   label="conf loss",  color="C2", ls=":")
ax.axhline(0.01, color="red", ls="--", alpha=0.7, label="target=0.01")
ax.set_xlabel("step"); ax.set_ylabel("loss")
ax.set_title("Losses over 200 steps")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(log_steps, log_duck_px, label="duck median px", color="C3", linewidth=2)
ax.plot(log_steps, log_cup_px,  label="cup median px",  color="C4", linewidth=2)
ax.axhline(5.0, color="red", ls="--", alpha=0.7, label="target=5px")
ax.set_xlabel("step"); ax.set_ylabel("pixel error (median)")
ax.set_title("Pixel errors over 200 steps")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.tight_layout()
plot_path = OUT_DIR / "sanity.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"    Saved: {plot_path}")

# ---- verdict ----
loss_pass = final_loss < 0.01
px_pass   = final_duck_px < 5.0 or final_cup_px < 5.0
passed    = loss_pass and px_pass

print("\n" + "=" * 60)
print(
    f"FINAL: loss={final_loss:.5f} duck_px={final_duck_px:.2f} cup_px={final_cup_px:.2f}"
)
if passed:
    print("*** PASS ***")
else:
    reasons = []
    if not loss_pass:
        reasons.append(f"loss={final_loss:.5f} >= 0.01")
    if not px_pass:
        reasons.append(f"duck_px={final_duck_px:.2f} AND cup_px={final_cup_px:.2f} both >= 5")
    print(f"*** FAIL *** ({'; '.join(reasons)})")
print("=" * 60)

# ---- findings.md ----
findings_path = OUT_DIR / "findings.md"
status_str = "PASS" if passed else "FAIL"
with open(findings_path, "w") as f:
    f.write(f"# Overfit Sanity — Top Cam\n\n")
    f.write(f"- **Result**: {status_str}\n")
    f.write(
        f"- **Final numbers**: loss={final_loss:.5f}, duck_px={final_duck_px:.2f}px, "
        f"cup_px={final_cup_px:.2f}px (after 200 steps, batch=32, lr=1e-3)\n"
    )
    concerns = []
    if not loss_pass:
        concerns.append(f"Loss did not reach <0.01 ({final_loss:.5f}); model may need more capacity or steps to memorise 32 samples")
    if not px_pass:
        concerns.append(f"Both pixel errors above 5px (duck={final_duck_px:.1f}, cup={final_cup_px:.1f}); coord head may be underfit or trust mask filtering too aggressive")
    if passed:
        concerns.append("Pipeline sane — safe to launch overnight sweep")
    f.write(f"- **Concerns**: {'; '.join(concerns) if concerns else 'None'}\n")

print(f"\nArtifacts written to: {OUT_DIR}")

log_fp.close()
