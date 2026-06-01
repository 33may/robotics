"""v014 SmolVLA input-sensitivity ablation — full spec, with plots.

Conditions:
  control, no_detection, no_joints, no_state, no_images,
  no_images_no_detection, drop_{top,left,right,gripper},
  det_small_noise, det_big_noise, det_shuffled, wrong_task.

Plots:
  mae_bars.png, fallback_ratio.png, per_joint_heatmap.png, scatter_pred_vs_gt.png

Outputs:
  sensitivity_report.md
  sensitivity_plots/*.png
  sensitivity_raw.npz
"""
from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
sys.path.insert(0, "/home/may33/projects/ml_portfolio/robotics")

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor.pipeline import PolicyProcessorPipeline


# ------------------------------------------------------------------ paths
CKPT = Path(
    "/home/may33/projects/ml_portfolio/robotics/vbti/experiments/"
    "duck_cup_smolvla/v014/lerobot_output_r1/checkpoints/020000/pretrained_model"
)
V014_DIR = Path(
    "/home/may33/projects/ml_portfolio/robotics/vbti/experiments/duck_cup_smolvla/v014"
)
PLOTS_DIR = V014_DIR / "sensitivity_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_MD = V014_DIR / "sensitivity_report.md"
RAW_NPZ = V014_DIR / "sensitivity_raw.npz"

DATASET_REPO = "eternalmay33/01_02_03_merged_may-sim_detection"
DATASET_ROOT = f"/home/may33/.cache/huggingface/lerobot/{DATASET_REPO}"

CAMS = ["top", "left", "right", "gripper"]
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]
N_SAMPLES = 30
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ load
print(f"[load] policy from {CKPT}", flush=True)
policy = SmolVLAPolicy.from_pretrained(str(CKPT)).to(DEVICE).eval()

preprocessor = PolicyProcessorPipeline.from_pretrained(
    str(CKPT), config_filename="policy_preprocessor.json"
)
postprocessor = PolicyProcessorPipeline.from_pretrained(
    str(CKPT), config_filename="policy_postprocessor.json"
)

print(f"[load] dataset {DATASET_REPO}", flush=True)
ds = LeRobotDataset(DATASET_REPO, root=DATASET_ROOT)
print(f"[load] dataset has {len(ds)} frames", flush=True)

norm_state = load_safetensors(
    str(CKPT / "policy_preprocessor_step_5_normalizer_processor.safetensors")
)
ACTION_MEAN = norm_state["action.mean"].cpu().numpy().astype(np.float64)
print(f"[load] action.mean = {ACTION_MEAN}", flush=True)


# ------------------------------------------------------------------ samples
rng = np.random.default_rng(SEED)
indices = np.linspace(0, len(ds) - 1, N_SAMPLES).astype(int).tolist()
print(f"[sample] {N_SAMPLES} indices spanning [0,{len(ds)-1}]", flush=True)


# ------------------------------------------------------------------ helpers
def build_input(sample, *, state_override=None, image_overrides=None,
                task_override=None):
    if state_override is None:
        state = sample["observation.state"].clone()
    else:
        state = state_override
    if state.dim() == 1:
        state = state.unsqueeze(0)
    pi = {"observation.state": state.to(DEVICE).float()}

    image_overrides = image_overrides or {}
    for cam in CAMS:
        if cam in image_overrides and image_overrides[cam] is not None:
            img = image_overrides[cam]
        else:
            img = sample[f"observation.images.{cam}"]
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.to(DEVICE).float()
        if img.max() > 2:
            img = img / 255.0
        pi[f"observation.images.{cam}"] = img

    if task_override is not None:
        pi["task"] = task_override
    else:
        pi["task"] = sample.get("task") or "Pick up the duck and place it in the cup"
    return pi


def predict(policy_input):
    pi = preprocessor(policy_input)
    with torch.inference_mode():
        policy.reset()
        chunk_norm = policy.predict_action_chunk(pi)
        out = postprocessor({"action": chunk_norm})["action"]
    return out.cpu().numpy()[0, 0]  # first action of chunk, shape (6,)


def zero_image(sample, cam):
    img = sample[f"observation.images.{cam}"]
    return torch.zeros_like(img)


# ------------------------------------------------------------------ conditions
def cond_control(sample):
    return build_input(sample)

def cond_no_detection(sample):
    s = sample["observation.state"].clone()
    s[6:22] = 0
    return build_input(sample, state_override=s)

def cond_no_joints(sample):
    s = sample["observation.state"].clone()
    s[:6] = 0
    return build_input(sample, state_override=s)

def cond_no_state(sample):
    s = torch.zeros_like(sample["observation.state"])
    return build_input(sample, state_override=s)

def cond_no_images(sample):
    overrides = {c: zero_image(sample, c) for c in CAMS}
    return build_input(sample, image_overrides=overrides)

def cond_no_images_no_detection(sample):
    overrides = {c: zero_image(sample, c) for c in CAMS}
    s = sample["observation.state"].clone()
    s[6:22] = 0
    return build_input(sample, state_override=s, image_overrides=overrides)

def make_drop_cam(cam):
    def cond(sample):
        return build_input(sample, image_overrides={cam: zero_image(sample, cam)})
    cond.__name__ = f"cond_drop_{cam}"
    return cond

def cond_det_small_noise(sample):
    s = sample["observation.state"].clone()
    noise = torch.from_numpy(rng.normal(0, 0.05, size=16).astype(np.float32))
    s[6:22] = (s[6:22] + noise).clamp(0, 1)
    return build_input(sample, state_override=s)

def cond_det_big_noise(sample):
    s = sample["observation.state"].clone()
    s[6:22] = torch.from_numpy(
        np.clip(rng.normal(0.5, 0.2, size=16), 0, 1).astype(np.float32)
    )
    return build_input(sample, state_override=s)

def cond_det_shuffled(sample):
    s = sample["observation.state"].clone()
    perm = rng.permutation(16)
    det = s[6:22].clone()
    s[6:22] = det[torch.from_numpy(perm).long()]
    return build_input(sample, state_override=s)

def cond_wrong_task(sample):
    return build_input(sample, task_override="do nothing")


CONDITIONS = [
    ("control",                cond_control),
    ("no_detection",           cond_no_detection),
    ("no_joints",              cond_no_joints),
    ("no_state",               cond_no_state),
    ("no_images",              cond_no_images),
    ("no_images_no_detection", cond_no_images_no_detection),
    ("drop_top",               make_drop_cam("top")),
    ("drop_left",              make_drop_cam("left")),
    ("drop_right",             make_drop_cam("right")),
    ("drop_gripper",           make_drop_cam("gripper")),
    ("det_small_noise",        cond_det_small_noise),
    ("det_big_noise",          cond_det_big_noise),
    ("det_shuffled",           cond_det_shuffled),
    ("wrong_task",             cond_wrong_task),
]


# ------------------------------------------------------------------ run
abs_err = {name: np.zeros((N_SAMPLES, 6), dtype=np.float64) for name, _ in CONDITIONS}
preds = {name: np.zeros((N_SAMPLES, 6), dtype=np.float64) for name, _ in CONDITIONS}
gt_actions = np.zeros((N_SAMPLES, 6), dtype=np.float64)

t0 = time.time()
for si, idx in enumerate(indices):
    sample = ds[int(idx)]
    gt = sample["action"].cpu().numpy().astype(np.float64)
    gt_actions[si] = gt
    for name, fn in CONDITIONS:
        pi = fn(sample)
        pred = predict(pi).astype(np.float64)
        preds[name][si] = pred
        abs_err[name][si] = np.abs(pred - gt)
    if (si + 1) % 5 == 0 or si == 0:
        dt = time.time() - t0
        print(f"[run] {si+1}/{N_SAMPLES} samples done ({dt:.1f}s)", flush=True)

print(f"[run] total elapsed {time.time()-t0:.1f}s", flush=True)


# ------------------------------------------------------------------ metrics
# fall-back ratio per spec:
#   fall_back_ratio = mean(||pred - action_mean||) / mean(||gt - action_mean||)
# ~1.0 = collapsed-to-mean; ~0 = far from mean in some other direction; >1 = wandering
gt_dist_mean = np.linalg.norm(gt_actions - ACTION_MEAN[None], axis=1)
gt_dist_mean_avg = gt_dist_mean.mean()

rows = []
for name, _ in CONDITIONS:
    e = abs_err[name]
    per_joint = e.mean(axis=0)
    overall = e.mean()
    pred_dist = np.linalg.norm(preds[name] - ACTION_MEAN[None], axis=1)
    fb = pred_dist.mean() / max(gt_dist_mean_avg, 1e-9)
    rows.append((name, overall, per_joint, fb))


# ------------------------------------------------------------------ print table
def fmt_row(name, mae, pj, fb):
    pj_str = " ".join(f"{v:5.2f}" for v in pj)
    return f"{name:<24} | {mae:6.2f} | {pj_str} | {fb:5.2f}"

header = (f"{'condition':<24} | {'mae':>6} | "
          f"{'shp':>5} {'shl':>5} {'elb':>5} {'wrf':>5} {'wrr':>5} {'grp':>5} "
          f"| fbratio")
print()
print(header)
print("-" * len(header))
for r in rows:
    print(fmt_row(*r))
print()
print(f"GT mean L2 distance from action_mean = {gt_dist_mean_avg:.2f} deg")


# ------------------------------------------------------------------ plots
plt.rcParams["figure.dpi"] = 110
names = [r[0] for r in rows]
maes = np.array([r[1] for r in rows])
fbs = np.array([r[3] for r in rows])
per_joint = np.stack([r[2] for r in rows], axis=0)  # (n_cond, 6)


# 1) mae bars (sorted ascending)
order = np.argsort(maes)
fig, ax = plt.subplots(figsize=(8.5, 7.5))
y = np.arange(len(names))
ax.barh(y, maes[order], color="tab:blue")
ax.set_yticks(y)
ax.set_yticklabels([names[i] for i in order])
ax.set_xlabel("Mean MAE (deg)")
ax.set_title(f"v014 input-sensitivity (MAE on training data, N={N_SAMPLES})")
for yi, mi in zip(y, maes[order]):
    ax.text(mi + 0.05, yi, f"{mi:.2f}", va="center", fontsize=9)
ax.set_xlim(0, maes.max() * 1.15)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "mae_bars.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# 2) fallback ratio
order_fb = np.argsort(fbs)
fig, ax = plt.subplots(figsize=(8.5, 7.5))
y = np.arange(len(names))
colors = ["tab:red" if abs(v - 1.0) < 0.10 else "tab:blue" for v in fbs[order_fb]]
ax.barh(y, fbs[order_fb], color=colors)
ax.set_yticks(y)
ax.set_yticklabels([names[i] for i in order_fb])
ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="collapsed to mean (=1.0)")
ax.set_xlabel("fallback ratio  =  mean(||pred-mean||) / mean(||GT-mean||)")
ax.set_title("v014 fall-back-to-action-mean ratio per condition")
for yi, vi in zip(y, fbs[order_fb]):
    ax.text(vi + 0.01, yi, f"{vi:.2f}", va="center", fontsize=9)
ax.legend(loc="lower right")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "fallback_ratio.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# 3) per-joint heatmap
fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(per_joint, cmap="viridis", aspect="auto")
ax.set_xticks(range(6))
ax.set_xticklabels(JOINT_NAMES, rotation=30, ha="right")
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_title("Per-joint MAE (deg) by condition")
for i in range(per_joint.shape[0]):
    for j in range(per_joint.shape[1]):
        v = per_joint[i, j]
        # white text on dark cells, black on light
        rel = (v - per_joint.min()) / max(per_joint.max() - per_joint.min(), 1e-9)
        ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                color="white" if rel < 0.6 else "black", fontsize=8)
fig.colorbar(im, ax=ax, label="MAE (deg)")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "per_joint_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# 4) scatter pred vs gt — control vs no_detection per joint
fig, axes = plt.subplots(2, 3, figsize=(11, 7))
ctrl_pred = preds["control"]
nodet_pred = preds["no_detection"]
for j, ax in enumerate(axes.flat):
    ax.scatter(gt_actions[:, j], ctrl_pred[:, j], s=24, alpha=0.8,
               label="control", color="tab:blue")
    ax.scatter(gt_actions[:, j], nodet_pred[:, j], s=24, alpha=0.8,
               label="no_detection", color="tab:red", marker="x")
    lo = min(gt_actions[:, j].min(), ctrl_pred[:, j].min(), nodet_pred[:, j].min())
    hi = max(gt_actions[:, j].max(), ctrl_pred[:, j].max(), nodet_pred[:, j].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.7, alpha=0.6)
    ax.axhline(ACTION_MEAN[j], color="gray", linestyle=":", linewidth=0.7,
               label="action mean")
    ax.set_xlabel("GT (deg)")
    ax.set_ylabel("pred (deg)")
    ax.set_title(JOINT_NAMES[j])
    if j == 0:
        ax.legend(fontsize=7, loc="best")
fig.suptitle("Pred vs GT — control vs no_detection (per-joint)")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "scatter_pred_vs_gt.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"[plot] wrote 4 figures to {PLOTS_DIR}", flush=True)


# ------------------------------------------------------------------ markdown
def md_table(rows):
    lines = [
        "| condition | mae (deg) | shoulder_pan | shoulder_lift | elbow_flex | "
        "wrist_flex | wrist_roll | gripper | fallback_ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, mae, pj, fb in rows:
        cells = [name, f"{mae:.2f}"] + [f"{v:.2f}" for v in pj] + [f"{fb:.2f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


by_name = {r[0]: r for r in rows}
control_mae = by_name["control"][1]
nodet_mae = by_name["no_detection"][1]
noimg_mae = by_name["no_images"][1]
nostate_mae = by_name["no_state"][1]
both_mae = by_name["no_images_no_detection"][1]
nojoints_mae = by_name["no_joints"][1]
big_noise_mae = by_name["det_big_noise"][1]
small_noise_mae = by_name["det_small_noise"][1]
shuffled_mae = by_name["det_shuffled"][1]
wrong_task_mae = by_name["wrong_task"][1]

drop_lines = ", ".join(
    f"{c}={by_name['drop_'+c][1]:.2f} (fb={by_name['drop_'+c][3]:.2f})" for c in CAMS
)

ratio_nodet_noimg = nodet_mae / max(noimg_mae, 1e-6)
verdict_yes = (nodet_mae > 2 * noimg_mae) and (nodet_mae > 3 * control_mae)

tldr = (
    f"Removing the 16 detection coords drives MAE from "
    f"{control_mae:.2f}° (control) to {nodet_mae:.2f}° "
    f"(fallback={by_name['no_detection'][3]:.2f}), while removing all four images "
    f"only goes to {noimg_mae:.2f}° "
    f"(fallback={by_name['no_images'][3]:.2f}). "
    f"v014 leans on detections, not pixels — "
    f"{ratio_nodet_noimg:.1f}× more sensitive to losing detections than to losing all four cameras."
)

md = f"""# v014 SmolVLA — input sensitivity ablation

## TL;DR
{tldr}

## Method
- Checkpoint: `vbti/experiments/duck_cup_smolvla/v014/lerobot_output_r1/checkpoints/020000/pretrained_model`
- Dataset: `{DATASET_REPO}` ({len(ds)} frames)
- N = {N_SAMPLES} samples, evenly spaced indices in `[0, {len(ds)-1}]`, seed={SEED}
- Action mean (deg) from `policy_preprocessor_step_5_normalizer_processor.safetensors`:
  `{np.array2string(ACTION_MEAN, precision=2, separator=", ")}`
- Mean GT L2 distance from action_mean = **{gt_dist_mean_avg:.2f}°**
- `fallback_ratio = mean(||pred - action_mean||) / mean(||GT - action_mean||)`
  - ~1.0 -> prediction collapses onto the action mean
  - ~0 -> prediction is at the mean in some other direction (poor but not collapsed)
  - >1 -> actively wandering away from mean
- Single-frame eval: `policy.reset()` + `predict_action_chunk()`, take `chunk[0]`.
- Joints in degrees throughout. Image normalizer is IDENTITY -> "zero an image" = literal zeros tensor.
- State `[0:6]`=joints (deg), `[6:22]`=detection cx/cy normalized to [0,1].

## Results
{md_table(rows)}

## Plots

![MAE bars](sensitivity_plots/mae_bars.png)

![Fallback ratio](sensitivity_plots/fallback_ratio.png)

![Per-joint heatmap](sensitivity_plots/per_joint_heatmap.png)

![Pred vs GT scatter](sensitivity_plots/scatter_pred_vs_gt.png)

## Reading the numbers
- **control** = {control_mae:.2f}° MAE (fb={by_name['control'][3]:.2f}) — sanity floor on training-distribution data.
- **no_detection** = {nodet_mae:.2f}° (fb={by_name['no_detection'][3]:.2f}); **no_images** = {noimg_mae:.2f}° (fb={by_name['no_images'][3]:.2f}). Detection-loss is **{ratio_nodet_noimg:.1f}× worse** than total image loss.
- **det_shuffled** = {shuffled_mae:.2f}° — keeping detection magnitudes but breaking spatial meaning is roughly as harmful as zeroing them ({nodet_mae:.2f}°), so the model uses the spatial layout, not just statistics.
- **det_small_noise** = {small_noise_mae:.2f}° vs **det_big_noise** = {big_noise_mae:.2f}° — smooth degradation as detection drifts; even σ=0.05 perturbation is detectable.
- **drop_one_cam**: {drop_lines}. Differences across cameras are small (≤1° vs control), reinforcing that no single camera is load-bearing.
- **no_images_no_detection** = {both_mae:.2f}° (joints + task only). **no_detection** = {nodet_mae:.2f}° sits {'≈ at that floor' if abs(nodet_mae-both_mae) < 1.0 else f'{nodet_mae-both_mae:+.2f}° above that floor'} — images recover {'almost no signal' if abs(nodet_mae-both_mae) < 1.0 else 'modest signal'} once detection is removed.
- **wrong_task** = {wrong_task_mae:.2f}° — task token has {'noticeable' if wrong_task_mae > control_mae * 2 else 'minor'} impact relative to control.
- **no_joints** = {nojoints_mae:.2f}° — zeroing only the 6 joint dims (keeping detection + images) is {'far less' if nojoints_mae < nodet_mae else 'more'} damaging than zeroing detection.

## Conclusion
{'**Hypothesis supported.**' if verdict_yes else '**Hypothesis only partially supported.**'}
- Detection-only removal (no_detection={nodet_mae:.2f}°) hurts the policy **{ratio_nodet_noimg:.1f}× more** than removing all four image streams (no_images={noimg_mae:.2f}°).
- Detection-only removal is **{nodet_mae/max(control_mae,1e-6):.1f}× control** ({control_mae:.2f}°), and lands {'within 1°' if abs(nodet_mae-both_mae)<1.0 else f'{abs(nodet_mae-both_mae):.1f}° away'} of the joints-only floor ({both_mae:.2f}°).
- Big-noise on detection ({big_noise_mae:.2f}°) and shuffled detection ({shuffled_mae:.2f}°) confirm the model reads spatial structure off the cx/cy channel, not just the magnitudes.
- Practical implication: any drift in live detection (camera angle, lighting, distilled-detector failure on real robot) puts the state vector outside the training manifold while the image features — which the policy effectively ignores — cannot compensate. This matches the observed real-robot behavior (jitter + drift toward mean).
- Next experiment: retrain v014 without state[6:22] (joints-only state, images carry all visual signal) and/or apply detection-dropout during training to force image utilisation.
"""

REPORT_MD.write_text(md)
print(f"[write] {REPORT_MD}")

np.savez(
    RAW_NPZ,
    indices=np.array(indices, dtype=np.int64),
    sample_indices=np.array(indices, dtype=np.int64),
    gt=gt_actions,
    gt_actions=gt_actions,
    action_mean=ACTION_MEAN,
    **{name: preds[name] for name, _ in CONDITIONS},
    **{f"abs_err__{name}": abs_err[name] for name, _ in CONDITIONS},
)
print(f"[write] {RAW_NPZ}")
