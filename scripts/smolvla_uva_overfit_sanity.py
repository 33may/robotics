"""End-to-end smoke test for the SmolVLA-UVA pipeline.

Phase 1 — bake video features into a new dataset copy (subprocess).
Phase 2 — build policy via make_policy + train ~N steps in-process.
Phase 3 — assert both action_loss and video_loss decreased.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from statistics import mean

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot.policies.smolvla_uva.configuration_smolvla_uva import SmolVLAUVAConfig

logging.basicConfig(
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1 — bake
# ---------------------------------------------------------------------------

def phase_bake(args: argparse.Namespace) -> None:
    log.info("[Phase 1] Baking video features → %s", args.baked_output)
    cmd = [
        sys.executable, "-m", "vbti.logic.dataset.add_video_features",
        "--dataset", args.dataset,
        "--teacher", args.teacher,
        "--layer", "siglip_output",
        "--spatial-size", "4",
        "--t-future", "4",
        "--target-camera", "observation.images.gripper",
        "--output", args.baked_output,
    ]
    if args.force:
        cmd.append("--force")
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=True)
    log.info("[Phase 1] Done (returncode=%d)", result.returncode)


# ---------------------------------------------------------------------------
# Phase 2 — build policy + dataset, train N steps
# ---------------------------------------------------------------------------

def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def phase_train(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("[Phase 2] Device: %s", device)

    # -- dataset metadata from baked copy --
    log.info("[Phase 2] Loading dataset metadata from %s", args.baked_output)
    ds_meta = LeRobotDatasetMetadata(repo_id="local/uva_smoke", root=args.baked_output)

    # -- UVA config --
    cfg = SmolVLAUVAConfig(device=str(device))

    # -- resolve delta timestamps from config + metadata --
    delta_timestamps = resolve_delta_timestamps(cfg, ds_meta)
    log.info("[Phase 2] delta_timestamps keys: %s", list(delta_timestamps.keys()) if delta_timestamps else None)

    # -- open baked dataset --
    dataset = LeRobotDataset(
        repo_id="local/uva_smoke",
        root=args.baked_output,
        delta_timestamps=delta_timestamps,
    )
    log.info("[Phase 2] Dataset length: %d rows", len(dataset))

    # -- build policy via make_policy (exercises validate_features exclusion) --
    log.info("[Phase 2] Building SmolVLAUVA policy via make_policy …")
    policy = make_policy(cfg, ds_meta=ds_meta)
    policy.train().to(device)

    # -- DataLoader + optimizer --
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)

    action_history: list[float] = []
    video_history: list[float] = []

    log.info("[Phase 2] Training for %d steps (batch_size=%d) …", args.steps, args.batch_size)

    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        batch = move_batch_to_device(batch, device)

        opt.zero_grad(set_to_none=True)
        total_loss, loss_dict = policy.forward(batch)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        action_history.append(loss_dict["action_loss"])
        video_history.append(loss_dict["video_loss"])

        if step % 10 == 0 or step == 1:
            log.info(
                "  step %3d/%d  action_loss=%.4f  video_loss=%.4f  total=%.4f",
                step, args.steps,
                loss_dict["action_loss"],
                loss_dict["video_loss"],
                loss_dict["loss"],
            )

    return action_history, video_history


# ---------------------------------------------------------------------------
# Phase 3 — assert losses decreased
# ---------------------------------------------------------------------------

def phase_assert(action_history: list[float], video_history: list[float], steps: int) -> None:
    log.info("[Phase 3] Comparing early vs late losses …")
    window = min(20, steps // 2)

    early_action = mean(action_history[:window])
    late_action  = mean(action_history[-window:])
    early_video  = mean(video_history[:window])
    late_video   = mean(video_history[-window:])

    log.info(
        "  action_loss  early=%.4f  late=%.4f  delta=%.4f",
        early_action, late_action, late_action - early_action,
    )
    log.info(
        "  video_loss   early=%.4f  late=%.4f  delta=%.4f",
        early_video, late_video, late_video - early_video,
    )

    assert late_action < early_action, (
        f"action_loss did not decrease: {early_action:.4f} -> {late_action:.4f}"
    )
    assert late_video < early_video, (
        f"video_loss did not decrease: {early_video:.4f} -> {late_video:.4f}"
    )
    print("SMOKE PASS")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end overfit smoke test for SmolVLA-UVA (bake → train → assert)"
    )
    p.add_argument("--dataset", required=True,
                   help="HF repo_id of the SOURCE dataset (e.g. eternalmay33/06_black_cup_red_bg_depth)")
    p.add_argument("--teacher", required=True,
                   help="Path to teacher SmolVLAPolicy checkpoint")
    p.add_argument("--baked-output", required=True,
                   help="Absolute path for the baked dataset copy")
    p.add_argument("--steps", type=int, default=100,
                   help="Number of training steps (default: 100)")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Batch size for training loop (default: 4)")
    p.add_argument("--skip-bake", action="store_true",
                   help="Skip Phase 1 — assume baked dataset already exists at --baked-output")
    p.add_argument("--force", action="store_true",
                   help="Pass --force to the bake script (allow overwriting output path)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_bake:
        phase_bake(args)
    else:
        log.info("[Phase 1] Skipped (--skip-bake)")

    action_history, video_history = phase_train(args)
    phase_assert(action_history, video_history, args.steps)


if __name__ == "__main__":
    main()
