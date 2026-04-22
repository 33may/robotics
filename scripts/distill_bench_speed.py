"""Measure inference latency for a trained distillation run.

Loads best.pt from training/<run>/<cam>/ for all 4 cams and measures:
  - b1: single-frame latency per cam (ms)
  - b4: batched 4-cam inference latency (ms) — stack all 4 cam inputs, one forward pass

Writes:
  - training/<run>/inference_bench.json (machine-readable)
  - appends a section to training/<run>/verdict.md if it exists
  - prints both numbers to stdout in the csv format log_run.py expects:
      INFERENCE_B1=2.1,2.0,2.2,2.1
      INFERENCE_B4=3.4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from vbti.logic.detection.distill_model import DistilledDetector

TRAINING_ROOT = Path(
    "/home/may33/Documents/Obsidian Vault/vbti/researches/"
    "engineering tricks/detection/distillation/training"
)
CAMERAS = ["left", "right", "top", "gripper"]
IMG_SIZE = 224
WARMUP = 30
ITERS = 200


def _load_student(run: str, cam: str, device: torch.device) -> DistilledDetector:
    ckpt_path = TRAINING_ROOT / run / cam / "best.pt"
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    backbone = cfg.get("model_backbone", "mobilenet_v3_small")
    model = DistilledDetector(backbone=backbone, pretrained=False).eval().to(device)
    model.load_state_dict(ck["model"])
    return model


@torch.inference_mode()
def bench_b1(model: DistilledDetector, device: torch.device) -> float:
    x = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    # warmup
    for _ in range(WARMUP):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1000.0  # ms/iter


@torch.inference_mode()
def bench_b4_4cams(models: list[DistilledDetector], device: torch.device) -> float:
    """Simulate 4-cam deployment: 4 per-cam models, each given a batch=1 input.
    We run them sequentially (not stacked) because cam models have different weights.
    """
    x = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    # warmup
    for _ in range(WARMUP):
        for m in models:
            _ = m(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        for m in models:
            _ = m(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1000.0  # ms for all 4 cams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[bench] device: {device}")
    print(f"[bench] run: {args.run}  warmup={WARMUP}  iters={ITERS}")

    students = {}
    per_cam_b1 = {}
    for cam in CAMERAS:
        print(f"[bench] loading {cam} best.pt ...")
        m = _load_student(args.run, cam, device)
        students[cam] = m
        ms = bench_b1(m, device)
        per_cam_b1[cam] = ms
        print(f"[bench]   {cam:8s} b1 = {ms:.3f} ms/frame")

    models_list = [students[c] for c in CAMERAS]
    b4_ms = bench_b4_4cams(models_list, device)
    print(f"[bench] 4-cam sequential (per 4-cam inference cycle): {b4_ms:.3f} ms")
    print(f"[bench] 4-cam equiv per-cam: {b4_ms/4:.3f} ms  (effective FPS per cam at 4-cam sync: {1000/b4_ms:.1f})")

    out = {
        "run": args.run,
        "device": str(device),
        "warmup": WARMUP,
        "iters": ITERS,
        "img_size": IMG_SIZE,
        "per_cam_ms_b1": per_cam_b1,
        "four_cam_sequential_ms": b4_ms,
        "four_cam_fps": 1000.0 / b4_ms,
    }
    out_path = TRAINING_ROOT / args.run / "inference_bench.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[bench] wrote {out_path}")

    # Print in the format log_run.py consumes
    b1_csv = ",".join(f"{per_cam_b1[c]:.3f}" for c in CAMERAS)
    print(f"\nINFERENCE_B1={b1_csv}")
    print(f"INFERENCE_B4={b4_ms:.3f}")


if __name__ == "__main__":
    main()
