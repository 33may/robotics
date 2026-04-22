"""Append a finished run's metrics to runs_index.csv + results.md.

Reads per-cam metrics.csv produced by distill.py (training/<run>/<cam>/metrics.csv),
extracts the row at best_epoch (min pixel-score), writes:

  1. One row per cam in training/runs_index.csv
  2. One block per run in training/results.md

Usage:
    python scripts/log_run.py --run v1_baseline --parent "" \
        --started "2026-04-20T21:00:00" --ended "2026-04-20T22:30:00" \
        --verdict win --notes "first new-pipeline baseline" \
        [--inference-b1 '2.1,2.0,2.2,2.1' --inference-b4 '3.4']
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd

TRAINING_ROOT = Path(
    "/home/may33/Documents/Obsidian Vault/vbti/researches/"
    "engineering tricks/detection/distillation/training"
)
INDEX_CSV = TRAINING_ROOT / "runs_index.csv"
RESULTS_MD = TRAINING_ROOT / "results.md"

CAMERAS = ["left", "right", "top", "gripper"]

INDEX_FIELDS = [
    "run_name", "parent", "cam", "started_at", "ended_at",
    "best_epoch", "epochs_ran", "wall_time_s",
    "main_duck_med_px", "main_duck_p95_px",
    "main_cup_med_px", "main_cup_p95_px",
    "main_duck_fp", "main_cup_fp",
    "no_obj_duck_fp", "no_obj_cup_fp",
    "inference_ms_b1", "inference_ms_b4",
    "verdict", "notes",
]


def _get(row, key, default=""):
    v = row.get(key, default)
    try:
        import math as _m
        if isinstance(v, float) and _m.isnan(v):
            return ""
    except Exception:
        pass
    return v


def _fmt_px(v):
    if v == "" or v is None:
        return ""
    try:
        return f"{float(v):.1f}"
    except Exception:
        return str(v)


def _fmt_fp(v):
    if v == "" or v is None:
        return ""
    try:
        return f"{float(v):.3f}"
    except Exception:
        return str(v)


def load_best_row(run: str, cam: str) -> dict | None:
    p = TRAINING_ROOT / run / cam / "metrics.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if len(df) == 0:
        return None
    # best = min pixel-score (matches distill.py early-stop monitor)
    if "score_pixel" in df.columns:
        best_idx = df["score_pixel"].idxmin()
    else:
        # fallback: sum medians
        df["_score"] = df["main_duck_err_median_px"].fillna(9999) + df["main_cup_err_median_px"].fillna(9999)
        best_idx = df["_score"].idxmin()
    best = df.loc[best_idx].to_dict()
    best["_epochs_ran"] = int(df["epoch"].max())
    best["_best_epoch"] = int(best["epoch"])
    return best


def _ensure_header():
    if INDEX_CSV.exists():
        with INDEX_CSV.open() as fp:
            first = fp.readline().strip()
        if first == ",".join(INDEX_FIELDS):
            return
    with INDEX_CSV.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=INDEX_FIELDS)
        w.writeheader()


def append_index_rows(
    run: str, parent: str, started: str, ended: str, wall_s: float,
    verdict: str, notes: str,
    inference_b1: dict[str, float] | None = None,
    inference_b4: float | None = None,
):
    _ensure_header()
    rows = []
    for cam in CAMERAS:
        best = load_best_row(run, cam)
        if best is None:
            rows.append({
                "run_name": run, "parent": parent, "cam": cam,
                "started_at": started, "ended_at": ended,
                "verdict": "crashed",
                "notes": f"no metrics.csv; {notes}",
            })
            continue
        row = {
            "run_name": run,
            "parent": parent,
            "cam": cam,
            "started_at": started,
            "ended_at": ended,
            "best_epoch": best.get("_best_epoch", ""),
            "epochs_ran": best.get("_epochs_ran", ""),
            "wall_time_s": int(wall_s),
            "main_duck_med_px": _fmt_px(_get(best, "main_duck_err_median_px")),
            "main_duck_p95_px": _fmt_px(_get(best, "duck_err_p95_px")),
            "main_cup_med_px": _fmt_px(_get(best, "main_cup_err_median_px")),
            "main_cup_p95_px": _fmt_px(_get(best, "cup_err_p95_px")),
            "main_duck_fp": _fmt_fp(_get(best, "duck_false_positive_rate")),
            "main_cup_fp": _fmt_fp(_get(best, "cup_false_positive_rate")),
            "no_obj_duck_fp": _fmt_fp(_get(best, "no_obj_duck_false_positive_rate")),
            "no_obj_cup_fp": _fmt_fp(_get(best, "no_obj_cup_false_positive_rate")),
            "inference_ms_b1": f"{inference_b1[cam]:.2f}" if inference_b1 and cam in inference_b1 else "",
            "inference_ms_b4": f"{inference_b4:.2f}" if inference_b4 is not None else "",
            "verdict": verdict,
            "notes": notes,
        }
        rows.append(row)

    with INDEX_CSV.open("a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=INDEX_FIELDS)
        for r in rows:
            w.writerow({k: r.get(k, "") for k in INDEX_FIELDS})
    return rows


def append_results_block(run: str, parent: str, ended: str, verdict: str,
                        rows: list[dict], notes: str):
    lines = []
    lines.append(f"\n## {run}  —  {ended}")
    lines.append(f"Parent: `{parent or '-'}`    Verdict: **{verdict}**")
    if notes:
        lines.append(f"Notes: {notes}")
    lines.append("")
    lines.append("| cam | best ep | epochs | duck med px | duck p95 | cup med px | cup p95 | main_fp d/c | no_obj_fp d/c |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")
    for r in rows:
        lines.append(
            f"| {r['cam']} | {r.get('best_epoch','')} | {r.get('epochs_ran','')} | "
            f"{r.get('main_duck_med_px','')} | {r.get('main_duck_p95_px','')} | "
            f"{r.get('main_cup_med_px','')} | {r.get('main_cup_p95_px','')} | "
            f"{r.get('main_duck_fp','')}/{r.get('main_cup_fp','')} | "
            f"{r.get('no_obj_duck_fp','')}/{r.get('no_obj_cup_fp','')} |"
        )
    lines.append("")
    with RESULTS_MD.open("a") as fp:
        fp.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--parent", default="")
    p.add_argument("--started", default=datetime.now().isoformat(timespec="seconds"))
    p.add_argument("--ended", default=datetime.now().isoformat(timespec="seconds"))
    p.add_argument("--wall-s", type=float, default=0.0)
    p.add_argument("--verdict", default="pending", choices=["win", "loss", "tied", "crashed", "pending"])
    p.add_argument("--notes", default="")
    p.add_argument("--inference-b1", default="", help="csv of 4 floats (left,right,top,gripper) ms")
    p.add_argument("--inference-b4", type=float, default=None, help="batched 4-cam ms")
    args = p.parse_args()

    b1_map = None
    if args.inference_b1:
        vals = [float(x) for x in args.inference_b1.split(",")]
        if len(vals) == 4:
            b1_map = dict(zip(CAMERAS, vals))

    rows = append_index_rows(
        args.run, args.parent, args.started, args.ended, args.wall_s,
        args.verdict, args.notes,
        inference_b1=b1_map, inference_b4=args.inference_b4,
    )
    append_results_block(args.run, args.parent, args.ended, args.verdict, rows, args.notes)
    print(f"[log_run] appended {len(rows)} rows for {args.run} to {INDEX_CSV.name} + results.md")


if __name__ == "__main__":
    main()
