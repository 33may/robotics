#!/usr/bin/env python3
"""False-positive audit of detect-anchored prompting, per view, all runs.

The policy's failure mode that would MATTER is not a missed view (that
falls back to today's behaviour) — it is a WRONG mask that passes the
gates: the detector picking the mouse, the table, or a merged
switch+shadow instance, and the loop fusing it. This measures exactly
that, three ways:

1. **Mask level** — run the full ladder (detect -> containment pick ->
   tracker -> score/px gates -> box-prompt fallback) on every recorded
   view of every benchmark run, with the same inputs the run had. Score
   each accepted mask against the final-cloud silhouette: recall (got the
   object) and precision (nothing but the object, up to a 12 px pad).
2. **Decision level** — every detection the containment guard REJECTED
   despite confident objectness (score >= 0.5, containment < 0.5): what
   the detector was tempted by. Galleries show the temptations next to
   the pick.
3. **Where it lands** — worst-precision and worst-recall accepted views
   rendered as overlays, so the worst thing the policy ever did is a
   picture, not a percentile.

    p inspection/investigation/partial_prompt/false_positive_audit.py
    p inspection/investigation/partial_prompt/false_positive_audit.py --runs 1009-lan1

Writes /tmp/partial_prompt_audit/{audit.json,worst_*.png,tempt_*.png}.
"""
import json
from pathlib import Path

import numpy as np

from inspection.cell.geometry import is_half_turn, project_to_pixels, rotate180
from inspection.investigation.partial_prompt.partial_prompt_probe import (
    Sam3Probe, gt_pixels)
from inspection.investigation.run_replay.step_replay import (captures,
                                                             run_dir_for)
from inspection.run.segmenter import MIN_MASK_PX, MIN_MASK_SCORE

OUT = Path("/tmp/partial_prompt_audit")
RUNS = ("1009-lan1", "1009-lan2", "1009-lan3", "1009-cup1", "1009-cup2")
PAD = 12
MIN_CONTAINMENT = 0.5
MIN_DET_SCORE = 0.2


def pr(mask, gt_px):
    """(recall, precision) vs the final-cloud silhouette box, PAD-padded."""
    inside = mask[gt_px[:, 1], gt_px[:, 0]]
    lo = np.percentile(gt_px, 0.5, axis=0) - PAD
    hi = np.percentile(gt_px, 99.5, axis=0) + PAD
    ys, xs = np.nonzero(mask)
    n = max(1, len(xs))
    out = ((xs < lo[0]) | (xs > hi[0]) | (ys < lo[1]) | (ys > hi[1])).sum()
    return float(inside.mean()), float(1.0 - out / n)


def audit_view(probe, rgb, box, src_px):
    """Run the ladder once; return (branch, mask, extras)."""
    dets = tempted = None
    picked = None
    if src_px is not None and len(src_px):
        uv = np.round(src_px).astype(int)
        h, w = rgb.shape[:2]
        uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < w)
                & (uv[:, 1] >= 0) & (uv[:, 1] < h)]
        if len(uv):
            dets = probe.detect(rgb)
            scored = []
            for dbox, dscore, dmask in dets:
                if dscore < MIN_DET_SCORE:
                    continue
                cont = float(dmask[uv[:, 1], uv[:, 0]].mean())
                scored.append((dbox, dscore, cont, dmask))
            ok = [d for d in scored if d[2] >= MIN_CONTAINMENT]
            tempted = sorted((d for d in scored if d[2] < MIN_CONTAINMENT
                              and d[1] >= 0.5),
                             key=lambda d: -d[1])[:3]
            if ok:
                picked = max(ok, key=lambda d: d[1] * d[2])
    tries = ([("detect", picked[0])] if picked else []) + [("box", box)]
    for branch, b in tries:
        mask, score, _ = probe.track(rgb, box=b)
        if score >= MIN_MASK_SCORE and int(mask.sum()) >= MIN_MASK_PX:
            return branch, mask, score, b, picked, tempted
    return "miss", None, None, None, picked, tempted


def overlay(ax, rgb, mask, gt_px, box, title, color=(255, 200, 0)):
    from matplotlib.patches import Rectangle
    img = rgb.copy()
    if mask is not None:
        img[mask] = (0.45 * img[mask] + 0.55 * np.array(color)).astype(np.uint8)
    ax.imshow(img)
    if gt_px is not None:
        lo = np.percentile(gt_px, 0.5, axis=0)
        hi = np.percentile(gt_px, 99.5, axis=0)
        ax.add_patch(Rectangle(lo, *(hi - lo), fill=False,
                               edgecolor="#2ca02c", lw=1.3, ls="--"))
    if box is not None:
        ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0],
                               box[3] - box[1], fill=False,
                               edgecolor="#d62728", lw=1.3))
    ax.set_title(title, fontsize=8.5)
    ax.axis("off")


def main(runs=",".join(RUNS)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    probe = Sam3Probe()
    rows, worst_pool, tempt_pool = [], [], []
    for run in str(runs).split(","):
        run_dir = run_dir_for(run)
        session = json.loads((run_dir / "session.json").read_text())
        intr_color = session["intrinsics"]["color"]
        final = np.load(run_dir / "fused" / "cloud.npy")
        for step_id, cap, _ in captures(run_dir):
            meta = json.loads((run_dir / "steps" / f"{step_id:03d}"
                               / "step.json").read_text())
            seg = meta.get("segmentation")
            if not seg:
                continue
            flip = is_half_turn(cap["T_base_cam"])
            rgb = rotate180(cap["rgb"]) if flip else cap["rgb"]
            h, w = rgb.shape[:2]
            box = seg["box"]
            if flip:
                x0, y0, x1, y1 = box
                box = (w - x1, h - y1, w - x0, h - y0)
            prev = None
            for d in sorted((run_dir / "steps").iterdir(), reverse=True):
                if (d.is_dir() and d.name.isdigit() and int(d.name) < step_id
                        and (d / "fused" / "cloud.npy").exists()):
                    prev = np.load(d / "fused" / "cloud.npy")
                    break
            src_px = None
            if prev is not None and len(prev):
                uv = project_to_pixels(prev, cap["T_base_cam"], intr_color)
                if flip:
                    uv = np.column_stack([w - 1 - uv[:, 0],
                                          h - 1 - uv[:, 1]])
                src_px = uv
            gt_px = gt_pixels(final, cap["T_base_cam"], intr_color,
                              rgb.shape, flip)
            if len(gt_px) < 50:
                continue
            branch, mask, score, used, picked, tempted = audit_view(
                probe, rgb, box, src_px)
            rec = prec = None
            if mask is not None:
                rec, prec = pr(mask, gt_px)
                worst_pool.append((prec, rec, run, step_id, rgb, mask,
                                   gt_px, used, branch))
            if tempted:
                tempt_pool.append((run, step_id, rgb, gt_px, tempted,
                                   picked, mask))
            rows.append({"run": run, "step": step_id, "branch": branch,
                         "recall": rec, "precision": prec,
                         "track_score": score,
                         "picked_det_score": picked[1] if picked else None,
                         "picked_containment": round(picked[2], 3)
                         if picked else None,
                         "n_tempted": len(tempted or [])})
            print(f"{run} {step_id:03d} {branch:6s} "
                  f"recall {rec if rec is None else round(rec, 2)} "
                  f"precision {prec if prec is None else round(prec, 3)} "
                  f"tempted {len(tempted or [])}")

    (OUT / "audit.json").write_text(json.dumps(rows, indent=1) + "\n")

    accepted = [r for r in rows if r["precision"] is not None]
    det = [r for r in accepted if r["branch"] == "detect"]
    print(f"\n{len(accepted)} accepted views ({len(det)} detect-branch), "
          f"{sum(1 for r in rows if r['branch'] == 'miss')} misses")
    for name, key in (("precision", "precision"), ("recall", "recall")):
        v = np.array([r[key] for r in accepted])
        print(f"  {name}: min {v.min():.3f}  p5 {np.percentile(v, 5):.3f}  "
              f"median {np.median(v):.3f}")
    bad = [r for r in accepted if r["precision"] < 0.90]
    print(f"  views with precision < 0.90: {len(bad)} of {len(accepted)}"
          f"  -> {[(r['run'], r['step']) for r in bad]}")
    n_t = sum(r["n_tempted"] for r in rows)
    print(f"  confident-but-rejected detections (temptations): {n_t} "
          f"across {sum(1 for r in rows if r['n_tempted'])} views; "
          f"wrong pick accepted: "
          f"{sum(1 for r in det if r['recall'] < 0.3)}")

    # Galleries -----------------------------------------------------------
    worst_pool.sort(key=lambda t: t[0])
    for tag, pool in (("worst_precision", worst_pool[:6]),
                      ("worst_recall",
                       sorted(worst_pool, key=lambda t: t[1])[:6])):
        fig, axes = plt.subplots(2, 3, figsize=(13, 5.4))
        for ax, (prec, rec, run, sid, rgb, mask, gt_px, used, br) in zip(
                axes.ravel(), pool):
            overlay(ax, rgb, mask, gt_px, used,
                    f"{run} {sid:03d} [{br}] recall {rec:.2f} "
                    f"precision {prec:.3f}")
        for ax in axes.ravel()[len(pool):]:
            ax.axis("off")
        fig.suptitle(f"{tag.replace('_', ' ')} — the worst accepted masks "
                     "across all runs (green dashed = object GT, "
                     "red = prompt used)", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f"{tag}.png", dpi=110)
        plt.close(fig)

    for i, (run, sid, rgb, gt_px, tempted, picked, mask) in enumerate(
            tempt_pool[:4]):
        fig, axes = plt.subplots(1, len(tempted) + 1,
                                 figsize=(4.4 * (len(tempted) + 1), 2.9))
        axes = np.atleast_1d(axes)
        for ax, (dbox, dscore, cont, dmask) in zip(axes, tempted):
            overlay(ax, rgb, dmask, gt_px, dbox,
                    f"REJECTED det {dscore:.2f} containment {cont:.2f}",
                    color=(230, 60, 60))
        overlay(axes[-1], rgb, mask, gt_px, None,
                f"PICKED -> tracker (containment "
                f"{picked[2]:.2f})" if picked else "fallback: box prompt")
        fig.suptitle(f"{run} view {sid:03d} — what the detector offered vs "
                     "what containment let through", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"tempt_{i}_{run}_{sid:03d}.png", dpi=110)
        plt.close(fig)
    print(f"galleries: {OUT}")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(main)
