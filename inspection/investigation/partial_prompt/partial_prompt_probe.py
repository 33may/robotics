#!/usr/bin/env python3
"""Which SAM 3 prompt survives a half-seen object? Single-view A/B, offline.

The failure this measures (runs 1009-lan1/2/3, black NETGEAR switch): passive
stereo returns depth on only part of a dark glossy object, so the accumulated
cloud is partial, the prompt box built from it frames only that part, and the
tracker — for which a box is a spatial constraint, not a hint — returns a
partial mask. Partial mask -> partial fusion -> the box can barely grow: the
run needs ~5-10 views of mutual crawl before box and mask converge
(lan1 001: 15.9k px at 0.64 vs the ~47k px the converged views measure).
`floor_z` (geometry.py:prompt_box) already breaks the VERTICAL version of
this loop; the lateral version is untouched.

Candidates, all derived from exactly what the run knew before segmenting
(the recorded prompt box and the previous step's fused cloud — never GT):

  base     the recorded box, replayed verbatim              (the control)
  loose25  box grown 25% of its own size per side
  loose50  box grown 50% of its own size per side
  iter     base, then re-prompt growing any side the mask presses against
  pts      positive points FPS-sampled from the projected cloud, no box
  boxpts   box + those points
  mask2    base mask fed back as input_masks with a grown box (SAM refine)
  detect   Sam3Model text="object", instance picked by projected-cloud
           containment, its box through the tracker

Pseudo-GT per view: the run's FINAL fused cloud projected into the view.
It is the best available stand-in for the full silhouette (the runs do
converge), identical across candidates, and never fed to any prompt.
recall = GT pixels the mask covers; spill = mask pixels far outside the GT
box (grabbing the mouse/table would show here).

    p inspection/investigation/partial_prompt/partial_prompt_probe.py --run 1009-lan1
    p inspection/investigation/partial_prompt/partial_prompt_probe.py --run 1009-lan1 --overlays all

Writes /tmp/partial_prompt_probe/<run>/{metrics.json,overlay_NNN.png}.
Recorded runs are read-only; everything lands in /tmp.
"""
import json
import time
from pathlib import Path

import numpy as np

from inspection.cell.geometry import (is_half_turn, project_to_pixels,
                                      rotate180)
from inspection.investigation.run_replay.step_replay import captures, run_dir_for

OUT = Path("/tmp/partial_prompt_probe")
#: mask pixels this far outside the GT box count as spill (px).
SPILL_PAD = 12
#: a side is "pressed against" when the mask fills this fraction of the
#: 4-px band just inside it.
HUG_BAND_PX = 4
HUG_MIN_FILL = 0.15
#: iter growth per pressed side, as a fraction of the box dim (40 px floor).
ITER_GROW = 0.35
ITER_ROUNDS = 3
N_POINTS = 5


class Sam3Probe:
    """Raw tracker/detector access — `Sam3Backend.segment` only takes a box,
    and this probe needs points, mask refinement and detection too. Pre/post
    processing mirrors eyes/verbs_local.py exactly (same classes, same
    post_process calls) so a candidate that wins here ports 1:1."""

    def __init__(self, device="cuda", dtype="bfloat16", repo="facebook/sam3"):
        self.device, self.dtype, self.repo = device, dtype, repo
        self._trk = self._trk_proc = self._det = self._det_proc = None

    def _tracker(self):
        if self._trk is None:
            import torch
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor
            self._trk_proc = Sam3TrackerProcessor.from_pretrained(self.repo)
            self._trk = Sam3TrackerModel.from_pretrained(
                self.repo,
                dtype=getattr(torch, self.dtype)).to(self.device).eval()
        return self._trk, self._trk_proc

    def _detector(self):
        if self._det is None:
            import torch
            from transformers import Sam3Model, Sam3Processor
            self._det_proc = Sam3Processor.from_pretrained(self.repo)
            self._det = Sam3Model.from_pretrained(
                self.repo,
                dtype=getattr(torch, self.dtype)).to(self.device).eval()
        return self._det, self._det_proc

    def track(self, rgb, box=None, points=None, mask_input=None,
              prev=None):
        """One tracker call -> (mask bool HxW, iou score, raw outputs).

        `mask_input`+`prev` is the documented SAM refine pattern: low-res
        logits from a previous call plus its image embeddings.
        """
        import torch
        model, proc = self._tracker()
        kw = {}
        if box is not None:
            kw["input_boxes"] = [[list(map(float, box))]]
        if points is not None:
            kw["input_points"] = [[[list(map(float, p)) for p in points]]]
            kw["input_labels"] = [[[1] * len(points)]]
        if mask_input is not None:
            inputs = proc(**kw, original_sizes=prev["original_sizes"],
                          return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = model(**inputs, input_masks=mask_input,
                            image_embeddings=prev["image_embeddings"],
                            multimask_output=False)
            sizes = prev["original_sizes"]
        else:
            inputs = proc(images=rgb, **kw,
                          return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = model(**inputs)
            sizes = inputs["original_sizes"]
        masks = proc.post_process_masks(out.pred_masks, sizes)[0]
        scores = out.iou_scores.flatten()
        best = int(scores.argmax())
        return (masks[0][best].cpu().numpy().astype(bool),
                float(scores[best]),
                {"original_sizes": sizes,
                 "image_embeddings": out.image_embeddings,
                 "pred_masks": out.pred_masks, "best": best})

    def detect(self, rgb, phrase="object"):
        """Text detection -> [(box, score, mask bool HxW)], all 200 raw
        queries; the caller's containment test is the real threshold."""
        import torch
        model, proc = self._detector()
        inputs = proc(images=rgb, text=phrase,
                      return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = model(**inputs)
        res = proc.post_process_instance_segmentation(
            out, threshold=0.0, target_sizes=[rgb.shape[:2]])[0]
        return [(tuple(b.tolist()), float(s), m.cpu().numpy().astype(bool))
                for b, s, m in zip(res["boxes"], res["scores"],
                                   res["masks"])]


# ------------------------------------------------------------- prompt makers
def grow_box(box, frac, shape):
    x0, y0, x1, y1 = box
    gx, gy = frac * (x1 - x0), frac * (y1 - y0)
    h, w = shape[:2]
    return (max(0.0, x0 - gx), max(0.0, y0 - gy),
            min(float(w), x1 + gx), min(float(h), y1 + gy))


def hugged_sides(mask, box):
    """Which box sides the mask presses against (band just inside each)."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    b = HUG_BAND_PX
    sides = {}
    bands = {"left": mask[y0:y1, x0:min(x0 + b, x1)],
             "right": mask[y0:y1, max(x1 - b, x0):x1],
             "top": mask[min(y0, mask.shape[0]):min(y0 + b, y1), x0:x1],
             "bottom": mask[max(y1 - b, y0):y1, x0:x1]}
    for name, band in bands.items():
        sides[name] = band.size > 0 and float(band.mean()) >= HUG_MIN_FILL
    return sides


def grow_sides(box, sides, shape):
    x0, y0, x1, y1 = box
    gx = max(ITER_GROW * (x1 - x0), 40.0)
    gy = max(ITER_GROW * (y1 - y0), 40.0)
    h, w = shape[:2]
    return (max(0.0, x0 - gx) if sides["left"] else x0,
            max(0.0, y0 - gy) if sides["top"] else y0,
            min(float(w), x1 + gx) if sides["right"] else x1,
            min(float(h), y1 + gy) if sides["bottom"] else y1)


def fps_points(px, k=N_POINTS):
    """Farthest-point sample k pixels from an (N,2) footprint — spread
    positive clicks, the SAMAug placement heuristic without a GT mask."""
    px = np.asarray(px, dtype=float)
    if len(px) <= k:
        return px
    picked = [int(np.argmin(np.linalg.norm(px - px.mean(0), axis=1)))]
    d = np.linalg.norm(px - px[picked[0]], axis=1)
    for _ in range(k - 1):
        nxt = int(d.argmax())
        picked.append(nxt)
        d = np.minimum(d, np.linalg.norm(px - px[nxt], axis=1))
    return px[picked]


# ----------------------------------------------------------------- the A/B
def candidates(probe, rgb, box, src_px):
    """name -> (mask, score, extra_calls). Every candidate sees only what
    the run saw: the recorded box and the projected prior cloud."""
    out = {}
    shape = rgb.shape

    base, s, raw = probe.track(rgb, box=box)
    out["base"] = (base, s, 0)

    for name, frac in (("loose25", 0.25), ("loose50", 0.50)):
        m, sc, _ = probe.track(rgb, box=grow_box(box, frac, shape))
        out[name] = (m, sc, 1)

    m, sc, b, calls = base, s, tuple(map(float, box)), 0
    for _ in range(ITER_ROUNDS):
        sides = hugged_sides(m, b)
        if not any(sides.values()):
            break
        b2 = grow_sides(b, sides, shape)
        if b2 == b:
            break
        b = b2
        m, sc, _ = probe.track(rgb, box=b)
        calls += 1
    out["iter"] = (m, sc, calls)
    out["iter_box"] = b

    if src_px is not None and len(src_px) >= N_POINTS:
        pts = fps_points(src_px)
        m, sc, _ = probe.track(rgb, points=pts)
        out["pts"] = (m, sc, 1)
        m, sc, _ = probe.track(rgb, box=box, points=pts)
        out["boxpts"] = (m, sc, 1)
        out["pts_used"] = pts.tolist()

    try:
        import torch
        mask_in = raw["pred_masks"][:, :, raw["best"]]
        m, sc, _ = probe.track(rgb, box=grow_box(box, 0.25, shape),
                               mask_input=mask_in, prev=raw)
        out["mask2"] = (m, sc, 1)
    except Exception as e:                                # noqa: BLE001
        out.setdefault("errors", {})["mask2"] = repr(e)

    if src_px is not None and len(src_px):
        try:
            dets = probe.detect(rgb)
            uv = np.round(src_px).astype(int)
            best, best_gain = None, 0.0
            for dbox, dscore, dmask in dets:
                if dscore < 0.2:
                    continue
                inside = dmask[np.clip(uv[:, 1], 0, shape[0] - 1),
                               np.clip(uv[:, 0], 0, shape[1] - 1)]
                gain = dscore * float(inside.mean())
                if gain > best_gain:
                    best, best_gain = (dbox, dscore, dmask), gain
            if best is not None:
                m, sc, _ = probe.track(rgb, box=best[0])
                out["detect"] = (m, sc, 2)
                out["detect_pick"] = {"box": [round(v, 1) for v in best[0]],
                                      "score": best[1],
                                      "gain": round(best_gain, 3)}
        except Exception as e:                            # noqa: BLE001
            out.setdefault("errors", {})["detect"] = repr(e)
    return out


def gt_pixels(final_cloud, T_base_cam, intr_color, shape, flip):
    """Final fused cloud -> integer pixel set in the MODEL frame."""
    uv = np.round(project_to_pixels(final_cloud, T_base_cam,
                                    intr_color)).astype(int)
    h, w = shape[:2]
    uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < w)
            & (uv[:, 1] >= 0) & (uv[:, 1] < h)]
    if flip:
        uv = np.column_stack([w - 1 - uv[:, 0], h - 1 - uv[:, 1]])
    return np.unique(uv, axis=0)


def score_mask(mask, gt_px):
    if mask is None or gt_px is None or not len(gt_px):
        return None
    h, w = mask.shape
    inside = mask[gt_px[:, 1], gt_px[:, 0]]
    lo = np.percentile(gt_px, 0.5, axis=0) - SPILL_PAD
    hi = np.percentile(gt_px, 99.5, axis=0) + SPILL_PAD
    ys, xs = np.nonzero(mask)
    out = ((xs < lo[0]) | (xs > hi[0]) | (ys < lo[1]) | (ys > hi[1]))
    return {"px": int(mask.sum()), "recall": round(float(inside.mean()), 4),
            "spill_px": int(out.sum())}


# ------------------------------------------------------------------ overlay
def draw_overlay(path, rgb, results, gt_px, box, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    names = [n for n in ("base", "loose25", "loose50", "iter", "pts",
                         "boxpts", "mask2", "detect") if n in results]
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 2.6 * rows))
    lo = np.percentile(gt_px, 0.5, axis=0)
    hi = np.percentile(gt_px, 99.5, axis=0)
    for ax, name in zip(np.atleast_1d(axes).ravel(), names):
        mask, sc, _ = results[name]
        img = rgb.copy()
        img[mask] = (0.45 * img[mask]
                     + 0.55 * np.array([255, 200, 0])).astype(np.uint8)
        ax.imshow(img)
        ax.add_patch(Rectangle(lo, *(hi - lo), fill=False,
                               edgecolor="#2ca02c", lw=1.2, ls="--"))
        b = results.get("iter_box") if name == "iter" else box
        if name == "detect" and "detect_pick" in results:
            b = results["detect_pick"]["box"]
        if name != "pts" and b is not None:
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                   fill=False, edgecolor="#d62728", lw=1.2))
        if name in ("pts", "boxpts") and "pts_used" in results:
            p = np.array(results["pts_used"])
            ax.scatter(p[:, 0], p[:, 1], c="#1f77b4", s=18, marker="x")
        m = score_mask(mask, gt_px)
        ax.set_title(f"{name}  recall {m['recall']:.2f}  "
                     f"{m['px'] / 1000:.1f}k px  s{sc:.2f}", fontsize=9)
        ax.axis("off")
    for ax in np.atleast_1d(axes).ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(title + "   (green dashed = final-cloud GT box, "
                 "red = prompt box)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------- main
def probe_run(run, out=str(OUT), overlays="failures"):
    run = str(run)
    run_dir = run_dir_for(run)
    session = json.loads((run_dir / "session.json").read_text())
    intr_color = session["intrinsics"]["color"]
    final = np.load(run_dir / "fused" / "cloud.npy")
    out_dir = Path(out) / run
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = Sam3Probe()
    rows = []
    for step_id, cap, stored in captures(run_dir):
        meta = json.loads(
            (run_dir / "steps" / f"{step_id:03d}" / "step.json").read_text())
        seg = meta.get("segmentation")
        if not seg:
            print(f"{step_id:03d}: no recorded segmentation — skipped")
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
            keep = ((uv[:, 0] >= 0) & (uv[:, 0] < w)
                    & (uv[:, 1] >= 0) & (uv[:, 1] < h))
            uv = uv[keep]
            if flip:
                uv = np.column_stack([w - 1 - uv[:, 0], h - 1 - uv[:, 1]])
            src_px = uv

        gt_px = gt_pixels(final, cap["T_base_cam"], intr_color,
                          rgb.shape, flip)
        if len(gt_px) < 50:
            print(f"{step_id:03d}: object out of frame in GT — skipped")
            continue

        t0 = time.time()
        res = candidates(probe, rgb, box, src_px)
        dt = time.time() - t0
        row = {"step": step_id, "flip": bool(flip),
               "recorded_score": seg["score"], "recorded_px": seg["px"],
               "ms": int(dt * 1000)}
        for name in ("base", "loose25", "loose50", "iter", "pts",
                     "boxpts", "mask2", "detect"):
            if name in res:
                mask, sc, extra = res[name]
                row[name] = dict(score_mask(mask, gt_px) or {},
                                 model_score=round(sc, 3),
                                 extra_calls=extra)
        if "errors" in res:
            row["errors"] = res["errors"]
        rows.append(row)

        base_recall = row["base"]["recall"]
        want = (overlays == "all" or base_recall < 0.95
                or step_id % 8 == 0)
        if want:
            draw_overlay(out_dir / f"overlay_{step_id:03d}.png", rgb, res,
                         gt_px, box,
                         f"{run} view {step_id:03d} — base recall "
                         f"{base_recall:.2f}")
        print(f"{step_id:03d} base {base_recall:.2f} | " + " | ".join(
            f"{n} {row[n]['recall']:.2f}" for n in
            ("loose25", "loose50", "iter", "pts", "boxpts", "mask2",
             "detect") if n in row))

    (out_dir / "metrics.json").write_text(
        json.dumps({"run": run, "rows": rows}, indent=1) + "\n")

    fail = [r for r in rows if r["base"]["recall"] < 0.95]
    ok = [r for r in rows if r["base"]["recall"] >= 0.95]
    print(f"\n== {run}: {len(fail)} failure views, {len(ok)} control views")
    for name in ("base", "loose25", "loose50", "iter", "pts", "boxpts",
                 "mask2", "detect"):
        f = [r[name]["recall"] for r in fail if name in r]
        o = [r[name]["recall"] for r in ok if name in r]
        sp = [r[name]["spill_px"] for r in rows if name in r]
        if f or o:
            print(f"  {name:8s} fail-mean {np.mean(f):.3f} (n={len(f)})  "
                  f"ctrl-mean {np.mean(o):.3f}  "
                  f"spill p95 {int(np.percentile(sp, 95)) if sp else '-'}")
    print(f"metrics: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(probe_run)
