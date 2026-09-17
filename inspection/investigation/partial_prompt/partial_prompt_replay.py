#!/usr/bin/env python3
"""Closed-loop replay: detect-anchored prompts vs the recorded lan runs.

The single-view A/B (`partial_prompt_probe.py`) showed WHY the partial-box
failure happens (a box is an extent ceiling) and which prompt escapes it:
Sam3Model detection picked by projected-cloud containment recovers the full
object on the failure views (recall 0.87-0.90 vs base 0.31-0.70 on
1009-lan1 001-005) and matches the base box everywhere else, while every
box-enlargement scheme (loose25/50, edge-hug iteration) collapses the
tracker score to 0.01-0.45 and pulls in the table.

This replays the WHOLE loop with that policy: the shipped
`run.segmenter.object_view` — same call `machine.py` makes, same floor_z,
the recorded table plane forced back in (deterministic, the step_replay.py
recipe) — but the segmenter now anchors its prompt on the detector:

  1. Sam3Model text="object" on the (upright) frame -> ~200 instances.
  2. Identity = the projected prior cloud: the instance whose mask contains
     the most of what the arm already knows it saw, weighted by detector
     score; containment < 0.5 means "the detector never saw our object"
     and the policy steps down.
  3. Sam3Tracker on that instance's box; the shipped score/px gates stay.
  4. Any miss falls back to the recorded-behavior box prompt, then to the
     depth path — strictly ADDING a rung above the shipped ladder.

The recorded runs themselves are the baseline: they ran the identical loop
with the box-only prompt. Every per-step number here is compared against
step.json written live by the robot.

    p inspection/investigation/partial_prompt/partial_prompt_replay.py --run 1009-lan1

Writes /tmp/partial_prompt_replay/<run>/{steps.json,curves.png,masks_NNN.png}.
Recorded runs stay read-only.
"""
import json
from pathlib import Path

import numpy as np

from inspection.cell.geometry import (CloudAccumulator, is_half_turn,
                                      project_to_pixels, rotate180)
from inspection.investigation.partial_prompt.partial_prompt_probe import Sam3Probe
from inspection.investigation.run_replay.step_replay import (_forced_plane, captures,
                                                  recorded, run_dir_for)
from inspection.run.segmenter import (MIN_MASK_PX, MIN_MASK_SCORE,
                                      Segmentation, object_view)

OUT = Path("/tmp/partial_prompt_replay")
#: below this fraction of the projected prior cloud inside a detection's
#: mask, the detection is not our object — step down to the box prompt.
MIN_CONTAINMENT = 0.5
MIN_DET_SCORE = 0.2


def floor_z_from_cell():
    """The same probed-table z machine.py hands the live loop."""
    import yaml
    from inspection.cell.world import CELL_YAML, _frame_to_base
    cfg = yaml.safe_load(Path(CELL_YAML).read_text())
    return float(_frame_to_base(cfg["frames"], "table").translation[2])


class DetectAnchoredSegmenter:
    """`mask_for`-compatible: detector box when identity confirms it,
    shipped box prompt otherwise. `src_px` (the projected prior cloud, the
    same `source` object_view builds its prompt box from) is set per view
    by the replay; in a shipped integration it is in scope inside
    `object_view` — no signature change needed there."""

    def __init__(self, probe=None):
        self.probe = probe or Sam3Probe()
        self.src_px = None          # (N,2) float px, RAW frame, or None
        self.calls = self.misses = 0
        self.branches = []          # per accepted view: "detect" | "box"

    def _pick(self, frame, src):
        dets = self.probe.detect(frame)
        best, best_gain, best_cont = None, 0.0, 0.0
        uv = np.round(src).astype(int)
        h, w = frame.shape[:2]
        uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < w)
                & (uv[:, 1] >= 0) & (uv[:, 1] < h)]
        if not len(uv):
            return None, 0.0
        for box, score, mask in dets:
            if score < MIN_DET_SCORE:
                continue
            cont = float(mask[uv[:, 1], uv[:, 0]].mean())
            if cont < MIN_CONTAINMENT:
                continue
            if score * cont > best_gain:
                best, best_gain, best_cont = box, score * cont, cont
        return best, best_cont

    def mask_for(self, rgb, box, T_base_cam=None, src_px=None):
        # `src_px` (now also plumbed by the shipped object_view) is ignored:
        # this class predates the integration and sets its own self.src_px.
        if rgb is None or box is None:
            return None
        self.calls += 1
        h, w = rgb.shape[:2]
        flip = T_base_cam is not None and is_half_turn(T_base_cam)
        if flip:
            x0, y0, x1, y1 = box
            frame, prompt = rotate180(rgb), (w - x1, h - y1, w - x0, h - y0)
            src = (None if self.src_px is None else np.column_stack(
                [w - 1 - self.src_px[:, 0], h - 1 - self.src_px[:, 1]]))
        else:
            frame, prompt, src = rgb, tuple(box), self.src_px

        tries = []
        if src is not None and len(src):
            try:
                det_box, cont = self._pick(frame, src)
                if det_box is not None:
                    tries.append(("detect", det_box))
            except Exception:                             # noqa: BLE001
                pass
        tries.append(("box", prompt))

        for branch, b in tries:
            try:
                mask, score, _ = self.probe.track(frame, box=b)
            except Exception:                             # noqa: BLE001
                continue
            n = int(mask.sum())
            if score < MIN_MASK_SCORE or n < MIN_MASK_PX:
                continue
            self.branches.append(branch)
            if flip:
                mask = rotate180(mask)
                x0, y0, x1, y1 = b
                b = (w - x1, h - y1, w - x0, h - y0)
            return Segmentation(np.asarray(mask, dtype=bool),
                                tuple(int(v) for v in b), float(score))
        self.misses += 1
        return None


# --------------------------------------------------------------------- main
def replay(run, out=str(OUT), shipped=False):
    """`--shipped` drives the PRODUCTION ObjectSegmenter (whose detect rung
    landed from this investigation) instead of the prototype policy class —
    the closed-loop check that the integration reproduces the benchmark."""
    run = str(run)
    run_dir = run_dir_for(run)
    session = json.loads((run_dir / "session.json").read_text())
    intr = session["intrinsics"]["ir_left"]
    intr_color = session["intrinsics"]["color"]
    scale = session["depth_scale_m_per_unit"]
    rec = recorded(run_dir)
    floor_z = floor_z_from_cell()
    out_dir = Path(out) / run
    out_dir.mkdir(parents=True, exist_ok=True)

    if shipped:
        from inspection.run.segmenter import ObjectSegmenter
        seg = ObjectSegmenter()
    else:
        seg = DetectAnchoredSegmenter()
    acc = CloudAccumulator()
    steps = []
    print(f"floor_z {floor_z:.4f}   {len(rec)} fused steps to replay")
    print(f"{'k':>3} {'branch':>7} {'px':>6} {'rec px':>6} {'score':>5} "
          f"{'offered':>7} {'cloud':>6} {'extent mm':>22} {'rec ext':>22}")
    for step_id, cap, _stored in captures(run_dir):
        if step_id not in rec:
            continue
        meta = json.loads(
            (run_dir / "steps" / f"{step_id:03d}" / "step.json").read_text())
        _, _, plane = rec[step_id]
        seg.src_px = None
        if len(acc.points):
            uv = project_to_pixels(acc.points, cap["T_base_cam"], intr_color)
            seg.src_px = uv
        fell = []
        n_branch = len(seg.branches)
        with _forced_plane(plane) if plane is not None else _null():
            view, s = object_view(cap, intr, intr_color, scale, acc.points,
                                  seg, on_fallback=fell.append,
                                  floor_z=floor_z)
        # The prototype needed a hand-rolled first-view retry; the shipped
        # rung gets src_px from object_view on EVERY view, survey included.
        if (not shipped and seg.src_px is None and s is None
                and view.get("prompt_box")):
            # First view: object_view built its box from this frame's own
            # depth cluster; give the detector the same evidence and retry.
            seg.src_px = project_to_pixels(view["prompt_source"],
                                           cap["T_base_cam"], intr_color)
            s = seg.mask_for(cap.get("rgb"), view["prompt_box"],
                             cap["T_base_cam"])
            if s is not None:
                from inspection.cell.geometry import object_in_base
                masked = object_in_base(cap["depth_aligned"], intr_color,
                                        scale, cap["T_base_cam"],
                                        mask=s.mask, rgb=cap.get("rgb"))
                if masked["centroid"] is not None:
                    view, fell = masked, []
        pts = view["points"]
        if len(pts):
            acc.add(pts, view.get("colors"))
        mn, mx = (acc.aabb() if len(acc.points)
                  else (np.zeros(3), np.zeros(3)))
        ext = np.round((mx - mn) * 1000, 1)
        # Raw min/max is fuzz-sensitive; the q0.5-99.5 spread says whether
        # growth is real coverage or a few boundary points.
        rob = (np.round((np.percentile(acc.points, 99.5, axis=0)
                         - np.percentile(acc.points, 0.5, axis=0))
                        * 1000, 1).tolist() if len(acc.points) else [0, 0, 0])
        np.save(out_dir / f"step_{step_id:03d}.npy", acc.points)
        rseg = meta.get("segmentation") or {}
        rext = (meta.get("geometry") or {}).get("extent_mm")
        branch = seg.branches[n_branch] if len(seg.branches) > n_branch \
            else "depth"
        steps.append({"step": step_id, "branch": branch,
                      "px": None if s is None else int(s.mask.sum()),
                      "score": None if s is None else round(s.score, 3),
                      "recorded_px": rseg.get("px"),
                      "recorded_score": rseg.get("score"),
                      "offered": int(len(pts)),
                      "cloud": int(len(acc.points)),
                      "extent_mm": ext.tolist(),
                      "robust_extent_mm": rob,
                      "recorded_extent_mm": rext,
                      "fallbacks": fell})
        print(f"{step_id:>3} {branch:>7} "
              f"{steps[-1]['px'] or 0:>6} {rseg.get('px') or 0:>6} "
              f"{steps[-1]['score'] or 0:>5} {len(pts):>7} "
              f"{len(acc.points):>6} {str(ext):>22} {str(rext):>22}")

    if len(acc.points):
        np.save(out_dir / "cloud.npy", acc.points)
        if acc.colors is not None:
            np.save(out_dir / "colors.npy", acc.colors)
    (out_dir / "steps.json").write_text(json.dumps(
        {"run": run, "floor_z": floor_z, "steps": steps,
         "branches": seg.branches, "misses": seg.misses}, indent=1) + "\n")
    render_curves(out_dir, run, steps)
    area = [s["extent_mm"][0] * s["extent_mm"][1] for s in steps]
    rarea = [s["recorded_extent_mm"][0] * s["recorded_extent_mm"][1]
             if s["recorded_extent_mm"] else 0.0 for s in steps]
    # One yardstick for both curves: the footprint the RECORDED run ended
    # at — "own max" would penalise whichever curve ends larger.
    tgt = 0.95 * max(rarea)
    for name, a in (("replayed", area), ("recorded", rarea)):
        k = next((i for i, v in enumerate(a) if v >= tgt), None)
        at = ("never reached" if k is None
              else f"reached at view {k} ({steps[k]['step']:03d})")
        print(f"{name}: 95% of the recorded final xy footprint {at}")
    print(f"branches: { {b: seg.branches.count(b) for b in set(seg.branches)} }"
          f"   misses {seg.misses}")
    print(f"out: {out_dir}")


def render_curves(out_dir, run, steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = list(range(len(steps)))
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    a.plot(k, [s["recorded_px"] or 0 for s in steps], "-o", ms=3,
           color="#1f77b4", label="recorded run (box prompt)")
    a.plot(k, [s["px"] or 0 for s in steps], "-o", ms=3,
           color="#d62728", label="replay (detect-anchored)")
    a.set_ylabel("mask [px]")
    a.grid(alpha=0.3)
    a.legend(fontsize=9)
    a.set_title(f"{run}: mask size and object footprint per view")
    b.plot(k, [s["recorded_extent_mm"][0] * s["recorded_extent_mm"][1] / 100
               if s["recorded_extent_mm"] else 0 for s in steps], "-o",
           ms=3, color="#1f77b4", label="recorded")
    b.plot(k, [s["extent_mm"][0] * s["extent_mm"][1] / 100 for s in steps],
           "-o", ms=3, color="#d62728", label="replay")
    b.set_ylabel("cloud xy footprint [cm$^2$]")
    b.set_xlabel("fused view #")
    b.grid(alpha=0.3)
    b.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=140)
    plt.close(fig)


from contextlib import contextmanager   # noqa: E402


@contextmanager
def _null():
    yield


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(replay)
