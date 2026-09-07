#!/usr/bin/env python3
"""Replay a recorded run's object cloud view by view, and check it exact.

Why this exists. `cloud_replay.py` next door is depth-only: it grows from
`depth_raw` with a seed and never looks at a mask, so for any run captured
after 2026-08-24 it replays a pipeline the robot did not run. These runs
fused SAM 3 masks — `run/segmenter.py:159` deprojects `depth_aligned`
masked by the segmentation, with the COLOUR intrinsics, and the depth path
is only the fallback. This module replays THAT, and writes the cloud after
every view, so "when did the cloud stop being the cup" is answerable per
capture instead of only at the end.

How it stays the loop rather than a copy of it. The stored `mask.png` is
handed back through a stub segmenter, so the function that runs is
`run.segmenter.object_view` — the same call `machine.py:518` makes, same
argument order — instead of a hand-rolled `object_in_base(..., mask=)`.
The stub is the smaller lie of the two: `object_view` also owns the prompt
box, the four fallback branches and the "mask lifted no depth" check, and
calling `object_in_base` directly would silently skip all of them.

Frames. `rgb.png`, `depth_aligned.npy` AND `mask.png` are stored upright —
`capture.py:58` rotates the first two on a half turn and `capture.py:95`
rotates the mask by the same rule — while `depth_raw` and `T_base_cam` stay
raw. The loop works in RAW orientation throughout, so all three are
un-rotated here. Feeding the disk mask straight in would deproject an
upside-down cup against a right-way-up pose.

Two things stand between "same inputs" and "same floats":

  the workspace crop — `object_in_base` used to crop the MASKED points to
      WORKSPACE too, and the note that removed it (`geometry.py:333`) cites
      run 2408-cup4 as its evidence. The whole change reached git in one
      commit, e0976e4 at 16:30 on 2026-08-24, AFTER every cup run, so the
      runs must date themselves — and their own recorded counts do it
      cleanly: cup3 (15:34) and cup4 (16:03) reproduce only WITH the crop,
      cup5 (16:24) only without. Uncropped, cup4 view 010 offers 1667 points
      where the run recorded 202; cropped, cup5 view 001 offers 2733 where
      the run recorded 4298. `auto` reads that off the counts, `--crop
      on|off` overrides it.

  the table plane — `fit_table` (geometry.py:156) is RANSAC with no seed,
      so the plane moves by 1-3 mm of height and up to ~1 deg of tilt
      between calls, and `above_table` then keeps a slightly different set
      of points. Nothing on disk records the plane the run used. What
      run.json DOES record is the count that plane produced and the cloud it
      fused into (`machine.py:436` writes "+N pts, fused M"), so by default
      the replay re-draws the plane until both agree — that pins the one
      input the recorder dropped. `--no-pin` shows the honest spread
      instead: on these runs it lands 4-11 mm and a few hundred points away.

Which makes the replay a stochastic search, and open3d's RANSAC is
OpenMP-parallel so seeding does not make it repeatable. `--attempts`
re-runs the whole thing until the final cloud is exact. That is not fitting
to the answer: `fused_cloud.npy` is never an input, only the verdict.

    p inspection/investigation/step_replay.py --run 2408-cup3
    p inspection/investigation/step_replay.py --run 2408-cup5 --no-pin
    p inspection/investigation/step_replay.py --run 2408-cup4 --crop on

Everything it writes goes to /tmp/step_replay/<run>/ — step_NNN.ply/.npy
(NNN is the capture dir the step fused), the three renders, summary.json.
"""
import argparse
import json
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import inspection.cell.geometry as geom
from inspection.cell.geometry import (ABOVE_TABLE_M, CloudAccumulator,
                                      above_table, cam_to_base, crop_workspace,
                                      deproject, fit_table, rotate180)
from inspection.run.segmenter import Segmentation, object_view

RUNS = Path(__file__).resolve().parents[1] / "data" / "runs"
OUT = Path("/tmp/step_replay")

#: `machine.py:436` — "+{npts} pts, fused {len(acc.points)}".
_TURN = re.compile(r"\+(\d+) pts, fused (\d+)$")


# --------------------------------------------------------------- disk -> loop
def recorded(run_dir: Path) -> dict:
    """{pose_id: (offered, fused)} for every turn that actually fused a view.

    `machine.py:609` numbers a capture with `_next_cell_step()`, which counts
    prior CELL turns only (survey is always 000), so run.json's turn list maps
    back onto the NNN dirs without guessing. A turn that never reached a
    capture still consumes its number — `_record_turn` runs for a software
    stop too — which is exactly why 2408-cup5 has no 029/ on disk.
    """
    turns = json.loads((run_dir / "run.json").read_text())["turns"]
    out, cells = {}, 0
    for t in turns:
        if t["target"] == "survey":
            step = 0
        else:
            cells += 1
            step = cells
        m = _TURN.match(t["result"])
        if m:
            out[step] = (int(m.group(1)), int(m.group(2)))
    return out


def captures(run_dir: Path):
    """(pose_id, cap, stored_mask) per capture, in capture order, RAW frame.

    Capture order IS sorted dir order: the id is a monotone counter over
    turns (`machine.py:685`), never reused, so sorting the digits sorts the
    run in time.
    """
    import cv2
    for d in sorted(p for p in run_dir.iterdir()
                    if p.is_dir() and p.name.isdigit()):
        meta = json.loads((d / "meta.json").read_text())
        rot = int(meta.get("rgb_rotation_deg", 0))
        rgb = cv2.cvtColor(cv2.imread(str(d / "rgb.png")), cv2.COLOR_BGR2RGB)
        depth_aligned = np.load(d / "depth_aligned.npy")
        mask = None
        if (d / "mask.png").exists() and "mask" in meta:
            mask = cv2.imread(str(d / "mask.png"), cv2.IMREAD_UNCHANGED) > 0
        if rot == 180:
            rgb, depth_aligned = rotate180(rgb), rotate180(depth_aligned)
            mask = None if mask is None else rotate180(mask)
        cap = {"dir": str(d), "rgb": rgb,
               "depth_raw": np.load(d / "depth_raw.npy"),
               "depth_aligned": depth_aligned,
               "T_base_cam": np.array(meta["T_base_cam"], dtype=float)}
        yield int(d.name), cap, (None if mask is None else
                                 (mask, meta["mask"]["score"],
                                  tuple(meta["mask"]["box"])))


class StoredMask:
    """Replays the mask the run fused instead of calling SAM 3 again.

    `object_view` asks a segmenter for exactly one thing, `mask_for`, and
    `mask.png` is the array that produced the fused points — `machine.py:624`
    saves it from the same `seg` object it is about to fuse, before
    `acc.add`. Returning it keeps every other branch of the shipped policy
    live while making the replay deterministic, which re-running the model
    would not be. `None` reproduces a miss, and `object_view` then takes the
    depth fallback with `seed=acc.points` just as the run did.
    """

    __slots__ = ("stored", "calls", "misses")

    def __init__(self, stored):
        self.stored, self.calls, self.misses = stored, 0, 0

    def mask_for(self, rgb, box, T_base_cam=None):
        self.calls += 1
        if rgb is None or box is None or self.stored is None:
            self.misses += 1
            return None
        mask, score, stored_box = self.stored
        # The STORED box, not the freshly computed one: it is what the model
        # was actually prompted with. It cannot move a point — the mask is
        # the only thing `object_in_base` reads — but it keeps `seg.box`
        # honest for anything that draws it.
        return Segmentation(np.asarray(mask, dtype=bool), stored_box,
                            float(score))


@contextmanager
def legacy_workspace_crop():
    """Put the workspace crop back on the masked branch, for pre-fix runs.

    Patching `cam_to_base` rather than copying `object_in_base` keeps every
    other line of the shipped function in the replay. The scene branch
    (`geometry.py:324`) already crops and `crop_workspace` is idempotent and
    order-preserving, so it is untouched; only the masked branch
    (`geometry.py:345`) gains its crop back.
    """
    orig = geom.cam_to_base
    geom.cam_to_base = lambda p, T: crop_workspace(orig(p, T))
    try:
        yield
    finally:
        geom.cam_to_base = orig


def detect_crop(run_dir, intr_color, scale, rec, probes=5, draws=14) -> str:
    """"on"/"off": which masked branch reproduces this run's recorded counts.

    Deliberately built out of the geometry primitives rather than out of
    `object_view` — it is choosing a MODE, not producing a point, and it has
    to score both branches under a plane that moves. For each of the first
    few masked views it draws `draws` table planes and asks which branch's
    range of counts brackets the number run.json recorded.
    """
    on = off = 0
    for pose_id, cap, stored in captures(run_dir):
        if stored is None or pose_id not in rec or probes <= 0:
            continue
        probes -= 1
        want = rec[pose_id][0]
        da, T = cap["depth_aligned"], cap["T_base_cam"]
        scene = crop_workspace(
            cam_to_base(deproject(da, intr_color, scale), T))
        raw = cam_to_base(
            deproject(np.where(stored[0], da, 0), intr_color, scale), T)
        cropped = crop_workspace(raw)
        n_off, n_on = [], []
        for _ in range(draws):
            plane = fit_table(scene)
            n_off.append(len(above_table(raw, plane, ABOVE_TABLE_M)))
            n_on.append(len(above_table(cropped, plane, ABOVE_TABLE_M)))
        off += min(n_off) <= want <= max(n_off)
        on += min(n_on) <= want <= max(n_on)
    return "on" if on > off else "off"


# ------------------------------------------------------------------- replay
@contextmanager
def _tap():
    """Watch what `object_in_base` hands its two primitives, changing neither.

    The plane search below needs the arrays that live INSIDE the shipped
    function — the scene the plane is fitted on, and the masked points the
    plane is thresholding. Observing the call rather than recomputing it is
    the difference between searching over the loop and searching over a
    lookalike: whatever the crop mode, the arrays recorded here are the ones
    the run's own points came from.
    """
    rec = {"fit_in": [], "above_in": []}
    ft, am = geom.fit_table, geom._above_mask
    geom.fit_table = lambda pts: (rec["fit_in"].append(pts), ft(pts))[1]
    # `object_in_base` thresholds through `_above_mask` (the boolean core
    # that lets colours ride along) — `above_table` is only the public
    # wrapper now, so the tap moves to where the loop actually calls.
    geom._above_mask = lambda pts, plane, margin=ABOVE_TABLE_M: (
        rec["above_in"].append((pts, plane)), am(pts, plane, margin))[1]
    try:
        yield rec
    finally:
        geom.fit_table, geom._above_mask = ft, am


@contextmanager
def _pinned_plane(plane, at_call):
    """Hand `object_in_base` a chosen plane on its `at_call`-th fit."""
    ft, n = geom.fit_table, [0]

    def fit(points):
        n[0] += 1
        return np.asarray(plane, dtype=float) if n[0] - 1 == at_call \
            else ft(points)

    geom.fit_table = fit
    try:
        yield
    finally:
        geom.fit_table = ft


def pinned_view(cap, intr, intr_color, scale, acc, segmenter, want, tries,
                cands=64):
    """`object_view` with the table plane pinned to what run.json recorded.

    The one input the recorder dropped is the RANSAC plane (`fit_table` is
    unseeded, geometry.py:156). The two things it kept about it are the
    number of points that cleared it and the size of the cloud they fused
    into (`machine.py:436` writes both), and this searches real `fit_table`
    draws until BOTH agree. The second number is what makes the first one
    trustworthy: equal counts through a distance threshold usually means
    equal sets, but not always, and on 2408-cup5 view 021 a plane matching
    only the offered count put two extra points into the cloud.

    Drawn, not constructed: only planes RANSAC can actually return are
    admissible, otherwise the replay would be reproducing a recorded number
    with a plane the loop could never have had. Some views need a lot of
    draws — cup5 view 018 lands on its recorded 2666 about twice in 40 000 —
    which is why the search runs on the two arrays `_tap` caught rather than
    on a full `object_view` call.
    """
    import copy
    want_n, want_fused = (want if want else (None, None))

    def fused_n(view):
        """Cloud size this view would leave behind, on a throwaway copy."""
        trial = copy.deepcopy(acc)
        if len(view["points"]):
            trial.add(view["points"])
        return len(trial.points)

    def fused_ok(view):
        return want_fused is None or fused_n(view) == want_fused

    def call(plane=None, at_call=0):
        fell = []
        ctx = _pinned_plane(plane, at_call) if plane is not None else _null()
        with ctx:
            v, s = object_view(cap, intr, intr_color, scale, acc.points,
                               segmenter, on_fallback=fell.append)
        return v, s, fell

    fell = []
    with _tap() as rec:
        view, seg = object_view(cap, intr, intr_color, scale, acc.points,
                                segmenter, on_fallback=fell.append)
    if want_n is None or (len(view["points"]) == want_n and fused_ok(view)):
        return view, seg, 1, fell

    best, best_seg, best_fell = view, seg, fell
    if seg is not None and len(rec["fit_in"]) >= 2:
        # Masked path: the last fit is the one on `depth_aligned` and the last
        # `above_table` is the mask's own, so the count can be scored on the
        # cached arrays alone — only a candidate that passes gets a full call.
        scene, m_pts = rec["fit_in"][-1], rec["above_in"][-1][0]
        at = len(rec["fit_in"]) - 1
        left, miss = cands, None
        for i in range(tries):
            plane = geom.fit_table(scene)
            if len(geom.above_table(m_pts, plane)) != want_n:
                continue
            view, seg, fell = call(plane, at)
            n = fused_n(view)
            if n == want_fused:
                return view, seg, i + 1, fell
            if miss is None or abs(n - want_fused) < miss:
                miss = abs(n - want_fused)
                best, best_seg, best_fell = view, seg, fell
            left -= 1
            if left <= 0:
                break
        return best, best_seg, tries, best_fell
    # Depth fallback: the plane sits upstream of growth/clustering, so there
    # is nothing cheap to score — re-draw the whole view.
    for i in range(2, max(2, min(tries, 200)) + 1):
        view, seg, fell = call()
        if len(view["points"]) == want_n and fused_ok(view):
            return view, seg, i, fell
        near = abs(len(view["points"]) - want_n)
        if near < abs(len(best["points"]) - want_n):
            best, best_seg, best_fell = view, seg, fell
    return best, best_seg, min(tries, 200), best_fell


def replay(args):
    run_dir = RUNS / args.run
    session = json.loads((run_dir / "session.json").read_text())
    intr = session["intrinsics"]["ir_left"]
    intr_color = session["intrinsics"]["color"]
    scale = session["depth_scale_m_per_unit"]
    rec = recorded(run_dir)

    crop = args.crop
    if crop == "auto":
        # Detected once and pinned onto args: a retry must not re-decide it.
        crop = args.crop = detect_crop(run_dir, intr_color, scale, rec)
        print(f"workspace crop on masked points: {crop} (detected)")
    else:
        print(f"workspace crop on masked points: {crop}")

    out = Path(args.out) / args.run
    out.mkdir(parents=True, exist_ok=True)

    acc = CloudAccumulator()
    steps, fallbacks, unfused = [], [], []
    print(f"{'k':>3} {'view':>4} {'src':>5} {'offered':>8} {'rec':>8} "
          f"{'draws':>6} {'drop':>5} {'cloud':>6} {'rec':>6} "
          f"{'extent mm':>22}")
    ctx = legacy_workspace_crop() if crop == "on" else _null()
    with ctx:
        k = 0
        for pose_id, cap, stored in captures(run_dir):
            if pose_id not in rec:
                unfused.append(pose_id)
                print(f"    {pose_id:>4}  on disk but no fused turn — skipped")
                continue
            k += 1
            want, want_fused = rec[pose_id]
            view, seg, tries, fell = pinned_view(
                cap, intr, intr_color, scale, acc, StoredMask(stored),
                None if args.no_pin else rec[pose_id], args.tries)
            pts = view["points"]
            dropped = acc.add(pts, view.get("colors")) if len(pts) else 0
            P = acc.points
            mn, mx = acc.aabb()
            np.save(out / f"step_{pose_id:03d}.npy", P)
            _write_ply(out / f"step_{pose_id:03d}.ply", P, acc.colors)
            for why in fell:
                fallbacks.append((pose_id, why))
            steps.append({"k": k, "pose_id": pose_id,
                          "source": "mask" if seg is not None else "depth",
                          "offered": int(len(pts)), "offered_recorded": want,
                          "tries": tries, "dropped": int(dropped),
                          "cloud": int(len(P)), "cloud_recorded": want_fused,
                          "extent_mm": [round(float(v) * 1000, 1)
                                        for v in (mx - mn)],
                          "box_delta_px": _box_delta(view, seg)})
            flag = "" if len(pts) == want else "  <<< offered mismatch"
            flag += "" if len(P) == want_fused else "  <<< cloud mismatch"
            print(f"{k:>3} {pose_id:>4} "
                  f"{('mask' if seg is not None else 'depth'):>5} "
                  f"{len(pts):>8} {want:>8} {tries:>6} {dropped:>5} "
                  f"{len(P):>6} {want_fused:>6} "
                  f"{str(np.round((mx - mn) * 1000, 1)):>22}{flag}")
            for why in fell:
                print(f"       ! {why}")
    # The colour backfill artifact: row-aligned with the replayed cloud,
    # which `compare` proves is the stored fused_cloud.npy. Written HERE,
    # never into the run dir — recorded runs are read-only.
    if len(acc.points):
        np.save(out / "fused_colors.npy", acc.colors)
        _write_ply(out / "fused_colored.ply", acc.points, acc.colors)
    return acc.points, steps, fallbacks, unfused, out, crop


@contextmanager
def _null():
    yield


def _box_delta(view, seg):
    """Max corner distance between the recomputed prompt box and the stored
    one — a free check that the seed cloud is tracking the live one."""
    box, stored = view.get("prompt_box"), None if seg is None else seg.box
    if box is None or stored is None:
        return None
    return int(max(abs(a - b) for a, b in zip(box, stored)))


def _write_ply(path, points, colors=None):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(
            np.asarray(colors, dtype=np.float64) / 255.0)
    o3d.io.write_point_cloud(str(path), pcd)


# ------------------------------------------------------------- verification
def compare(stored, replayed, verbose=True):
    """Stored fused_cloud.npy vs the replay, in numbers not adjectives."""
    say = print if verbose else (lambda *a, **k: None)
    r = {"n_stored": int(len(stored)), "n_replayed": int(len(replayed))}
    say(f"\nstored {len(stored)} pts   replayed {len(replayed)} pts")
    if len(stored) == len(replayed):
        a = stored[np.lexsort(stored.T[::-1])]
        b = replayed[np.lexsort(replayed.T[::-1])]
        d = float(np.abs(a - b).max())
        r["max_abs_coord_diff_m"] = d
        r["verdict"] = "exact" if d == 0.0 else "tolerance"
        say(f"counts match; max |coord diff| after lexsort = {d:.3e} m")
        if d == 0.0:
            say("EXACT — identical floats")
            return r
    from scipy.spatial import cKDTree
    da = cKDTree(replayed).query(stored)[0]
    db = cKDTree(stored).query(replayed)[0]
    r.update({"nn_stored_to_replayed": {"max": float(da.max()),
                                        "mean": float(da.mean())},
              "nn_replayed_to_stored": {"max": float(db.max()),
                                        "mean": float(db.mean())}})
    say(f"nn stored->replayed  max {da.max() * 1000:9.4f} mm   "
        f"mean {da.mean() * 1000:9.4f} mm")
    say(f"nn replayed->stored  max {db.max() * 1000:9.4f} mm   "
        f"mean {db.mean() * 1000:9.4f} mm")
    r["unmatched"] = {}
    for tol in (1e-6, 1e-4, 1e-3):
        u = (int((da > tol).sum()), int((db > tol).sum()))
        r["unmatched"][f"{tol:g}"] = u
        say(f"  unmatched at {tol:g} m: {u[0]} of {len(stored)} stored, "
            f"{u[1]} of {len(replayed)} replayed")
    r.setdefault("verdict", "mismatch")
    return r


# ------------------------------------------------------------------ renders
def _limits(P, pad=0.01):
    mn, mx = P.min(0) - pad, P.max(0) + pad
    c, half = (mn + mx) / 2, float((mx - mn).max()) / 2
    return c - half, c + half


def _axes(ax, lo, hi, elev=22, azim=-63, bare=False):
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    if bare:
        # The strip's whole point is that the camera and the extent never
        # move, so nine copies of the same ticks are noise.
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_ticklabels([])
        ax.tick_params(length=0)
        return
    ax.tick_params(labelsize=6)
    ax.set_xlabel("x", fontsize=7); ax.set_ylabel("y", fontsize=7)
    ax.set_zlabel("z", fontsize=7)


def render_strip(out, steps, final, n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = [steps[i] for i in
            np.unique(np.linspace(0, len(steps) - 1, n).round().astype(int))]
    lo, hi = _limits(final)
    cols = min(4, len(pick))
    rows = int(np.ceil(len(pick) / cols))
    fig = plt.figure(figsize=(3.1 * cols, 3.1 * rows))
    for i, s in enumerate(pick):
        P = np.load(out / f"step_{s['pose_id']:03d}.npy")
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=0.7, c="#1f77b4",
                   linewidths=0, alpha=0.75)
        _axes(ax, lo, hi, bare=True)
        ax.set_title(f"step {s['k']} — view {s['pose_id']:03d} — {len(P)} pts",
                     fontsize=9, pad=-2)
    box = np.round((final.max(0) - final.min(0)) * 1000, 0)
    fig.suptitle(f"object cloud after each fused view — fixed camera, "
                 f"fixed {box} mm extent", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.01,
                        hspace=0.08, wspace=0.0)
    p = out / "growth_strip.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


def render_overlay(out, stored, replayed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = _limits(np.vstack([stored, replayed]))
    fig = plt.figure(figsize=(11, 5.2))
    for i, (elev, azim) in enumerate(((22, -63), (78, -90))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(stored[:, 0], stored[:, 1], stored[:, 2], s=6,
                   c="#1f77b4", linewidths=0, alpha=0.55,
                   label=f"stored fused_cloud ({len(stored)})")
        ax.scatter(replayed[:, 0], replayed[:, 1], replayed[:, 2], s=1.6,
                   c="#d62728", linewidths=0, alpha=0.9,
                   label=f"replayed ({len(replayed)})")
        _axes(ax, lo, hi, elev, azim)
        ax.set_title("oblique" if i == 0 else "top-down", fontsize=9)
        if i == 0:
            ax.legend(loc="upper left", fontsize=8, markerscale=3)
    fig.suptitle("stored vs replayed final cloud", fontsize=11)
    fig.tight_layout()
    p = out / "final_overlay.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


def render_curve(out, steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = [s["k"] for s in steps]
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                               height_ratios=(2, 1))
    a.plot(k, [s["cloud_recorded"] for s in steps], "-", lw=4, alpha=0.35,
           color="#1f77b4", label="recorded (run.json)")
    a.plot(k, [s["cloud"] for s in steps], "-o", ms=3, lw=1.2,
           color="#d62728", label="replayed")
    a.set_ylabel("fused cloud size [pts]")
    a.grid(alpha=0.3); a.legend(fontsize=9)
    a.set_title("cloud size after each fused view")
    b.bar([x - 0.2 for x in k], [s["offered_recorded"] for s in steps],
          width=0.4, color="#1f77b4", alpha=0.6, label="offered (recorded)")
    b.bar([x + 0.2 for x in k], [s["offered"] for s in steps],
          width=0.4, color="#d62728", alpha=0.8, label="offered (replayed)")
    b.set_xlabel("step"); b.set_ylabel("offered [pts]")
    b.grid(alpha=0.3, axis="y"); b.legend(fontsize=8)
    fig.tight_layout()
    p = out / "cloud_size.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--crop", choices=("auto", "on", "off"), default="auto",
                    help="workspace crop on the MASKED points — the pre-"
                         "2026-08-24 branch (geometry.py:333)")
    ap.add_argument("--no-pin", action="store_true",
                    help="do not re-draw the table plane to the count "
                         "run.json recorded; shows the raw RANSAC spread")
    ap.add_argument("--tries", type=int, default=100000,
                    help="plane re-draw budget per view when pinning")
    ap.add_argument("--seed", type=int, default=0,
                    help="open3d global RNG seed")
    ap.add_argument("--attempts", type=int, default=4,
                    help="repeat the whole replay until the final cloud is "
                         "exact; the search over planes is stochastic")
    ap.add_argument("--no-renders", action="store_true")
    args = ap.parse_args()

    import open3d as o3d
    o3d.utility.random.seed(args.seed)

    stored = np.load(RUNS / args.run / "fused_cloud.npy")
    # The plane search is a search, and open3d's RANSAC is OpenMP-parallel —
    # `utility.random.seed` does not make it repeatable, so two attempts do
    # not draw the same planes. A view can therefore end on a plane that
    # matches both recorded numbers yet differs in a point or two, and starve
    # a LATER view of any reachable draw. Nothing is being fitted to the
    # answer by repeating: every attempt is judged against fused_cloud.npy,
    # which the replay never reads as an input.
    for attempt in range(1, max(1, args.attempts) + 1):
        P, steps, fallbacks, unfused, out, crop = replay(args)
        ver = compare(stored, P, verbose=False)
        print(f"attempt {attempt}/{args.attempts}: {ver['verdict']}")
        if ver["verdict"] == "exact" or args.no_pin:
            break
    ver = compare(stored, P)

    bad = [s for s in steps if s["offered"] != s["offered_recorded"]]
    print(f"\n{len(steps)} views fused, {len(bad)} with an offered count the "
          f"replay could not reach")
    for s in bad:
        print(f"  view {s['pose_id']:03d}: replayed {s['offered']} vs "
              f"recorded {s['offered_recorded']}")
    print(f"fallback (depth-path) views: "
          f"{[f'{p:03d}' for p, _ in fallbacks] or 'none'}")
    print(f"capture dirs never fused: "
          f"{[f'{p:03d}' for p in unfused] or 'none'}")
    boxes = [s["box_delta_px"] for s in steps if s["box_delta_px"] is not None]
    if boxes:
        print(f"prompt box vs stored box: max corner delta {max(boxes)} px")

    renders = {}
    if not args.no_renders:
        renders = {"growth_strip": str(render_strip(out, steps, P)),
                   "final_overlay": str(render_overlay(out, stored, P)),
                   "cloud_size": str(render_curve(out, steps))}
        for k, v in renders.items():
            print(f"render {k}: {v}")

    (out / "summary.json").write_text(json.dumps(
        {"run": args.run, "crop": crop, "pinned": not args.no_pin,
         "seed": args.seed, "steps": steps, "verification": ver,
         "fallbacks": fallbacks, "unfused_dirs": unfused,
         "renders": renders}, indent=2) + "\n")
    print(f"summary: {out / 'summary.json'}")
    return 0 if ver.get("verdict") == "exact" else 1


if __name__ == "__main__":
    sys.exit(main())
