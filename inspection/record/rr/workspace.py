#!/usr/bin/env python3
"""Build the review workspace: one `.rrd` per recorded run.

Reads the derived tree for a run — the step-k clouds `step_replay.py`
wrote to inspection/data/derived/<run>/steps/ and every primitive
decomposition the fit bank wrote to .../fits/<method>/step_NNN.json — plus
what the run itself recorded under inspection/data/runs/<run>/ (poses,
intrinsics, rgb, mask, chain renders, the stored final cloud), and hands
all of it to `rr/log.py`, which owns the schema. Nothing here calls Rerun
directly.

    p inspection/record/rr/workspace.py --run 2408-cup3
    p inspection/record/rr/workspace.py --all

Writes inspection/data/derived/<run>/rrd/<run>.rrd — durable, gitignored,
regeneratable, next to the inputs it was built from. Open it with:

    p inspection/record/rr/view.py show 2408-cup3

The run directories are read-only artifacts. The only thing this script
writes outside the run's own `rrd/` directory is the handful of rgb frames
it has to re-rotate (see `_frustum_image`), and those are scratch, so they
go to /tmp/rrd/_img/.

Methods are swept, never listed: any directory under `fits/` holding
`step_NNN.json` becomes one channel, coloured deterministically by name.
Adding a fitter to the bank therefore adds a channel here with no edit —
that is the whole promise of the closed-kind contract.

The six stories the file has to answer, and where each is served:

  1 open a run          `rr/log.py:blueprint`, baked in by `save`
  2 scrub the scan      the `step` timeline: cloud, fresh glow, frusta
  3 review a fit        `_fit_channels` -> `rr/log.py:log_primitives`
  4 see what it saw     `_chain_slots` -> the 2x2 image grid
  5 trust the replay    `rr/log.py:log_overlay`, static, hidden by default
  6 receive evidence    the file itself
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from inspection.record.rr import log as rr_log

DATA = Path(__file__).resolve().parents[2] / "data"
RUNS = DATA / "runs"
DERIVED = DATA / "derived"
#: Where step_replay used to write before the derived tree existed. Kept as
#: a fallback so a machine that still has /tmp populated (and no derived
#: steps yet) keeps building.
REPLAY = Path("/tmp/step_replay")
#: Rotated rgb frames are scratch, not derived data: they are an artefact of
#: how `capture.py` stored the png, reproducible in a second, and nothing
#: downstream reads them. They stay in /tmp.
IMG_CACHE = Path("/tmp/rrd/_img")

#: The accumulator voxelises at 3 mm and re-centroids the whole cloud on
#: every `add` (`geometry.py:CloudAccumulator.add`), so the points of step
#: k-1 do not survive into step k as the same floats. "Fresh" therefore has
#: to mean "further from the previous cloud than the quantisation can move
#: a point", not "not present before". 0.6 voxel is comfortably above the
#: centroid jitter and comfortably below the spacing of genuinely new
#: surface.
FRESH_TOL_M = 0.0018

#: The chain, in the order the loop walks it: what the camera saw, what the
#: segmenter was prompted with, what came back, what survived the table and
#: the jump gate. Four cells, so the panel is a square 2x2 (Anton,
#: 2026-09-03) rather than a column of letterboxed strips.
CHAIN_FULL = {"rgb": "rgb.png", "prompt": "chain_prompt.png",
              "mask": "chain_mask.png", "kept": "chain_kept.png"}
#: The spec's fallback for runs that predate the chain renders — e.g.
#: 2408-cup3. Same grid, two cells filled.
CHAIN_FALLBACK = {"rgb": "rgb.png", "mask": "mask.png"}


# ------------------------------------------------------------------- inputs
def _replay_dir(run: str) -> Path:
    """The step-k clouds: the durable derived tree first, /tmp second.

    Both directories carry the same `summary.json` and the same clouds
    (verified point-for-point). The derived one wins because it is what the
    fit JSONs were computed from, so a fit's `inliers` index the very array
    logged next to it — and because /tmp does not survive a reboot.
    """
    for d in (DERIVED / run / "steps", REPLAY / run):
        if (d / "summary.json").exists():
            return d
    raise SystemExit(f"no step clouds for {run} in {DERIVED / run / 'steps'} "
                     f"or {REPLAY / run} — run "
                     f"inspection/investigation/step_replay.py --run {run}")


def _step_cloud(rep: Path, pose_id: int) -> tuple[np.ndarray, np.ndarray | None]:
    """One step's (points, colors|None), from whichever encoding is there.

    Points prefer the npy (the exact array the fits indexed). Colours only
    exist in the step PLY — replays before the colour capture (2026-09-03)
    wrote colourless PLYs, so `None` here is ordinary, not an error."""
    import open3d as o3d
    npy = rep / f"step_{pose_id:03d}.npy"
    ply = rep / f"step_{pose_id:03d}.ply"
    if not npy.exists() and not ply.exists():
        raise SystemExit(f"missing step cloud {npy} / {ply}")
    pcd = o3d.io.read_point_cloud(str(ply)) if ply.exists() else None
    pts = np.load(npy) if npy.exists() else np.asarray(pcd.points)
    colors = None
    if pcd is not None and pcd.has_colors() and len(pcd.points) == len(pts):
        colors = np.clip(np.rint(np.asarray(pcd.colors) * 255.0),
                         0, 255).astype(np.uint8)
    return pts, colors


def _fit_channels(run: str) -> dict[str, dict[int, Path]]:
    """Sweep `fits/*/` — {method: {pose_id: decomposition json}}.

    A directory counts as a method iff it holds `step_NNN.json`, which is
    what keeps `fits/renders/` (evidence pngs) and `fits/scores.jsonl` out
    of the entity tree without this file having to know either exists. The
    NNN is the POSE id, matching `steps/step_NNN.ply`, not the slider
    index: 2408-cup5 skips view 29 and both trees skip it together.
    """
    root = DERIVED / run / "fits"
    out: dict[str, dict[int, Path]] = {}
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files = {int(f.stem.split("_")[1]): f
                 for f in sorted(d.glob("step_*.json"))}
        if files:
            out[d.name] = files
    return out


def _chain_slots(run_dir: Path) -> dict[str, str]:
    """Which image set this run's chain panel shows.

    Decided once for the run, not per view, so the panel does not gain and
    lose rows mid-scrub; a run either has the chain renders or it does not.
    """
    if all((run_dir / "000" / f).exists() for f in CHAIN_FULL.values()):
        return dict(CHAIN_FULL)
    return dict(CHAIN_FALLBACK)


def _frustum_image(view_dir: Path, cache: Path) -> Path:
    """The rgb in the frame `T_base_cam` is expressed in.

    `capture.py` stores `rgb.png` UPRIGHT — rotated by 180 when the wrist
    was — while the pose stays raw (`step_replay.py` module docstring). The
    frustum is placed by the pose, so a run's half-turn views need the
    stored png turned back, or the cup on the image plane hangs upside
    down over a right-way-up cloud. Rotated copies are cached in /tmp;
    unrotated views reference the original file and cost nothing.
    """
    meta = json.loads((view_dir / "meta.json").read_text())
    src = view_dir / "rgb.png"
    if int(meta.get("rgb_rotation_deg", 0)) != 180:
        return src
    import cv2
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / f"{view_dir.name}_rgb_raw.png"
    if not dst.exists():
        cv2.imwrite(str(dst), np.rot90(cv2.imread(str(src)), 2).copy())
    return dst


def _table_plane(run_dir: Path, session: dict) -> tuple[np.ndarray, int]:
    """Re-fit the table from view 000, the way the loop fits it.

    Nothing on disk records the plane the run used — `fit_table` is
    unseeded RANSAC (`geometry.py:156`) and only its consequences were
    written down. So this is a re-draw, not a recovery, and it is logged
    labelled as such. It is here at all because a cloud floating in an
    empty 3D view gives the eye nothing to judge "is this cup sitting on
    the table" against.
    """
    from inspection.cell.geometry import (cam_to_base, crop_workspace,
                                          deproject, fit_table, rotate180)
    d = run_dir / "000"
    meta = json.loads((d / "meta.json").read_text())
    da = np.load(d / "depth_aligned.npy")
    if int(meta.get("rgb_rotation_deg", 0)) == 180:
        da = rotate180(da)
    scene = crop_workspace(cam_to_base(
        deproject(da, session["intrinsics"]["color"],
                  session["depth_scale_m_per_unit"]),
        np.array(meta["T_base_cam"], dtype=float)))
    return fit_table(scene), len(scene)


def _fresh(cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """The part of the step-k cloud that step k-1 did not already cover."""
    if prev is None or len(prev) == 0 or len(cur) == 0:
        return cur
    from scipy.spatial import cKDTree
    return cur[cKDTree(prev).query(cur)[0] > FRESH_TOL_M]


# -------------------------------------------------------------------- build
def build(run: str, out_dir: Path | None = None, overlay: bool = False) -> Path:
    """Write <derived>/<run>/rrd/<run>.rrd.

    `overlay=True` ships the stored-vs-replayed channel visible by default;
    the fit channels are visible either way, because they are what this
    workspace is for.
    """
    run_dir = RUNS / run
    out_dir = Path(out_dir) if out_dir else DERIVED / run / "rrd"
    rep = _replay_dir(run)
    summary = json.loads((rep / "summary.json").read_text())
    session = json.loads((run_dir / "session.json").read_text())
    intr = session["intrinsics"]["color"]
    slots = _chain_slots(run_dir)
    cache = IMG_CACHE / run
    fits = _fit_channels(run)

    # Dash, not slash: 0.37 treats the application id as an entry name and
    # silently migrates anything with a `/` in it to a hashed name, so the
    # viewer's title bar would read "review-2408-cup38a8a".
    app_id = f"review-{run}"
    rr_log.begin(app_id)
    rr_log.log_world()

    plane, n_scene = _table_plane(run_dir, session)
    print(f"{run}: table plane re-fit from view 000 on {n_scene} scene pts "
          f"-> {np.round(plane, 4).tolist()}")
    for m, files in fits.items():
        # Declared before the loop so an all-abstain method still has a
        # channel in the tree — see rr_log.open_method_channel.
        rr_log.open_method_channel(m)
        print(f"{run}: fit channel {m!r} — {len(files)} decompositions, "
              f"colour {rr_log.method_color(m)}")

    prev = None
    final = None
    live: dict[str, set[str]] = {m: set() for m in fits}
    drawn: dict[str, int] = {m: 0 for m in fits}
    for k, s in enumerate(summary["steps"]):
        pid = s["pose_id"]
        view_dir = run_dir / f"{pid:03d}"
        cloud, cloud_rgb = _step_cloud(rep, pid)
        rr_log.at(k)
        rr_log.log_cloud(cloud, cloud_rgb)
        fresh = _fresh(cloud, prev)
        rr_log.log_fresh(fresh)
        meta = json.loads((view_dir / "meta.json").read_text())
        rr_log.log_camera(pid, np.array(meta["T_base_cam"], dtype=float),
                          intr, _frustum_image(view_dir, cache))
        counts = _log_fits(fits, pid, live)
        for m, n in counts.items():
            drawn[m] += n
        rr_log.log_chain({name: view_dir / f
                          for name, f in slots.items()
                          if (view_dir / f).exists()})
        prev, final = cloud, cloud
        print(f"  step {k:>3}  view {pid:03d}  cloud {len(cloud):>5}  "
              f"fresh {len(fresh):>5}  src {s['source']}  "
              f"fits {' '.join(f'{m}:{n}' for m, n in counts.items()) or '-'}")

    if k == 0:
        raise SystemExit(f"{run}: no steps in {rep}/summary.json")
    # The centre of the slab: under the cloud, not at the base origin.
    rr_log.log_table(plane, [float(final[:, 0].mean()),
                             float(final[:, 1].mean()), 0.0])
    rr_log.log_overlay(np.load(run_dir / "fused_cloud.npy"), final)

    # CLOUD_RGB starts with its eye closed everywhere: the neutral cloud is
    # the working backdrop, the captured colours are the one-click toggle.
    bp = rr_log.blueprint(run, list(slots),
                          hidden=(rr_log.CLOUD_RGB,) if overlay
                          else (rr_log.OVERLAY, rr_log.CLOUD_RGB))
    path = rr_log.save(out_dir / f"{run}.rrd", bp, app_id)
    v = summary["verification"]
    print(f"{run}: {k + 1} steps, chain slots {list(slots)}, fits "
          f"{ {m: drawn[m] for m in fits} }, replay verdict "
          f"{v.get('verdict')} ({v.get('n_stored')} vs {v.get('n_replayed')} "
          f"pts) -> {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def _log_fits(fits: dict[str, dict[int, Path]], pose_id: int,
              live: dict[str, set[str]]) -> dict[str, int]:
    """Every method's decomposition for this step. Returns {method: n drawn}.

    Two things happen here that are easy to get wrong and invisible when
    you do.

    RETIREMENT. Rerun is latest-at, so an entity logged once is on screen
    for every later step. The cyl bank addresses its cylinders by
    geometric content (`body`, `cyl_e+00a030`, ...), so the label set
    CHANGES step to step and a plain re-log would accumulate every
    cylinder the method ever fitted into one bouquet. Each step therefore
    clears the paths that were live last step and are not live now.

    ABSTENTION. `planar` refuses on all 30 steps of 2408-cup3 and on 29 of
    30 in cup5; `cyl` refuses on one step of cup3. A refusal is a first
    class answer (contract.py), so it is logged as one: the geometry is
    retired and the reason lands on the method entity where the selection
    panel shows it, rather than the previous step's shape silently
    standing in for it.
    """
    counts: dict[str, int] = {}
    for m, files in fits.items():
        f = files.get(pose_id)
        prims, meta = [], {}
        if f is not None:
            d = json.loads(f.read_text())
            prims = [dict(p, method=m) for p in d.get("primitives", [])]
            meta = {"version": d.get("version", ""),
                    "abstain": bool(d.get("abstain", False)),
                    "reason": d.get("reason", ""),
                    "coverage": float(d.get("coverage", 0.0)),
                    "residual_mm": float(d.get("residual_mm", 0.0)),
                    "n_primitives": int(d.get("n_primitives", len(prims)))}
        else:
            meta = {"version": "", "abstain": True,
                    "reason": "no decomposition on disk for this step",
                    "coverage": 0.0, "residual_mm": 0.0, "n_primitives": 0}
        now = set(rr_log.log_primitives(prims))
        rr_log.clear_primitives(live[m] - now)
        live[m] = now
        rr_log.log_decomposition_meta(m, **meta)
        counts[m] = len(prims)
    return counts


def _all_runs() -> list[str]:
    """Every run with step clouds on disk, derived tree first."""
    seen = []
    for root, sub in ((DERIVED, "steps"), (REPLAY, "")):
        if not root.is_dir():
            continue
        for p in sorted(root.iterdir()):
            if (p / sub / "summary.json").exists() and p.name not in seen:
                seen.append(p.name)
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="e.g. 2408-cup3")
    ap.add_argument("--all", action="store_true",
                    help="every run with step clouds on disk")
    ap.add_argument("--out", default=None,
                    help="override the output DIRECTORY (default: "
                         "inspection/data/derived/<run>/rrd)")
    ap.add_argument("--overlay-on", action="store_true",
                    help="ship the stored/replayed overlay VISIBLE (the "
                         "spec's default is off)")
    a = ap.parse_args()
    runs = _all_runs() if a.all else [a.run]
    if not runs or runs == [None]:
        ap.error("--run or --all")
    for r in runs:
        build(r, a.out, overlay=a.overlay_on)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
