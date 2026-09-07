#!/usr/bin/env python3
"""The verb surface — what an agent may ask of a run.

Three things meet here: `view/grid.py` (how cells are named), `Run`
(record/run.py — what has actually been seen) and the capture dirs (the
pixels). Every return carries BOTH pixels and text, because which one a VLM
reads better is an empirical question the design leaves to measurement.

The text is deterministic provenance built from geometry — never a model's
words. Nothing in this module can move the robot: it reads captured views
and, at most, records that a view it wanted does not exist.

Port note (task-5, flows-design §4 "consumers ported"): this used to run
over `eyes/store.py`'s `RunStore`, which held views written by
`eyes/replay.py` (deleted). `RunStore` survives only for the brain's OTHER
role — mutable plan/hypothesis/finding state — so `writer=` here still binds
to it (a `PlanWriter`/`FindingWriter`), but views now come straight from
`Run`. Semantic mapping (binding, from the plan): old `RunStore` views were
`{cell: None|tuple, pose_id, cap_dir, T_base_cam, t}`; the `ViewRecord`
below adapts a `Step` to the same shape: `cell` is `None` for step 0 (the
survey) else its recorded grid address; `pose_id` is `step.id`; `cap_dir` is
`step.dir` — a FULL Path now, so the old `run_dir / cap_dir` join sites
below read `rec.cap_dir` directly; `t` is `step.record.t_captured`.
"""
from dataclasses import dataclass, replace

import numpy as np

from inspection.cell.geometry import (UPRIGHT_TOL_DEG, image_tilt_deg,
                                     rotate180)
from inspection.record.run import Run
from inspection.view.grid import (DEFAULT_R, H_BINS, V_ELEVATIONS,
                                  coverage_map, neighbors)


def upright(rgb, T_base_cam, stored_rotation_deg=None):
    """Pixels the right way up, plus a provenance note. `(rgb, note)`.

    `stored_rotation_deg` is `meta.json`'s `rgb_rotation_deg` — how much the
    file on disk was ALREADY rotated when it was written. Since 2026-08-24 the
    capture writer stores the colour frame upright, so this is normally 180 for
    a half-turn cell and nothing is left to do here. Runs captured before that
    have no such key; `None` means "raw on disk" and the correction happens
    now, from the pose. Without this distinction a new run would be flipped
    twice and land back upside down.

    Only the exact half-turn is ever corrected: a 180 deg flip is lossless and
    keeps the frame landscape, so a corrected view is still 848x480 and
    directly comparable. Anything else is REPORTED, not rotated — the
    hand-taught survey pose is not on the viewsphere (measured -16 deg on run
    2408-seeded) and straightening it would mean interpolating away the
    corners, while hiding that the taught pose is crooked.
    """
    tilt = image_tilt_deg(T_base_cam)
    if stored_rotation_deg:
        return rgb, " · upright on disk"
    if abs(abs(tilt) - 180.0) <= UPRIGHT_TOL_DEG:
        return rotate180(rgb), " · flipped upright"
    if abs(tilt) > UPRIGHT_TOL_DEG:
        return rgb, f" · TILTED {tilt:+.0f} deg, not corrected"
    return rgb, ""


@dataclass(frozen=True)
class ViewImage:
    """Pixels plus their provenance. `box` is set when this is a crop."""
    cell: tuple | None
    cap_dir: str
    rgb: np.ndarray                 # (H, W, 3) uint8, RGB order
    text: str
    box: tuple | None = None


class ViewRecord:
    """Adapts a `Step` to the old RunStore view-row shape (module docstring:
    semantic mapping). `.step` rides along for fields the old row never
    carried, e.g. `rgb_rotation_deg` (`get_view` below)."""

    def __init__(self, step):
        self.step = step
        self.cell = None if step.id == 0 else tuple(step.record.view.address)
        self.pose_id = step.id
        self.cap_dir = step.dir
        self.t = step.record.t_captured
        self.T_base_cam = step.T_base_cam


def _grid_of(run: Run):
    """(h_bins, v_elevs, r) for this run's viewsphere.

    Native runs carry them on the `viewsphere` view_method's params
    (`record/writer.py` callers). Legacy `run.json` never did — the deleted
    `eyes/replay.py` handed every legacy run the same defaults, preserved
    here (view/grid.py: H_BINS, V_ELEVATIONS); its `r` alone survives the
    legacy adapter (`record/legacy.py:adapt_run`), as `view_methods[0].params`.
    """
    r = None
    for vm in run.record.view_methods:
        r = vm.params.get("r") or r
        if "h_bins" in vm.params and "v_elevs" in vm.params:
            return vm.params["h_bins"], tuple(vm.params["v_elevs"]), r or DEFAULT_R
    return H_BINS, V_ELEVATIONS, r or DEFAULT_R


class ViewTools:
    """Verbs over one run. `writer` (optional) receives miss notes."""

    def __init__(self, run: Run, writer=None):
        self._run = run
        self._writer = writer
        self._h_bins, self._v_elevs, self._r = _grid_of(run)

    # ----------------------------------------------------------- geometry
    def _views(self):
        """Every captured view, survey included — mirrors the old
        `RunStore.views()` with no cell filter (used by `run()`'s opening
        survey pick and the survey/vlm agents' own survey lookup)."""
        return [ViewRecord(s) for s in self._run.captured]

    def view_at(self, cell):
        """Newest view of a cell, or None. A revisit supersedes the earlier
        look.

        `Run.at()` matches on the EXACT recorded address — the survey's is
        None, so it can never satisfy any cell tuple, (0, 0) included. It
        must stay that way: the survey is off-grid (loop.py's prompt says
        so explicitly), and a coalesced default would silently hand it back
        for a real, merely-uncaptured cell (task-5 review, 2026-09-07).
        """
        step = self._run.at(tuple(cell))
        return None if step is None else ViewRecord(step)

    def views_near(self, cell):
        """[(neighbour, ViewRecord | None)] — 4-connected, h wraps, v clamps.

        An uncaptured neighbour is not an error: it is a next-move hint that
        cost no motion, so it is recorded as a note when a writer is bound.
        """
        out = []
        for c in neighbors(tuple(cell), self._h_bins, len(self._v_elevs)):
            rec = self.view_at(c)
            if rec is None and self._writer is not None:
                self._writer.note(
                    f"wanted neighbour {list(c)} of {list(cell)} "
                    f"— not captured", cell=c)
            out.append((c, rec))
        return out

    def coverage(self, cur=None):
        """ASCII map of what has been seen, plus the count and a legend."""
        visited = {tuple(s.record.view.address) for s in self._run.captured
                  if s.id != 0}
        cov = np.zeros((len(self._v_elevs), self._h_bins), dtype=bool)
        for h, v in visited:
            cov[v, h] = True
        n_cells = cov.shape[0] * cov.shape[1]
        return (f"{int(cov.sum())}/{n_cells} cells seen  "
                f"(# seen · . unseen · @ current)\n"
                + coverage_map(cov, cur=cur, elevations=self._v_elevs))

    # ------------------------------------------------------------- pixels
    def get_view(self, target):
        """Load a view's RGB. `target` is a cell or a ViewRecord."""
        rec = self.view_at(tuple(target)) if isinstance(target, (tuple, list)) \
            else target
        if rec is None:
            raise KeyError(f"no captured view at cell {target}")
        path = rec.cap_dir / "rgb.png"
        import cv2                       # local: keeps pixel-free callers cheap
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        # capture.py:44 writes through RGB2BGR — invert it, or every image the
        # VLM sees has its red and blue channels swapped.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Orientation is corrected HERE and nowhere earlier: the geometry path
        # (`cell.geometry.object_in_base`) deprojects the raw depth against
        # this same T_base_cam, so flipping pixels upstream would silently
        # mirror the point cloud. Raw on disk stays raw. `rgb_rotation_deg`
        # comes straight off the step record now (always an int — the legacy
        # adapter synthesizes 0 when a run predates the field), no more
        # separate meta.json probe.
        rgb, note = upright(rgb, rec.T_base_cam, rec.step.record.rgb_rotation_deg)
        return ViewImage(rec.cell, rec.cap_dir.name, rgb, self._gloss(rec) + note)

    def crop(self, img: ViewImage, box):
        """Sub-image; box (x0, y0, x1, y1) is clipped to the frame."""
        h, w = img.rgb.shape[:2]
        x0, y0, x1, y1 = box
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(x1)), min(h, int(y1))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"empty crop {box} of {w}x{h}")
        return replace(img, rgb=img.rgb[y0:y1, x0:x1], box=(x0, y0, x1, y1),
                       text=f"{img.text} · crop {x0},{y0}-{x1},{y1} "
                            f"({x1 - x0}x{y1 - y0} px)")

    # -------------------------------------------------------------- notes
    def note(self, text, cell=None):
        if self._writer is None:
            raise RuntimeError("no writer bound — this tier may not write")
        self._writer.note(text, cell=cell)

    # ----------------------------------------------------------- survey
    def survey_bearing(self, rec, centre=None):
        """Where the survey pose sits on the azimuth ring: `(az_deg, h_float)`.

        The hand-taught survey is off-grid (measured -16 deg on 2408-seeded),
        so the planner cannot look its azimuth up — but it needs one to anchor
        the survey declaration's frame-relative words to addresses. This is
        deterministic geometry, code-computed, never model-authored.

        The centre is the object reconstruction, not a guess (Anton
        2026-08-27): live callers pass the Supervisor's `sup.sphere.center`,
        which the survey's own recon seeded; in replay the same recon is on
        disk as `fused_cloud.npy` and its centroid is used. Only the camera
        POSITION enters — no boresight, so no axis-convention risk.
        MEASURED (2026-08-27, 2608-aicam): the cloud centroid reproduces the
        six on-grid cells' own azimuths to ≤4.3°, mean 1.9°. No cloud and no
        centre -> None; a missing bearing must not stop a run.

        h = 0 faces the base: the reference direction is -centre_xy
        (view/viewsphere.py:68-69).
        """
        if centre is None:
            fused = self._run.fused()
            if fused is None:
                return None
            cloud = fused[0]
            if cloud.size == 0:
                return None
            centre = cloud[:, :3].mean(axis=0)
        centre = np.asarray(centre, float)
        ref = np.arctan2(-centre[1], -centre[0])
        p = rec.T_base_cam[:3, 3] - centre
        az = float(np.degrees(np.arctan2(p[1], p[0]) - ref)) % 360.0
        return az, az / (360.0 / self._h_bins)

    # ------------------------------------------------------------ helpers
    def _gloss(self, rec):
        """Deterministic provenance for a view — geometry, never opinion."""
        if rec.cell is None:
            return f"survey view · dir {rec.cap_dir.name}"
        h, v = rec.cell
        return (f"cell [{h}, {v}] · azimuth {h * 360 / self._h_bins:.0f} deg "
                f"(h{h} of {self._h_bins}) · elevation {self._v_elevs[v]:.0f} deg "
                f"· r {self._r:.2f} m · dir {rec.cap_dir.name}")


if __name__ == "__main__":
    import sys

    t = ViewTools(Run.load(sys.argv[1]))
    print(t.coverage())
    visited = sorted({tuple(s.record.view.address)
                      for s in t._run.captured if s.id != 0})
    for cell in visited[:3]:
        print(" ", t._gloss(t.view_at(cell)))
