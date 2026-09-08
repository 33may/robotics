"""The settle pipeline: one capture -> fused into the object cloud, or refused.

Extracted from the capture leg of `Supervisor._exec_worker` (`run/machine.py`)
so the offline replay and unit tests can drive the exact object-identity +
plane-gate + fuse logic the robot runs, without a live Supervisor thread.
`machine.py`'s module docstring explains why this leg is one of the two
places allowed to mutate run state off the dispatcher thread (it still is —
`settle_capture` calls `acc.add` directly); nothing about that contract moves
here. `settle_capture` itself does no publishing, no disk writes, no
`_recenter()`, and touches no `self.events` — those stay in the machine,
which also keeps the try/except that turns a raised exception into a
"capture failed: {e}" `settle_done` (this module lets exceptions propagate).
"""
from dataclasses import dataclass

import numpy as np

from inspection.cell.geometry import JUMP_GATE_M, WORKSPACE
from inspection.run.segmenter import object_view

#: Per-view sanity gate on the fitted table plane. The cell's table is probed
#: ground truth at z=0 in the base frame; a view whose RANSAC plane lands far
#: from that was placed with a wrong camera pose — whatever the cause (run
#: 2108-ui: a silently frozen RTDE q stamped five views with the survey pose;
#: their planes came out at z 90/-490/242 mm, tilted up to 48 deg). Rejecting
#: on the plane catches ANY pose error, not just the one failure mode we have
#: already met. Tolerances leave room for real arm flex (4-13 deg of tilt was
#: historic for the SO-101; the UR5e fits within ~2 deg and ~10 mm).
PLANE_Z_TOL_M = 0.03
PLANE_TILT_TOL_DEG = 5.0


def _plane_error(plane) -> str | None:
    """Why the fitted table plane contradicts the probed cell, or None if ok.

    `plane` is `fit_table`'s [a,b,c,d] with the unit normal forced +z. Height
    is evaluated at the workspace centre — the plane's d alone would measure
    height at the base origin, outside the crop, where a small tilt reads as
    a large offset.
    """
    a, b, c, d = (float(x) for x in plane)
    norm = float(np.linalg.norm((a, b, c)))
    if norm < 1e-9 or c <= 0:
        return f"degenerate table plane {plane!r}"
    tilt = float(np.degrees(np.arccos(np.clip(c / norm, -1.0, 1.0))))
    xc = (WORKSPACE["x"][0] + WORKSPACE["x"][1]) / 2
    yc = (WORKSPACE["y"][0] + WORKSPACE["y"][1]) / 2
    z_center = -(a * xc + b * yc + d) / c
    if abs(z_center) > PLANE_Z_TOL_M:
        return (f"table plane at z={z_center * 1000:.0f} mm "
                f"(> {PLANE_Z_TOL_M * 1000:.0f} mm) — camera pose is wrong")
    if tilt > PLANE_TILT_TOL_DEG:
        return (f"table plane tilted {tilt:.1f} deg "
                f"(> {PLANE_TILT_TOL_DEG:.0f} deg) — camera pose is wrong")
    return None


@dataclass
class SettleResult:
    ok: bool
    detail: str            # "" on success; rejection/failure reason otherwise
    view: dict | None      # object_view's dict (points, plane, centroid, ...)
    seg: object | None     # segmenter result or None
    npts: int
    dropped: int
    #: Why the mask path was abandoned, verbatim from `object_view`, or None
    #: when it was never tried (no segmenter) or never left. `seg is None`
    #: alone cannot tell those apart, and the record has a field for exactly
    #: this question (schema.py:GeometryStats.fallback_reason).
    fallback: str | None = None


def settle_capture(cap, rig, segmenter, acc, is_survey: bool,
                   on_warn=lambda msg: None,
                   floor_z: float | None = None) -> SettleResult:
    """One capture -> object identity, plane gate, and fuse into `acc`.

    Moved verbatim (same order) from `_exec_worker`'s capture leg:
    `object_view` (was `Supervisor._object_from_capture`) -> the plane gate
    (reject BEFORE anything from this view is kept) -> the survey-with-no-
    centroid rejection -> `acc.add` with its detached-points warning routed
    through `on_warn` rather than a hardwired publisher, so this function has
    no dependency on the Supervisor or a live UI.
    """
    fell_back = None

    def _fallback(why):
        nonlocal fell_back
        fell_back = why
        on_warn(why)

    view, seg = object_view(cap, rig.intr, rig.intr_color, rig.depth_scale,
                            acc.points, segmenter, on_fallback=_fallback,
                            floor_z=floor_z)
    # The plane gate sits BEFORE anything from this view is kept: a rejected
    # view must not touch the accumulator, publish a capture, or mark the
    # cell visited.
    bad = _plane_error(view["plane"])
    if bad:
        return SettleResult(ok=False, detail=f"view rejected: {bad}",
                            view=None, seg=None, npts=0, dropped=0,
                            fallback=fell_back)
    npts = len(view["points"]) if view["points"] is not None else 0
    if is_survey and view["centroid"] is None:
        return SettleResult(ok=False, detail="NO OBJECT above the table",
                            view=None, seg=None, npts=0, dropped=0,
                            fallback=fell_back)
    dropped = acc.add(view["points"], view.get("colors")) if npts else 0
    if dropped:
        # Loud on purpose: a detached blob is a scene problem (a cable,
        # a second object drifting in), not a routine filter event.
        on_warn(f"rejected {dropped} detached points "
                f"(>{JUMP_GATE_M*100:.0f} cm from the object)")
    return SettleResult(ok=True, detail="", view=view, seg=seg,
                        npts=npts, dropped=dropped, fallback=fell_back)
