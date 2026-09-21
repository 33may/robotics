"""Ray views (design: docs/2026-09-17-ray-views-design.md).

A view is a ray, not a point: distance is never computed by a formula —
it is the closest sample on the ray that is provably safe (sensor range,
IK, collision), found offline in the kinematic sandbox.
"""
from dataclasses import dataclass
from collections import namedtuple

import numpy as np

from inspection.view.faceview import (face_cam_pose, face_corners, face_frame,
                                      top_align_roll)
from inspection.view.grid import H_BINS, R_MAX, V_ELEVATIONS
from inspection.view.viewsphere import ROLLS

#: One knob for every ray discretisation: depth grid AND face shift grid.
RAY_STEP_M = 0.02
#: D405 minimum depth range at 720p (datasheet + camera.py note).
MIN_RANGE_M = 0.105
#: Containment margin [px]: the principal point is NOT at the frame centre
#: (D405 ppy≈356.7 vs 360 — measured -2 px corner cuts on the bench) plus
#: numeric slack. Anton 2026-09-17: "check the whole face is actually in
#: the frame" — containment must hold with the real sensor, not the ideal.
FOV_MARGIN_PX = 14.0


@dataclass(frozen=True)
class Ray:
    """Anchor point and unit direction, base frame."""
    anchor: tuple
    direction: tuple

    def __post_init__(self):
        d = np.asarray(self.direction, dtype=float)
        n = float(np.linalg.norm(d))
        if not np.isfinite(n) or n < 1e-9:
            raise ValueError("ray direction must be a nonzero vector")
        object.__setattr__(self, "direction", tuple(d / n))
        object.__setattr__(self, "anchor",
                           tuple(float(v) for v in self.anchor))


Sample = namedtuple("Sample", ["depth", "roll", "q", "T_cam", "T_flange"])


def sample_ray(world, ik, t_flange_cam, ray, clear_m=0.0, q_near=None,
               rolls=None):
    """All provably-safe samples on `ray` AT ONE ROLL, ascending depth: the
    roll whose closest valid sample is the closest of all (ties broken by
    `rolls` order, which is what "first roll wins" used to decide).

    `rolls` is the candidate walk — ROLLS when None. A face member may carry
    its own pair (the top face's yaw-aligned pair, Anton 2026-09-21); the
    two candidates must stay pi apart so the flip keeps whatever alignment
    the base roll establishes.

    `clear_m` = camera-to-object-surface distance already consumed at depth
    0 (0 for a face ray — its anchor sits ON the wall; the anchor→surface
    run for a centroid ray). The sensor floor is surface distance, so a
    sample is admissible from depth clear_m + MIN_RANGE_M.

    PER-ROLL LISTS (ruled amendment 2026-09-17, task 2b). This used to keep
    ONE sample per depth — the first roll that solved there — which made the
    list's roll change partway out. A slide is roll-locked (a wrist flip
    belongs in the hop, in open space), so such a list has to be cut at the
    first flip: on cell {3,1} of the bench box that left a ONE-sample
    corridor, and the hop then delivered the view pose laterally — the
    arrival mode hop-then-slide exists to prevent. Both rolls are now
    candidates at every depth under the same floors, and ONE roll's whole run
    is returned. The ENDPOINT is unchanged by construction: element 0 is
    still the closest valid sample on the ray.

    Cost: both rolls are checked only until the winner is DECIDED — the first
    valid (depth, roll) pair, scanning depths ascending and rolls in ROLLS
    order, settles it, and from there out only the winning roll is evaluated.
    The loser's remaining depths would be computed just to be discarded.

    NOT necessarily contiguous in depth. A depth the winning roll cannot
    solve is SKIPPED, not terminated on, so the returned list can carry a
    gap — an obstacle the ray grazes at one depth and clears again further
    out. Callers that need a contiguous corridor must build it themselves:
    `transaction._slide` does, by validating each segment and stopping at
    the first one the world refuses, which is what makes a slide a slide
    rather than a list of poses that happen to be safe.
    """
    t_fc_inv = np.linalg.inv(np.asarray(t_flange_cam, dtype=float))
    if rolls is None:
        rolls = ROLLS
    lo = clear_m + MIN_RANGE_M
    d0 = float(np.ceil(lo / RAY_STEP_M)) * RAY_STEP_M
    winner, out = None, []
    for d in np.arange(d0, R_MAX + 1e-9, RAY_STEP_M):
        for roll in (rolls if winner is None else (winner,)):
            T_cam = face_cam_pose(ray.anchor, ray.direction, float(d), roll)
            T_fl = T_cam @ t_fc_inv
            free = [q for q in ik.branches(T_fl, q_near=q_near)
                    if not world.is_colliding(q)]
            if free:
                winner = roll             # the first valid pair decides
                out.append(Sample(float(d), float(roll), free[0], T_cam, T_fl))
                break                     # the other roll at this depth is
                                          # the loser's list, never ours
    return out


CONE_STEP_DEG, CONE_MAX_DEG, CONE_AZIMUTHS = 5.0, 60.0, 12


def cell_ray(center, h, v):
    """Ray of viewsphere cell {h, v}: centroid anchor, grid direction —
    the same az0 convention as ViewSphere (h=0 faces the robot base)."""
    center = np.asarray(center, dtype=float)
    az0 = np.arctan2(-center[1], -center[0])
    az = az0 + h * (2 * np.pi / H_BINS)
    el = np.deg2rad(V_ELEVATIONS[v])
    return Ray(tuple(center),
               (np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                np.sin(el)))


def cell_clear_m(aabb, ray):
    """Anchor→surface run along the ray (slab method) — feeds sample_ray's
    clear_m so the sensor floor measures SURFACE distance, not anchor
    distance."""
    mn, mx = (np.asarray(v, dtype=float) for v in aabb)
    o, d = np.asarray(ray.anchor), np.asarray(ray.direction)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (mn - o) / d, (mx - o) / d
    t_exit = float(np.nanmin(np.maximum(t1, t2)))
    return max(t_exit, 0.0)


def cell_d_contain(aabb, ray, fy, height_px, corners=None):
    """Containment floor for a cell ray: the closest depth from which EVERY
    belief-AABB corner stays inside the frame (Anton 2026-09-17, extending
    the face ruling — a cell view's promise is the whole object).

    Exact per corner: a corner offset `o` from the anchor sits at camera
    depth d - o·dir with lateral spans |o·û|, |o·v̂| (both ⊥ dir, so
    d-independent); solving f*lat/(d - o·dir) <= half_px - margin per
    corner and axis gives d >= o·dir + lat*f/(half_px - margin); the floor
    is the max over corners and both image axes. Width reuses fy (D405 is
    16:9 with fx≈fy). This is the projected-support bound — it tracks the
    bearing, so flat objects keep their close cells instead of paying a
    bounding-sphere tax. `corners` (8x3, the belief BOX corners) tightens
    the bound over the axis-aligned aabb — a yawed box's AABB inflates up
    to ~20%/dim and that tax is centimetres of standoff."""
    mn, mx = (np.asarray(v, dtype=float) for v in aabb)
    o, d = np.asarray(ray.anchor), np.asarray(ray.direction)
    x = -d                                        # boresight, cam pose rule
    ref = np.array([0.0, 0.0, 1.0])
    if abs(x @ ref) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, x); u /= np.linalg.norm(u)  # image left
    v_img = np.cross(x, u)                        # image up
    half_v = height_px / 2.0 - FOV_MARGIN_PX
    half_u = (height_px * 16.0 / 9.0) / 2.0 - FOV_MARGIN_PX
    if corners is None:
        corners = np.array([[cx, cy, cz] for cx in (mn[0], mx[0])
                            for cy in (mn[1], mx[1])
                            for cz in (mn[2], mx[2])])
    d_min = 0.0
    for c in np.asarray(corners, dtype=float):
        off = c - o
        fwd = float(off @ d)
        d_min = max(d_min,
                    fwd + abs(float(off @ v_img)) * fy / half_v,
                    fwd + abs(float(off @ u)) * fy / half_u)
    return d_min


def box_corners(prim):
    """The 8 corners of a fitted box primitive, base frame (8x3).

    The ORIENTED box, never its AABB: a yawed box's axis-aligned hull
    inflates up to ~20%/dim and `cell_d_contain` would charge that
    inflation as standoff. ONE copy, here — the live machine and the
    offline replay bench (`investigation/ray_views/ray_replay.py`) must
    bound the same object with the same eight points.
    """
    he = np.asarray(prim.params["extents"], dtype=float) / 2.0
    sgn = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1)
                    for c in (-1, 1)], dtype=float)
    return prim.center + (sgn * he) @ prim.axes.T


def belief_bounds(prim, fallback=None):
    """`(aabb, corners)` for the belief a cell ray is measured against.

    ONE reduction, three callers: `machine._cell_map` (the live map),
    `collect._default_planner` (the sweep's ranking) and the replay bench.
    They were three verified-identical copies, which is a coincidence with
    an expiry date — a cell's resolved depth must not depend on WHICH of
    them asked.

    With a boxguard `prim`, both come from that one fit: the corners are
    the ORIENTED box's (`box_corners`) and the aabb is their hull —
    UNPADDED. `geometry.object_box`'s hull is the COLLISION box (2 cm of
    margin per side, 5 cm floor) and feeding those margins here would push
    every cell's `clear_m`, and so its resolved depth, outward from what
    the bench measured. With no fit yet, `fallback` (that padded hull) is
    the only bound there is and is used as the conservative stand-in, with
    `corners=None` — `cell_d_contain` then derives the eight axis-aligned
    corners itself.
    """
    if prim is None:
        return fallback, None
    corners = box_corners(prim)
    return (corners.min(0), corners.max(0)), corners


def _d_contain(fy, height_px, half_span):
    """Closest depth at which a vertical half-span [m] stays inside the
    frame — the inverse of shift_max(d) = d*(height_px/2-margin)/fy -
    half_span. FOV containment is a per-depth constraint (controller
    ruling, task-2-report.md): it cannot cap a shift at a fixed MIN_RANGE_M
    depth, it becomes each family member's own MINIMUM admissible depth.
    FOV_MARGIN_PX keeps the promise on the real sensor (asymmetric
    principal point) instead of the ideal one.

    Since 2026-09-21 this is the ANALYTIC REFERENCE (tests pin against it);
    the face floors themselves come from `_contain_depth` — walked and
    checked, not bounded."""
    return fy * half_span / (height_px / 2.0 - FOV_MARGIN_PX)


def _contain_depth(corners, anchor, direction, roll, fy, height_px):
    """Closest RAY_STEP_M grid depth whose frame contains every corner at
    this roll, or None within R_MAX — containment found the tunnel-slide
    way (Anton, 2026-09-21: "use the same tunnel slide idea to find the
    closest orthogonal"): each candidate depth is CHECKED against the real
    projection of the face's own corners, never bounded by an extent
    formula. The half-diagonal bound this replaces stood ~10 cm too far
    off an elongated box's top face. Containment is monotone in depth
    (backing straight off shrinks the projection), so the first containing
    grid depth is the floor, and the grid is `sample_ray`'s own, so the
    floor names an actually samplable depth. FOV_MARGIN_PX absorbs the
    real sensor's asymmetric principal point, as everywhere."""
    corners = np.asarray(corners, dtype=float)
    width_px = height_px * 16.0 / 9.0
    d0 = float(np.ceil(MIN_RANGE_M / RAY_STEP_M)) * RAY_STEP_M
    for d in np.arange(d0, R_MAX + 1e-9, RAY_STEP_M):
        T = face_cam_pose(anchor, direction, float(d), roll)
        pc = (corners - T[:3, 3]) @ np.column_stack(
            [-T[:3, 1], -T[:3, 2], T[:3, 0]])
        if np.any(pc[:, 2] <= 1e-9):
            continue
        u = fy * pc[:, 0] / pc[:, 2] + width_px / 2.0
        v = fy * pc[:, 1] / pc[:, 2] + height_px / 2.0
        if (u.min() >= FOV_MARGIN_PX and u.max() <= width_px - FOV_MARGIN_PX
                and v.min() >= FOV_MARGIN_PX
                and v.max() <= height_px - FOV_MARGIN_PX):
            return float(d)
    return None


def face_family(prim, face, fy, height_px):
    """Ordered (ray, shift, theta_deg, d_min, rolls) candidates for one
    boxguard face: parallel +z shifts ascending (fronto-parallel — a plane
    parallel to the sensor projects as a pure similarity, zero keystone),
    then the classic cone. `d_min` is the closest depth this member may be
    sampled from — MIN_RANGE_M for cone entries (tilted views never promised
    containment, they are already the degraded rung), and
    max(MIN_RANGE_M, containment depth) for shift/top entries, so a taller
    shift only pushes its own floor out rather than being rejected outright.
    `rolls` is the member's camera-roll walk for `sample_ray` — ROLLS,
    except the top face's exact member, which carries the yaw-aligned pair.
    First family member with samples (at its own d_min floor) wins; no
    member -> face inaccessible, honestly."""
    center, normal, (ext_a, ext_b) = face_frame(prim, face)
    corners = face_corners(prim, face)
    out = []
    if abs(normal[2]) < 0.5:                      # side faces shift up
        # d_min <= R_MAX is the real cap now; equivalently
        # s <= R_MAX*(height_px/2-margin)/fy - ext_b/2 (same shape as the
        # old cap, R_MAX in place of MIN_RANGE_M — enumerate while a depth
        # exists). Both ROLLS see the same containment (a 180-deg image
        # rotation under symmetric margins), so roll 0 checks for the pair.
        cap = R_MAX * (height_px / 2.0 - FOV_MARGIN_PX) / fy - ext_b / 2.0
        s = 0.0
        while s <= cap + 1e-9:
            anchor = center + np.array([0.0, 0.0, s])
            d_min = _contain_depth(corners, anchor, normal, 0.0,
                                   fy, height_px)
            if d_min is None:
                break                 # a taller shift only sees less
            out.append((Ray(tuple(anchor), tuple(normal)), float(s), 0.0,
                        float(d_min), tuple(ROLLS)))
            s += RAY_STEP_M
    else:                                         # top face: exact only
        # The camera ROLLS WITH THE BOX (Anton 2026-09-21, demo7: the yawed
        # box came back diagonal in the top capture): the exact member
        # carries the yaw-aligned roll pair, wider footprint side along the
        # image width axis — alignment holds mod pi, so the 180-deg
        # collision flip keeps it. The floor is the walked corner check at
        # that aligned roll, so the elongated box's top camera stands where
        # the ROTATED RECTANGLE fits, not where its diagonal would
        # (~10 cm closer on demo7's 2.5:1 box). None = no depth within
        # R_MAX contains the face: no exact member, the cone still offers.
        r0 = top_align_roll(prim)
        d_min = _contain_depth(corners, center, normal, r0, fy, height_px)
        if d_min is not None:
            out.append((Ray(tuple(center), tuple(normal)), 0.0, 0.0,
                        float(d_min), (r0, r0 + np.pi)))
    a = np.array([0.0, 0.0, 1.0])
    if abs(normal @ a) > 0.95:
        a = np.array([1.0, 0.0, 0.0])
    u = np.cross(a, normal); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    th = CONE_STEP_DEG
    while th <= CONE_MAX_DEG:
        t = np.deg2rad(th)
        for k in range(CONE_AZIMUTHS):
            phi = 2 * np.pi * k / CONE_AZIMUTHS
            d = (np.cos(t) * normal
                 + np.sin(t) * (np.cos(phi) * u + np.sin(phi) * v))
            if d[2] < 0.05:               # never approach from under table
                continue
            out.append((Ray(tuple(center), tuple(d)), 0.0, float(th),
                        MIN_RANGE_M, tuple(ROLLS)))
        th += CONE_STEP_DEG
    return out


RayResolution = namedtuple("RayResolution",
                           ["T_cam", "T_flange", "q", "depth", "shift",
                            "theta_deg", "roll", "samples"])


def _resolution(samples, shift, theta):
    s = samples[0]
    return RayResolution(s.T_cam, s.T_flange, s.q, s.depth, shift, theta,
                         s.roll, samples)


def resolve_cell(world, ik, t_flange_cam, center, aabb, h, v,
                 fy, height_px, q_near=None, corners=None):
    """Closest safe sample on cell {h,v}'s ray that keeps the WHOLE belief
    AABB in frame, or None (inaccessible). The containment floor reaches
    sample_ray through clear_m exactly as face members do:
    floor = clear_m + MIN_RANGE_M, so clear_m carries whichever is larger —
    the real surface run (sensor floor) or the containment depth."""
    ray = cell_ray(center, h, v)
    clear = max(cell_clear_m(aabb, ray),
                cell_d_contain(aabb, ray, fy, height_px, corners)
                - MIN_RANGE_M)
    samples = sample_ray(world, ik, t_flange_cam, ray,
                         clear_m=clear, q_near=q_near)
    return _resolution(samples, 0.0, 0.0) if samples else None


def cell_map(world, ik, t_flange_cam, center, aabb, fy, height_px,
             corners=None, q_near=None):
    """Every viewsphere cell resolved: `{(h, v): RayResolution | None}`.

    The WHOLE grid, always — an inaccessible cell is an explicit `None`,
    not a missing key. That dict IS the machine's accessibility gate
    (`machine.py:_reach`, read as `.get(cell) is None`), and a key that was
    never asked must not be indistinguishable from a ray that was asked and
    had nothing provably safe on it.

    A thin loop over `resolve_cell`; the belief `corners` are computed once
    by the caller and shared by every cell (they describe the OBJECT, not
    the view). Cell order matches `ViewSphere.cells()`. Measured ~140-150 ms
    for the 36-cell grid on the bench box — below the design's ~230 ms
    estimate and below Stage A's own ~175 ms, because per-roll sampling
    (task 2b) pays both rolls only until the winning roll is decided and
    then stops probing the loser, where Stage A re-tried roll 0 at every
    depth of a roll-pi ray. This is the budget the machine pays once per
    plan and once per fuse.
    """
    return {(h, v): resolve_cell(world, ik, t_flange_cam, center, aabb, h, v,
                                 fy, height_px, q_near=q_near,
                                 corners=corners)
            for v in range(len(V_ELEVATIONS)) for h in range(H_BINS)}


def resolve_face(world, ik, t_flange_cam, prim, face, fy, height_px,
                 q_near=None):
    """Closest safe sample in the face's family, fronto-parallel first.

    Two rungs, and the SHIFT rung is searched whole before the cone is
    consulted at all:

      1. every theta=0 member (the +z shift ladder; a single exact member
         for the top face) is sampled, and the winner is the member whose
         closest valid sample has the SMALLEST DEPTH — ties to the smaller
         shift.
      2. only if no theta=0 member has a sample: the cone members in family
         order, first hit wins (unchanged).

    AMENDED by ruling 2026-09-17 (bench amendment,
    investigation/ray_views/ray_replay.py + task-4-report.md). This used to
    return the FIRST shift with any valid sample, which is not the closest
    one: on the 1709 demo box `x+` resolved at shift 0 / depth 0.30 while
    shift 0.04 held a sample at 0.14 — 16 cm of standoff paid for ladder
    predictability, and the same pattern in all four benched runs (12-20 cm).
    A shift is optically free (a plane parallel to the sensor projects as a
    pure similarity at any shift, so no keystone is traded away) and the
    containment floor already keeps the face in frame, so closeness wins.
    Cost of scanning the whole rung instead of stopping early: ~10-27 ms
    measured. Tilt stays strictly later — a cone member is never preferred
    to a fronto-parallel one, however close it is.

    None = the face is honestly inaccessible. Each member's d_min
    (containment floor, per the earlier 2026-09-17 ruling) reaches
    sample_ray through clear_m: floor = clear_m + MIN_RANGE_M.
    """
    family = face_family(prim, face, fy, height_px)

    def _sample(ray, d_min, rolls):
        return sample_ray(world, ik, t_flange_cam, ray,
                          clear_m=max(0.0, d_min - MIN_RANGE_M),
                          q_near=q_near, rolls=rolls)

    best = None                                   # rung 1: whole shift rung
    for ray, shift, theta, d_min, rolls in family:
        if theta != 0.0:
            continue
        samples = _sample(ray, d_min, rolls)
        if not samples:
            continue
        key = (samples[0].depth, shift)
        if best is None or key < best[0]:
            best = (key, samples, shift, theta)
    if best is not None:
        return _resolution(best[1], best[2], best[3])

    for ray, shift, theta, d_min, rolls in family:  # rung 2: cone, first hit
        if theta == 0.0:
            continue
        samples = _sample(ray, d_min, rolls)
        if samples:
            return _resolution(samples, shift, theta)
    return None
