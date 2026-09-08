#!/usr/bin/env python3
"""Scene geometry for the inspection loop — v1 sandbox.

Depth makes the geometry; detection only labels it (project principle).
Pipeline per view: deproject raw depth (IR-left frame) -> transform to base
frame via FK + hand-eye -> drop table plane -> object points.

Which above-table points ARE the object depends on whether identity is
already established:

  first view (no seed) — largest above-table cluster. The bootstrap case
      only; the "one cluster -> take it" rule.
  later views (seeded)  — points connected to the cloud so far, by chained
      steps of at most GROW_EPS_M (`grow_from_seed`).

Re-electing the largest cluster every view is what broke run 2408-geomtest:
the black mug lost its depth at an oblique angle (D405 is passive stereo —
no projector, so a dark specular surface gives nothing to match), so a cable
out-pointed it 568 to 392 and got fused 280 mm away. The object box grew to
327x392x200 mm, swallowed the arm's own pose, and every plan after that died
at "RRTConnect: start tree could not be initialized". Identity is a region of
space, not a per-view popularity contest.

All point clouds are (N, 3) float64 in the robot base frame, meters.
"""

from pathlib import Path

import numpy as np

_CALIB_DIR = Path(__file__).resolve().parents[1] / "calib"

# OpenCV camera frame (calibration native): +Z boresight, +X right, +Y down.
# Viewsphere camera frame: +X boresight, +Y image-left, +Z image-up.
# Columns of _CV_TO_XFWD are the xfwd axes expressed in cv coordinates.
_CV_TO_XFWD = np.eye(4)
_CV_TO_XFWD[:3, :3] = np.array([[0.0, -1.0, 0.0],
                                [0.0, 0.0, -1.0],
                                [1.0, 0.0, 0.0]])

# Generous workspace box in base frame (m) — cuts the far room, keeps the table
WORKSPACE = {"x": (0.05, 0.60), "y": (-0.45, 0.45), "z": (-0.10, 0.50)}

TABLE_DIST_M = 0.006       # RANSAC inlier threshold for the table plane
ABOVE_TABLE_M = 0.008      # keep points this far above the fitted plane
VOXEL_M = 0.003            # accumulator downsample
DBSCAN_EPS_M = 0.02
DBSCAN_MIN_PTS = 20
#: Single-linkage step for seeded growth. Measured on run 2408-geomtest: the
#: mug's own fragments in the next view sat 0.4-0.8 mm from the seed, while
#: the nearest foreign cluster was 75 mm away and the cable 178 mm. 20 mm is
#: ~4x the pose error (2.5 mm hand-eye residual + mm-level FK) and ~4x under
#: the closest contaminant — and it is already this codebase's notion of
#: "connected" (DBSCAN_EPS_M).
GROW_EPS_M = 0.02
#: A fused view's points may not sit further than this from the cloud so far
#: (Anton 2026-08-24). Seeded growth misses the case that killed 2408-cup1: a
#: cable at the table edge was single-linkage connected to the cup WITHIN one
#: view, so growth had no grounds to refuse it. Measured on that run, every
#: legitimate view landed within 28 mm of the existing cloud and the cable view
#: reached 140 mm, so 80 mm separates them with room on both sides.
JUMP_GATE_M = 0.08
#: Percentile trimmed off each end when sizing the COLLISION box. Raw min/max
#: let 4.7% of points triple the box on 2408-cup1 until it contained the arm.
BOX_PCT = 1.0
MAX_RANGE_M = 0.60         # depth beyond this is noise for our cell
MIN_RANGE_M = 0.13         # gripper fingers live at 70-120mm from the camera —
                           # cut them out of every cloud (crude arm mask, v1)


def load_T_flange_cam(convention: str = "cv") -> np.ndarray:
    """Latest calibrated T_flange_cam (4x4) from calib/ (dated artifacts).

    The frame is the D405 LEFT eye = color/IR-left/depth viewpoint.
    convention: "cv"   — OpenCV camera axes (+Z boresight), projection math;
                "xfwd" — viewsphere camera axes (+X boresight, +Z image-up).
    """
    arts = sorted(_CALIB_DIR.glob("T_flange_cam_*.npy"))
    if not arts:
        raise FileNotFoundError(f"no T_flange_cam_*.npy in {_CALIB_DIR}")
    T = np.load(arts[-1])
    if convention == "cv":
        return T
    if convention == "xfwd":
        return T @ _CV_TO_XFWD
    raise ValueError(f"unknown convention {convention!r}")


UPRIGHT_TOL_DEG = 5.0


def image_tilt_deg(T_base_cam) -> float:
    """How far a capture's image-up sits from world-up, in degrees.

    Read from the POSE, not from a stored roll command: the camera frame is
    the OpenCV one (+X image-right, +Y image-down), so world-up projects into
    the image as (up.X, up.Y) and its angle from image-up is the tilt. Lives
    here because both the capture writer and the image tier need it, and this
    module already owns the camera-frame conventions — `perception` must not
    import `eyes`, and `eyes` must not grow motion dependencies.

    Viewsphere roll is restricted to {0, 180}, so for cell captures this is
    0 or +-180. The hand-taught survey pose is not on the viewsphere and
    measures a few degrees off (-16 on run 2408-seeded).
    """
    T = np.asarray(T_base_cam, dtype=float)
    return float(np.degrees(np.arctan2(T[2, 0], -T[2, 1])))


def is_half_turn(T_base_cam, tol=UPRIGHT_TOL_DEG) -> bool:
    """True when the capture is a 180 deg roll — losslessly correctable."""
    return abs(abs(image_tilt_deg(T_base_cam)) - 180.0) <= tol


def rotate180(img: np.ndarray) -> np.ndarray:
    """Half-turn an image/array. Lossless, no interpolation, keeps the shape
    (so a corrected frame is still 848x480 and directly comparable)."""
    return np.ascontiguousarray(img[::-1, ::-1])


def deproject(depth_u16: np.ndarray, intr: dict, depth_scale: float,
              max_range: float = MAX_RANGE_M, min_range: float = MIN_RANGE_M,
              stride: int = 2, return_px: bool = False):
    """Raw depth image -> (N,3) points in the camera (IR-left) frame.

    min_range acts as the v1 arm mask: the gripper fingers permanently hover
    7-12cm in front of the wrist camera and otherwise become the largest
    above-table cluster (observed 2026-08-14 — they track the camera).

    `return_px=True` additionally returns the (N,2) int array of (v, u)
    source pixels, in FULL-resolution image coordinates (the stride is
    already applied) — this is what pairs each point with its rgb pixel.
    """
    z = depth_u16[::stride, ::stride].astype(np.float64) * depth_scale
    h, w = z.shape
    us = np.arange(0, depth_u16.shape[1], stride)[:w]
    vs = np.arange(0, depth_u16.shape[0], stride)[:h]
    uu, vv = np.meshgrid(us, vs)
    valid = (z > min_range) & (z < max_range)
    z = z[valid]
    x = (uu[valid] - intr["ppx"]) / intr["fx"] * z
    y = (vv[valid] - intr["ppy"]) / intr["fy"] * z
    pts = np.column_stack([x, y, z])
    if return_px:
        return pts, np.column_stack([vv[valid], uu[valid]]).astype(int)
    return pts


def cam_to_base(points_cam: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    """Camera-frame points -> base frame. T_base_cam comes stamped in every
    capture bundle's meta.json (RTDE joints -> flange FK -> hand-eye)."""
    return points_cam @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]


def _workspace_mask(points: np.ndarray) -> np.ndarray:
    m = np.ones(len(points), dtype=bool)
    for i, ax in enumerate("xyz"):
        lo, hi = WORKSPACE[ax]
        m &= (points[:, i] >= lo) & (points[:, i] <= hi)
    return m


def crop_workspace(points: np.ndarray) -> np.ndarray:
    return points[_workspace_mask(points)]


def fit_table(points: np.ndarray) -> np.ndarray:
    """RANSAC plane fit -> [a,b,c,d] with unit normal pointing +z (up)."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    plane, _ = pcd.segment_plane(distance_threshold=TABLE_DIST_M,
                                 ransac_n=3, num_iterations=500)
    plane = np.asarray(plane)
    if plane[2] < 0:      # normal must point up
        plane = -plane
    return plane


def _above_mask(points: np.ndarray, plane: np.ndarray,
                margin: float = ABOVE_TABLE_M) -> np.ndarray:
    d = points @ plane[:3] + plane[3]
    return d > margin


def above_table(points: np.ndarray, plane: np.ndarray,
                margin: float = ABOVE_TABLE_M) -> np.ndarray:
    return points[_above_mask(points, plane, margin)]


def _largest_cluster_mask(points: np.ndarray) -> np.ndarray:
    import open3d as o3d
    if len(points) < DBSCAN_MIN_PTS:
        return np.ones(len(points), dtype=bool)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    labels = np.asarray(pcd.cluster_dbscan(eps=DBSCAN_EPS_M,
                                           min_points=DBSCAN_MIN_PTS))
    if labels.max() < 0:
        return np.ones(len(points), dtype=bool)
    counts = np.bincount(labels[labels >= 0])
    return labels == counts.argmax()


def largest_cluster(points: np.ndarray) -> np.ndarray:
    """Largest DBSCAN cluster — v1 stand-in for detection-seeded growth."""
    return points[_largest_cluster_mask(points)]


def grow_from_seed(seed: np.ndarray, points: np.ndarray,
                   eps: float = GROW_EPS_M) -> np.ndarray:
    """Points reachable from `seed` by chained steps of at most `eps`.

    Single-linkage growth, not DBSCAN: membership is "within eps of a point
    already in the set", with no density requirement, so a sparse fragment of
    the object still joins (in 2408-geomtest the mug survived a view only as
    229/93/70-point shards — a min_points rule would have discarded them).

    Growth is transitive by design. One view under-reports the object, so the
    set must be able to extend past its current extent as new surface comes
    into view; a frozen gate would reject real points forever. The price is
    the one failure mode this has: it can leak along a physical chain of
    points spaced under `eps`. It cannot latch onto a separate object, which
    is the failure that actually bit us.

    `seed` is the cloud so far, in the same base frame as `points`.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return points[_grow_mask(seed, points, eps)]


def _grow_mask(seed: np.ndarray, points: np.ndarray,
               eps: float = GROW_EPS_M) -> np.ndarray:
    """Boolean membership form of `grow_from_seed`, row-aligned with
    `points` — the form that lets a caller carry parallel data (colours)
    through the growth without disturbing the measured policy above."""
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=float).reshape(-1, 3)
    seed = np.asarray(seed, dtype=float).reshape(-1, 3)
    keep = np.zeros(len(points), dtype=bool)
    if not len(points) or not len(seed):
        return keep
    tree = cKDTree(points)
    frontier = seed
    while len(frontier):
        hits = [h for h in tree.query_ball_point(frontier, eps) if h]
        if not hits:
            break
        idx = np.unique(np.concatenate(hits)).astype(int)
        idx = idx[~keep[idx]]          # only the newly reached ones advance
        if not len(idx):
            break
        keep[idx] = True
        frontier = points[idx]
    return keep


#: Slack added around a reprojected cloud when it is used as a prompt box.
#: The box only has to say WHERE — the segmenter decides what is object and
#: what is background inside it — so a loose box is cheap and a tight one that
#: clips the object is not. Measured on 2408-cup2: boxes containing a large
#: slice of cable still returned clean masks (0.73-0.97).
#: PROMPT_GROW scales with the box (Anton 2026-09-08: "20% larger whatever
#: the bbox is" — 10% per side); PROMPT_MARGIN_PX is the FLOOR per side, so
#: a small far-away box keeps at least the old fixed slack.
PROMPT_GROW = 0.10
PROMPT_MARGIN_PX = 12
#: Percentile trimmed off the reprojection before boxing it, for the same
#: reason `aabb` has one: a handful of stray points must not define the box.
PROMPT_PCT = 1.0


def min_yaw_aabb(points: np.ndarray, pct: float = 0.0
                 ) -> tuple[np.ndarray, np.ndarray, float]:
    """Min-area box over YAW only: (center, dims, yaw), base frame.

    Yaw-only on purpose (Anton 2026-09-08): the object rests on the table,
    so the true tightening is in the tabletop plane — a 45-degree box under
    an axis-aligned AABB inflates its footprint by up to sqrt(2), which is
    exactly the oversized yellow box on run 0809-box2. A full 3D OBB would
    tilt the collision box into the table for nothing.

    Brute force over 1-degree steps of a quarter turn (the box is symmetric
    under 90 degrees), fully vectorised: `pct` trims stray points ONCE, in
    the base frame (same guard as `CloudAccumulator.aabb`, against the
    cable-point boxes of 2408-cup1), then one (N,2)@(2,90) matmul per axis
    gives every yaw's extents in a single pass — ~2 ms at 10k points where
    a per-angle percentile loop cost 20 ms, and this runs in the settle leg
    with the arm waiting (2x per step + once per sweep ranking pass).
    """
    pts = np.asarray(points, dtype=float)
    if pct > 0 and len(pts) > 2:
        lo3, hi3 = np.percentile(pts, [pct, 100.0 - pct], axis=0)
        keep = np.all((pts >= lo3) & (pts <= hi3), axis=1)
        if keep.any():
            pts = pts[keep]
    xy = pts[:, :2]
    # The ANGLE is searched on a stride sample (the argmin over 90 candidates
    # is insensitive to thinning a voxel-dedup'd surface cloud); the EXTENTS
    # are then exact, from every point at the winning yaw. Keeps the (N,90)
    # temporaries bounded, so the cost stays ~4 ms no matter the cloud.
    sample = xy[::max(1, len(xy) // 12_000)]
    th = np.radians(np.arange(90.0))
    c, s = np.cos(th), np.sin(th)
    u = sample @ np.vstack([c, s])                 # (n, 90): x' in each frame
    v = sample @ np.vstack([-s, c])                # (n, 90): y'
    i = int(np.argmin((u.max(0) - u.min(0)) * (v.max(0) - v.min(0))))
    yaw = float(th[i])
    ux = xy @ np.array([c[i], s[i]])               # exact, all points, one yaw
    vx = xy @ np.array([-s[i], c[i]])
    lo = np.array([ux.min(), vx.min()])
    hi = np.array([ux.max(), vx.max()])
    mid = (lo + hi) / 2
    zlo, zhi = pts[:, 2].min(), pts[:, 2].max()
    center = np.array([c[i] * mid[0] - s[i] * mid[1],
                       s[i] * mid[0] + c[i] * mid[1], (zlo + zhi) / 2])
    dims = np.array([hi[0] - lo[0], hi[1] - lo[1], zhi - zlo])
    return center, dims, yaw


def object_box(points: np.ndarray, pct: float = BOX_PCT
               ) -> tuple[np.ndarray, list]:
    """The object's collision box: `(dims, pose6)` for `world.set_object`.

    THE one box: `machine._recenter` (what the planner avoids),
    `machine._publish_object` (the yellow box the UI draws) and
    `collect._mirror_object_box` (the sweep's ranking world) must all say
    the same thing, and before this helper each carried its own copy of the
    margin math — the UI's had even drifted to an untrimmed AABB. Margins:
    2 cm each side (+4 cm per dim), 5 cm floor — coal adds its own padding
    per side on top, and the pct trim slightly under-reads the object,
    which those two margins are sized to cover (see `aabb`).
    """
    center, dims, yaw = min_yaw_aabb(points, pct=pct)
    dims = np.maximum(dims + 0.04, 0.05)
    return dims, [*center.tolist(), 0.0, 0.0, yaw]


def project_to_pixels(points: np.ndarray, T_base_cam: np.ndarray,
                      intr: dict) -> np.ndarray:
    """Base-frame points -> (u, v) pixels, dropping anything behind the lens.

    Plain pinhole projection in the OpenCV camera convention, which is what
    `T_base_cam` and the colour/ir intrinsics both use. No distortion: the
    D405 colour coefficients are small and this feeds a padded box, not a
    measurement.
    """
    points = np.asarray(points, dtype=float)
    if not len(points):
        return np.empty((0, 2))
    Ti = np.linalg.inv(np.asarray(T_base_cam, dtype=float))
    pc = (Ti[:3, :3] @ points.T).T + Ti[:3, 3]
    pc = pc[pc[:, 2] > 1e-3]
    if not len(pc):
        return np.empty((0, 2))
    return np.column_stack([intr["ppx"] + intr["fx"] * pc[:, 0] / pc[:, 2],
                            intr["ppy"] + intr["fy"] * pc[:, 1] / pc[:, 2]])


def prompt_box(points: np.ndarray, T_base_cam: np.ndarray, intr: dict,
               shape: tuple, margin: int = PROMPT_MARGIN_PX,
               pct: float = PROMPT_PCT,
               floor_z: float | None = None) -> tuple | None:
    """Where known object points land in THIS view, as a box to prompt with.

    This is how object identity survives a viewpoint change without any text
    prompt or tracking state: the accumulated cloud plus the arm's own pose
    already say where the object must appear, so each new frame gets a fresh
    prompt for free. Returns None when too little of the cloud is visible to
    trust — the caller then falls back rather than segmenting a guess.

    `floor_z` (base-frame table height) extrudes the cloud down to the table
    before projecting. Without it the box is anchored on whatever has been
    SEEN, and a run that opens top-down has seen only the top face — so every
    side view gets a top-band box, the mask stays on the top face, the sides
    never enter the cloud, and the anchor can never grow (run 0809-0809box:
    a 70 mm box believed 31 mm tall for all 27 views). Objects rest on the
    table — the same assumption the survey's "NO OBJECT above the table"
    check already makes — so the shadow of the seen cloud on the table is
    always inside the object's true silhouette in x/y and reaches its true
    bottom in z. Measured on that run's side view 015: 856 -> 4171 fused
    points, and the fused cloud grew real sides.
    """
    points = np.asarray(points, dtype=float)
    if floor_z is not None and len(points):
        floor = points.copy()
        floor[:, 2] = float(floor_z)
        points = np.vstack([points, floor])
    uv = project_to_pixels(points, T_base_cam, intr)
    if len(uv) < 10:
        return None
    h, w = shape[:2]
    lo = np.percentile(uv, pct, axis=0)
    hi = np.percentile(uv, 100.0 - pct, axis=0)
    grow = np.maximum(PROMPT_GROW * (hi - lo), margin)  # per side, per axis
    lo, hi = lo - grow, hi + grow
    box = (int(max(0, lo[0])), int(max(0, lo[1])),
           int(min(w, hi[0])), int(min(h, hi[1])))
    return box if box[2] > box[0] and box[3] > box[1] else None


def object_in_base(depth_u16: np.ndarray, intr: dict, depth_scale: float,
                   T_base_cam: np.ndarray, seed: np.ndarray | None = None,
                   eps: float = GROW_EPS_M,
                   mask: np.ndarray | None = None,
                   rgb: np.ndarray | None = None) -> dict:
    """One view -> object points in the BASE frame (no rectification).

    The loop plans in the base frame: the viewsphere center and the
    collision box must not live in a table-rectified frame. UR5e FK is
    mm-accurate, so no per-view tilt correction is needed.

    Two ways to decide which points are the object:

    `mask` — a (H, W) bool image mask of the object. When given it IS the
        answer, and `seed`/`eps` are ignored: identity was settled in image
        space, where a cable lying against the cup is obviously not the cup.
        This is the only discriminator that survives contact: every purely
        geometric guard we built (seeded growth, the 8 cm jump gate) reasons
        about DISTANCE, and on run 2408-cup2 view 003 the cable was within
        20 mm of the cup in 3D — genuinely adjacent, so distance had nothing
        to say. Measured there: 175.8 -> 89.1 mm object extent.
        Pass the mask and the depth in the SAME frame — `depth_aligned` with
        the colour intrinsics, since the mask is computed on rgb (the two
        agree with the ir_left reconstruction to 2.6 mm, under the hand-eye
        residual).

    `seed` — the object cloud accumulated so far, used when there is no mask
        (bootstrap, or the segmenter missed). Object points are the ones
        connected to it, not whichever cluster happens to be biggest in this
        frame. An empty or absent seed means bootstrap — elect the largest
        cluster.

    The table plane is always fitted on the FULL view, never on the masked
    subset: the plane needs the table, and the mask deliberately removes it.
    That full view IS workspace-cropped — the crop's job is to keep the plane
    fit and the depth fallback looking at the workcell rather than the room.
    The masked object points are NOT cropped: see the note below.

    A view that finds nothing returns zero points and a None centroid. That
    is an ordinary outcome, not an error: the view was valid, the object just
    wasn't legible from there. The caller adds nothing and carries on.

    `rgb` — the (H, W, 3) uint8 colour frame in the SAME orientation as the
    depth. When given, the result carries `"colors"`: uint8 (N, 3), row-
    aligned with `"points"` — each point's source pixel sampled from `rgb`.
    On the mask path (depth_aligned + colour intrinsics) the pairing is
    exact by construction. On the depth paths (depth_raw + IR intrinsics)
    it is approximate: the D405 derives colour from the left imager, so the
    viewpoints coincide and only the ~1.6% intrinsics delta shifts the
    sample a few pixels — accepted for a fallback path whose job is a
    legible cloud, not radiometric truth. Without `rgb`, `"colors"` is None.
    """
    def _take(a, m):
        return None if a is None else a[m]

    pts_cam, px_all = deproject(depth_u16, intr, depth_scale, return_px=True)
    pts_all = cam_to_base(pts_cam, T_base_cam)
    if len(pts_all) != len(px_all):
        # A wrapped `cam_to_base` changed the row set (the replay's legacy
        # workspace crop does this) — pixel pairing is gone, so colours are
        # honestly None rather than silently misaligned.
        px_all = None
    ws = _workspace_mask(pts_all)
    pts, pts_px = pts_all[ws], _take(px_all, ws)
    if len(pts) < 100:
        raise RuntimeError(f"only {len(pts)} workspace points — bad view?")
    plane = fit_table(pts)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != depth_u16.shape:
            raise ValueError(f"mask {mask.shape} does not match "
                             f"depth {depth_u16.shape}")
        # NOT crop_workspace (Anton 2026-08-24). `WORKSPACE` is a SCENE
        # filter — it exists to cut the far room so the table plane and the
        # depth fallback see only the workcell — and a masked pixel is by
        # definition not scene clutter. Cropping the object with it made the
        # cloud a hard axis-aligned slab: on run 2408-cup4 the mug sat against
        # the x=0.05 face, 22% of every masked view was amputated, and the
        # object measured 92.8 mm instead of 126.9. Worse, the missing half
        # was the side FACING THE ROBOT, so the collision box was ~32 mm short
        # exactly where the arm reaches across.
        # Masked points are still bounded: `deproject` clips to
        # [MIN_RANGE_M, MAX_RANGE_M], `above_table` drops the table, and the
        # accumulator's jump gate remains the model-independent backstop.
        m_pts_cam, m_px = deproject(np.where(mask, depth_u16, 0), intr,
                                    depth_scale, return_px=True)
        m_pts = cam_to_base(m_pts_cam, T_base_cam)
        if len(m_pts) != len(m_px):
            m_px = None
        if len(m_pts):
            am = _above_mask(m_pts, plane)
            obj, px = m_pts[am], _take(m_px, am)
        else:
            obj, px = m_pts, m_px
    elif seed is None or not len(seed):
        am = _above_mask(pts, plane)
        above, above_px = pts[am], _take(pts_px, am)
        if len(above):
            cm = _largest_cluster_mask(above)
            obj, px = above[cm], _take(above_px, cm)
        else:
            obj, px = above, above_px
    else:
        am = _above_mask(pts, plane)
        above, above_px = pts[am], _take(pts_px, am)
        gm = _grow_mask(seed, above, eps)
        obj, px = above[gm], _take(above_px, gm)
    colors = None
    if rgb is not None and px is not None:
        rgb = np.asarray(rgb)
        colors = np.ascontiguousarray(
            rgb[px[:, 0], px[:, 1]]).astype(np.uint8).reshape(-1, 3)
    return {
        "points": obj,
        "colors": colors,
        "centroid": obj.mean(axis=0) if len(obj) else None,
        "plane": plane,
        "n_scene": len(pts),
    }


class CloudAccumulator:
    """Grows the object cloud across views; centroid tracks the fused cloud."""

    #: Colour given to points fused without one — mid-grey, so an uncoloured
    #: contribution is visibly "no data" rather than a plausible colour.
    GRAY = 128

    def __init__(self, voxel: float = VOXEL_M):
        self.voxel = voxel
        self._points = np.empty((0, 3))
        self._colors = np.empty((0, 3), dtype=np.uint8)
        #: The subset of the LAST `add` that survived the jump gate — what
        #: this view actually contributed. A diagnostic side-channel for the
        #: chain overlay: the fused cloud is voxelised and outlier-filtered
        #: as a whole, so after the fact there is no way to ask which points
        #: came from where. Never read by the loop itself.
        self.last_kept = np.empty((0, 3))

    def add(self, points: np.ndarray,
            colors: np.ndarray | None = None) -> int:
        """Fuse one view's points. Returns how many were REJECTED as detached.

        Points further than `JUMP_GATE_M` from anything already accumulated do
        not belong to this object (Anton 2026-08-24). Seeded growth is not
        enough on its own: on run 2408-cup1 the 10 deg ring caught a cable at
        the table edge, and at GROW_EPS_M=20 mm the cable was genuinely
        single-linkage connected to the cup INSIDE one view, so growth had no
        reason to refuse it. Measured there: every legitimate view put its new
        points within 28 mm of the existing cloud, while the cable view reached
        140 mm — a wide, safe separation.

        This cleans the CLOUD. It does not by itself fix the collision box:
        cable points that pass within 80 mm still stretch a raw min/max extent
        (measured: 118 x 154 mm, arm still inside its own object box). That is
        what `aabb(pct=...)` is for — the two guards are complementary.

        `colors` — uint8 (N, 3), row-aligned with `points` (the `"colors"`
        field of `object_in_base`). Colours ride every filter the points do:
        the jump gate drops them together, the voxel average blends them, the
        outlier filter keeps the survivors'. `None` fuses mid-grey so the
        cloud/colour row alignment can never break.
        """
        import open3d as o3d
        self.last_kept = np.empty((0, 3))
        if len(points) == 0:
            return 0
        if colors is None:
            colors = np.full((len(points), 3), self.GRAY, dtype=np.uint8)
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        if len(colors) != len(points):
            raise ValueError(f"{len(colors)} colors for {len(points)} points")
        rejected = 0
        if len(self._points) and JUMP_GATE_M:
            from scipy.spatial import cKDTree
            keep = cKDTree(self._points).query(points)[0] <= JUMP_GATE_M
            rejected = int((~keep).sum())
            points, colors = points[keep], colors[keep]
            if not len(points):
                return rejected
        self.last_kept = np.asarray(points)
        merged = np.vstack([self._points, points])
        merged_c = np.vstack([self._colors, colors])
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(merged))
        pcd.colors = o3d.utility.Vector3dVector(merged_c.astype(np.float64) / 255.0)
        pcd = pcd.voxel_down_sample(self.voxel)
        if len(pcd.points) > 50:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.5)
        self._points = np.asarray(pcd.points)
        self._colors = np.clip(np.rint(np.asarray(pcd.colors) * 255.0),
                               0, 255).astype(np.uint8)
        return rejected

    @property
    def points(self) -> np.ndarray:
        return self._points

    @property
    def colors(self) -> np.ndarray:
        """uint8 (N, 3), row-aligned with `points` for the life of the run."""
        return self._colors

    @property
    def centroid(self) -> np.ndarray | None:
        return self._points.mean(axis=0) if len(self._points) else None

    def aabb(self, pct: float = 0.0) -> tuple[np.ndarray, np.ndarray] | None:
        """Extent of the cloud. `pct` trims that percentage off each end.

        Raw min/max lets a handful of stray points define the collision box:
        on run 2408-cup1, 168 points (4.7%) grew it from 94 x 116 to
        177 x 188 mm, and the inflated box then CONTAINED the arm's own
        configuration, so OMPL could not initialise its start tree and the run
        died. `pct=1` reproduces the cup to within a few mm (87 x 111 vs a
        clean 94 x 116) while being immune to that.

        Deliberately NOT the default for every caller: a percentile extent is
        slightly UNDER the true object, which is only safe because the
        collision box adds 40 mm and coal another 20 mm per side.
        """
        if not len(self._points):
            return None
        if pct <= 0:
            return self._points.min(axis=0), self._points.max(axis=0)
        lo, hi = np.percentile(self._points, [pct, 100.0 - pct], axis=0)
        return lo, hi
