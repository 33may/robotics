#!/usr/bin/env python3
"""The one module that talks to Rerun. Everything else hands it structures.

Why a module and not a pile of `rr.log` calls in each script. The review
workspace is a *schema*: entity paths, colours, radii, what is static and
what moves with the slider, and the blueprint that arranges it. If every
producer invented its own paths, two `.rrd` files from two scripts would
be two different apps and the muscle memory of scrubbing one would not
transfer to the other. So the schema lives here, once, and a producer's
whole job is to hand over arrays.

The load-bearing part is `log_primitives`. A fitter backend — o3d planes,
RANSAC cylinders, superquadrics, whatever lands next — never learns a
Rerun symbol. It returns a list of plain dicts, the five CLOSED kinds of
`vbti/inspection/fit/contract.py`::

    {"kind":   "box" | "cylinder" | "plane_patch" | "superquadric" | "mesh",
     "pose":   4x4 matrix, or (center, 3x3 rotation), or None,
     "params": box          -> {"extents": (dx, dy, dz)}      # FULL extents
               cylinder     -> {"radius": r, "length": l}     # axis = local +Z
               plane_patch  -> {"extents": (dx, dy)}          # in plane XY
               superquadric -> {"scale": (a1,a2,a3), "eps": (e1,e2)}  # SEMI-axes
               mesh         -> {"vertices": (N,3), "faces": (M,3)},
     "label":  "face_0",            # entity leaf + on-screen label
     "method": "cyl",               # entity group + deterministic colour
     "inliers": optional index array into the step's cloud,
     "residual_mm", "support": optional, folded into the hover label}

and this module renders it. Adding a fitter is adding a function that
returns that; there is no viewer work. `superquadric` is triangulated
HERE (`superquadric_mesh`) rather than producer-side, because the kind is
closed and analytic: every fitter that emits one would otherwise ship its
own tessellation and two methods would be compared through two different
meshes. Anything genuinely non-analytic still enters as `mesh`.

Frames. Everything 3D is in the robot BASE frame, Z up, metres. The world
entity carries `ViewCoordinates.RIGHT_HAND_Z_UP` so Rerun's camera
gizmo agrees with the cell.

Timeline. One int sequence, `step`. Step k is the state of the world
AFTER the k-th view was fused, k from 0. Rerun's latest-at semantics do
the persistence for free: a camera frustum logged once at its own step
stays on screen for every later step, while an entity re-logged every
step (the cloud, the fresh-point glow, the chain images) shows only what
belongs to the step under the slider.

Every archetype name below was checked against the installed
rerun-sdk 0.37.0, not remembered.
"""
from __future__ import annotations

import colorsys
import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

# --------------------------------------------------------------- the schema
TIMELINE = "step"

WORLD = "/world"
CLOUD = "/world/cloud"          # fused cloud at step k          (per step)
CLOUD_RGB = "/world/cloud_rgb"  # same cloud, captured colours   (per step, off)
FRESH = "/world/fresh"          # what view k added, glowing     (per step)
TABLE = "/world/table"          # fitted table plane             (static)
CAMS = "/world/cam"             # frustum + rgb per view         (at its step)
FACES = "/world/faces"          # /world/faces/<name>: face-view candidates
                                 # (Task 6) — its OWN root, never under CAMS,
                                 # so a reviewer can toggle face views alone.
PRIM = "/world/prim"            # /world/prim/<method>/<label>   (per step)
OVERLAY = "/world/overlay"      # stored vs replayed final cloud (static, off)
OVERLAY_STORED = OVERLAY + "/stored"
OVERLAY_REPLAYED = OVERLAY + "/replayed"
CHAIN = "/chain"                # /chain/<slot>, 2D images       (per step)
ACTIVE = "/active_cam"          # step-k rgb + pinhole, ONE path (per step)
STEP_PLY = "/world/step_cloud"  # per-view cloud.ply, legacy      (off)

#: The accumulator voxel is 3 mm (`vbti/inspection/cell/geometry.py:46`), so a
#: 1.5 mm radius draws one ball per occupied voxel with no gaps and no mush.
R_CLOUD = 0.0015
R_FRESH = 0.0028                # deliberately fatter — the glow must win
R_OVERLAY = 0.0018

C_CLOUD = (152, 158, 170)       # neutral: the cloud is the backdrop, not the point
C_FRESH = (255, 146, 26)        # accent: "this is what view k bought you"
C_TABLE = (86, 100, 124, 70)    # translucent slab
C_STORED = (60, 170, 255)       # story 5, channel A
C_REPLAYED = (255, 74, 138)     # story 5, channel B
C_CAM = (120, 205, 255)
#: The frustum that captured the step under the slider. Same accent as the
#: fresh glow on purpose: both mean "this is what step k is", and orange
#: against thirty cyan cones is readable from across the room.
C_CAM_ACTIVE = (255, 146, 26)

#: Face-view candidates (Task 6), by status — see `log_face_views`. Chosen to
#: be distinct from every colour already in this module (cam cyan/orange,
#: fit green/violet/amber, fresh orange, overlay blue/pink).
C_FACE_READY = (120, 235, 140)       # green: reachable, orthogonal (or unresolved)
C_FACE_DEGRADED = (240, 200, 60)     # yellow: reachable, off-axis by off_angle_deg
C_FACE_UNREACHABLE = (205, 70, 70)   # red
C_FACE_VISITED = (90, 160, 255)      # blue
C_FACE_SETTLING = (140, 140, 148)    # gray: box mid-convergence (rare to log)
#: Face status carries no fixed on-screen length (there is no per-candidate
#: intrinsics block to size a frustum from) — a short arrow along the
#: boresight is the "small frustum or arrow" the design calls for.
FACE_ARROW_M = 0.035

#: Frustum length in metres. At the colour intrinsics this puts a ~120 mm
#: image plane at the end of each cone — readable, and small enough that
#: thirty of them at the end of a scrub still leave the 100 mm cup visible.
#: They live under one entity (`/world/cam`), so one eye-click in the
#: blueprint tree takes the whole thicket off (Anton, 2026-09-03).
FRUSTUM_M = 0.06

#: Colours are per METHOD and deterministic, so "the blue boxes" means the
#: same fitter in every run and in every screenshot. Hand-pinned for the
#: bank methods that exist (`vbti/inspection/fit`); hashed for the ones that do
#: not exist yet.
#:
#: The three pinned hues are chosen to be scoreable in a screenshot: each
#: one is far from every other colour this schema paints (neutral cloud,
#: orange fresh glow, cyan frusta, slate table, blue/pink overlay), so
#: "how much planar is on screen" is a pixel count and not an opinion.
_METHOD_COLORS = {
    "planar": (99, 220, 140),       # green
    "cyl": (150, 140, 255),         # violet
    "ems": (255, 214, 61),          # amber
}

#: A `plane_patch` has no thickness; it is drawn as a slab this thick so it
#: reads as a surface from any orbit angle (the same trick as `log_table`).
PLANE_T = 0.0016

#: Tessellation of a `superquadric`. 48x24 puts a quad every 7.5 degrees
#: around and every 7.5 degrees up, which is smooth at the ~100 mm scale of
#: our objects and costs ~1.2 k vertices per fit — 30 steps of it is under
#: a megabyte in the .rrd.
SQ_RES = (48, 24)

#: Alpha of the superquadric shell. It is a CLOSED hull around the cloud it
#: was fitted to, so an opaque one hides the very points the review is
#: about; the shell is drawn translucent so the cloud shows through it and
#: "does the hull hug the surface" stays answerable.
SQ_ALPHA = 110


def method_color(method: str) -> tuple[int, int, int]:
    """Stable RGB for a fitter backend name.

    Deterministic on the *name*, not on insertion order: a run with two
    methods and a run with five give the same method the same colour, so
    a screenshot from last week still reads correctly.
    """
    if method in _METHOD_COLORS:
        return _METHOD_COLORS[method]
    h = int(hashlib.sha1(method.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    r, g, b = colorsys.hsv_to_rgb(h, 0.62, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


# ------------------------------------------------------------------ session
def begin(app_id: str) -> None:
    """Start a recording whose only timeline is `step`.

    `set_log_time_enabled(False)` / `set_log_tick_enabled(False)` matter more
    than they look: without them Rerun adds `log_time` and `log_tick`
    timelines, the time panel opens on whichever it feels like, and the
    slider the user grabs is wall-clock of the *producer process* — a
    timeline about nothing. One run, one meaningful axis.
    """
    rr.init(app_id)
    rr.set_log_time_enabled(False)
    rr.set_log_tick_enabled(False)


def at(step: int) -> None:
    """Put every following log call on step k."""
    rr.set_time(TIMELINE, sequence=int(step))


def save(path: str | Path, blueprint: rrb.Blueprint, app_id: str) -> Path:
    """Write the complete `.rrd`, blueprint baked in (story 1: zero clicks).

    Two artefacts, one design. The `.rrd` carries the blueprint as its
    DEFAULT — enough for a viewer that has never seen this app-id. But the
    native viewer persists the ACTIVE blueprint per app-id on exit
    (`~/.local/share/rerun/blueprints/<app_id>.rbl`), and a default never
    overrides an active one: "the standard behavior is to only update the
    default blueprint … you need to click the reset button" (rerun docs,
    Configure the Viewer through code). An app-id that was reviewed days
    ago would therefore open with the days-old layout, silently.

    So the same blueprint is ALSO written as a sidecar `.rbl`. Loading an
    .rbl activates it, so a viewer launched with both files shows this
    design regardless of what its cache remembers — verified against
    rerun 0.37 with a deliberately stale persisted blueprint. `view.py`
    passes the sidecar whenever it sits next to the rrd.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rr.save(path, default_blueprint=blueprint)
    blueprint.save(app_id, str(path.with_suffix(".rbl")))
    return path


# ------------------------------------------------------------------- 3D log
def log_world() -> None:
    """Base frame, Z up. Static — it is true at every step."""
    rr.log(WORLD, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


def log_cloud(points: np.ndarray, colors: np.ndarray | None = None,
              static: bool = False) -> None:
    """The fused cloud as of the current step (story 2: it grows).

    Two entities, one cloud: `CLOUD` is the neutral backdrop the fits are
    judged against; `CLOUD_RGB` is the same points in their captured
    colours (`fused_colors.npy` / colored step PLYs), hidden by default —
    the eye icon IS the colour toggle. A step without colours logs an
    empty `CLOUD_RGB` so a scrub never shows a stale colouring.

    `static=True` for a run that has ONLY its final cloud (v1 runs in the
    reduced viewer): timeless, so the 3D pane has the object at every
    slider position — including step 0 — instead of only at the end.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rr.log(CLOUD, rr.Points3D(points, colors=C_CLOUD, radii=R_CLOUD),
           static=static)
    if colors is not None and len(colors) == len(points):
        rr.log(CLOUD_RGB, rr.Points3D(points, colors=np.asarray(colors),
                                      radii=R_CLOUD), static=static)
    else:
        rr.log(CLOUD_RGB, rr.Points3D(np.zeros((0, 3), dtype=np.float32)),
               static=static)


def log_fresh(points: np.ndarray) -> None:
    """Points this step's view contributed, in the accent colour.

    Logged to a single entity path every step rather than to a per-step
    path: latest-at then makes the glow *replace* itself, so exactly one
    view is ever highlighted and "what did view 17 add" is one drag away.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rr.log(FRESH, rr.Points3D(points, colors=C_FRESH, radii=R_FRESH))


def log_table(plane: Sequence[float], center: Sequence[float],
              half_extent: float = 0.14, thickness: float = 0.0015) -> None:
    """A thin translucent patch on the fitted table plane `[a,b,c,d]`.

    Drawn as a flattened box because Rerun 0.37 has no plane archetype and
    a slab reads as a surface from any orbit angle, which a wireframe
    quad does not.

    A PATCH under the object, not the whole workspace. The first version
    was 0.6 m square — the size of the WORKSPACE crop — and Rerun's
    auto-framing then opened every recording zoomed to the table, with the
    100 mm cup a speck in the middle and the camera frusta lost against
    it. The plane is context for the cloud, so it is sized to the cloud.
    """
    a, b, c, d = (float(v) for v in plane)
    n = np.array([a, b, c], dtype=float)
    n /= np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    center = np.asarray(center, dtype=float).copy()
    # Project the requested centre onto the plane so the slab sits ON it.
    center -= (float(np.dot(n, center)) + d / np.linalg.norm([a, b, c])) * n
    rr.log(TABLE,
           rr.Boxes3D(centers=[center], half_sizes=[(half_extent,
                                                     half_extent,
                                                     thickness)],
                      quaternions=[_quat_from_z(n)], colors=[C_TABLE],
                      fill_mode="Solid", labels=["table plane (re-fitted)"],
                      show_labels=False),
           static=True)


def _K(intr: Mapping) -> np.ndarray:
    """3x3 camera matrix from a session.json intrinsics block."""
    return np.array([[intr["fx"], 0.0, intr["ppx"]],
                     [0.0, intr["fy"], intr["ppy"]],
                     [0.0, 0.0, 1.0]], dtype=float)


def log_camera(pose_id: int, T_base_cam: np.ndarray, intr: Mapping | None,
               image_path: str | Path | None = None,
               label: str | None = None,
               color: tuple = C_CAM) -> None:
    """A frustum at the pose this view actually shot from, wearing its rgb.

    Logged at the current step and never again, so it appears when its
    view is fused and persists for the rest of the scrub (story 2). The
    producer passes `color=C_CAM_ACTIVE` for the step's own camera and
    demotes it next step with `recolor_camera` — see there.

    `intr` is a session.json intrinsics block (`fx, fy, ppx, ppy, width,
    height`) — or None (mock/legacy runs without intrinsics), which logs
    the pose alone: axes where the camera stood, no cone, no preview.
    `camera_xyz=RDF` is the OpenCV convention the cell's `deproject` uses
    (`geometry.py`), so the frustum points where the depth actually went.
    """
    T = np.asarray(T_base_cam, dtype=float)
    path = f"{CAMS}/{pose_id:03d}"
    # Three levels, and each one earns its place. The POSE is on the camera
    # entity, so hiding `/world/cam` takes the axis gizmos with it. The
    # PINHOLE is on a child rather than on the camera itself, because a
    # pinhole makes its children image-plane space and the preview has to
    # stay a sibling in metres. The PREVIEW is a textured mesh in camera
    # coordinates, which the parent transform then places.
    rr.log(path, rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
    if intr is None:
        return
    rr.log(f"{path}/frustum",
           rr.Pinhole(image_from_camera=_K(intr),
                      resolution=(intr["width"], intr["height"]),
                      camera_xyz=rr.ViewCoordinates.RDF,
                      image_plane_distance=FRUSTUM_M,
                      color=color))
    if image_path is not None:
        _log_preview_quad(path, intr, image_path)


def recolor_camera(pose_id: int, intr: Mapping | None,
                   color: tuple = C_CAM) -> None:
    """Re-log frustum `pose_id`'s pinhole in `color` at the CURRENT step.

    The active-camera highlight is latest-at state: step k logs its own
    frustum in `C_CAM_ACTIVE` (`log_camera(color=...)`) and demotes step
    k-1's back to `C_CAM` with this. Exactly one frustum is ever lit, and
    scrubbing backwards relights the right one for free.
    """
    if intr is None:
        return
    rr.log(f"{CAMS}/{int(pose_id):03d}/frustum",
           rr.Pinhole(image_from_camera=_K(intr),
                      resolution=(intr["width"], intr["height"]),
                      camera_xyz=rr.ViewCoordinates.RDF,
                      image_plane_distance=FRUSTUM_M,
                      color=color))


def _face_color(status: str, off_angle_deg: float | None) -> tuple:
    """Status -> colour (design: ready/orthogonal green, degraded yellow,
    unreachable red, visited blue, settling gray).

    `status` here is `record/schema.py:ViewCandidate.status` as WRITTEN by
    `machine.py::_view_state` — "available" (the schema's word for what
    `_face_states()` calls "ready"), "visited", or "unreachable"; "settling"
    candidates are never recorded, so that branch is defensive only. Ready
    splits into orthogonal/degraded by `off_angle_deg`: unresolved (no plan
    has run yet) and exactly 0 both read as orthogonal green.
    """
    if status == "unreachable":
        return C_FACE_UNREACHABLE
    if status == "visited":
        return C_FACE_VISITED
    if status == "settling":
        return C_FACE_SETTLING
    if off_angle_deg is not None and off_angle_deg > 0.5:
        return C_FACE_DEGRADED
    return C_FACE_READY


def log_face_views(faces: Iterable[tuple]) -> None:
    """Face-evidence view candidates (Task 6), one small arrow per face.

    `faces`: iterable of `(name, T_base_cam (4x4), status, off_angle_deg_or_
    None)` — exactly a step's `ViewState.candidates` whose `address ==
    ["face", name]`, unpacked by the caller (`show.py`). Logged ALL under
    `FACES` = "/world/faces/<name>" — its OWN root, never mixed into `CAMS`
    (Anton 2026-09-14: face candidates must be a group a reviewer can toggle
    on their own). Re-logged at every step that carries them, so a face's
    dot tracks its status (settling -> ready -> visited) across the scrub
    the same way a cell candidate would if this viewer drew those too.

    A no-op for an empty `faces` (old runs, pre-gate steps, runs with no
    boxguard fit): the loop below simply does not execute and nothing is
    logged, so the entity never appears in the tree for those runs.
    """
    for name, T_base_cam, status, off_angle_deg in faces:
        T = np.asarray(T_base_cam, dtype=float)
        pos, boresight = T[:3, 3], T[:3, 0]      # +X is the boresight (face_cam_pose)
        color = _face_color(status, off_angle_deg)
        label = (str(name) if off_angle_deg is None
                 else f"{name} {off_angle_deg:.0f}deg")
        rr.log(f"{FACES}/{_leaf(str(name))}", rr.Arrows3D(
            origins=[pos], vectors=[boresight * FACE_ARROW_M],
            colors=[color], radii=FACE_ARROW_M * 0.12,
            labels=[label], show_labels=True))


def clear_face_views() -> None:
    """Retire every face-view arrow at the current step (I2: gate regression).

    A step's `ViewState` carrying zero face candidates means either the
    stability gate just closed (the box moved; every unvisited face fell
    back to `settling`) or no boxguard has ever been fitted — either way
    there is nothing honest to draw right now. Without this, Rerun's
    latest-at semantics keep showing the arrows logged at the box's OLD
    pose from an earlier step, forever, across the whole rest of the scrub.
    One recursive clear on the root retires the whole subtree, the same
    `Clear(recursive=True)` primitive `clear_primitives` uses per-leaf.
    """
    rr.log(FACES, rr.Clear(recursive=True))


def log_active_view(intr: Mapping | None, image_path: str | Path) -> None:
    """Step k's rgb (upright, as stored) on ONE stable entity path.

    The standard rerun pattern for "the image of the current step": the
    right-hand pane is a `Spatial2DView` rooted at `ACTIVE`, this is
    re-logged every step, and latest-at makes the pane track the slider.
    The pinhole rides along (when the run has intrinsics) so the view
    frames the image at its true aspect.
    """
    if intr is not None:
        rr.log(ACTIVE, rr.Pinhole(image_from_camera=_K(intr),
                                  resolution=(intr["width"], intr["height"])))
    rr.log(f"{ACTIVE}/rgb", rr.EncodedImage(path=str(image_path)))


def log_step_ply(step_id: int, ply_path: str | Path) -> None:
    """A step's own single-view cloud.ply, `/world/step_cloud/<id>`.

    Legacy reduced-path data, kept logged but hidden by default: these
    accumulate over the scrub (each is logged at its step and persists),
    which is exactly what the per-step `CLOUD` entity is for — so they are
    an eye-click, not the default picture.
    """
    rr.log(f"{STEP_PLY}/{int(step_id):03d}", rr.Asset3D(path=str(ply_path)))


def _log_preview_quad(path: str, intr: Mapping,
                      image_path: str | Path) -> None:
    """The view's rgb as a textured quad on its own image plane.

    NOT `EncodedImage` under the pinhole, which is the obvious way to do
    this, and it is a mesh instead to kill the workspace's one red badge.
    A 2D visualiser in a 3D view needs its pinhole resolvable AT THE
    CURRENT TIME; a camera whose step has not arrived has no pinhole yet;
    so the view reported "2D visualizers require a pinhole ancestor to be
    shown in a 3D view" once per future camera — measured, 29 of 30 at
    step 0 and zero at the last step. A mesh is not a 2D visualiser and
    asks nobody for a pinhole, so the badge is simply gone.

    (Two other routes were measured and rejected on the way here. Putting
    the previews in a parallel `/world/preview` tree does give a
    previews-only toggle, but Anton wants ONE switch that leaves no camera
    anything on screen, and a second tree is a second click. Hiding a
    duplicated frustum with `color=(0,0,0,0)` or `line_width=0` does not
    work: the alpha is ignored and the lines come out BLACK, and the
    zero width still draws.)

    Corners are deprojected through the real intrinsics at `FRUSTUM_M` in
    CAMERA coordinates, so the parent's `Transform3D` lands the quad
    exactly on the cone the sibling pinhole draws.
    """
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return
    # Half resolution: the preview is ~120 mm wide on screen, nobody reads
    # its pixels, and an albedo texture is stored RAW — full res would put
    # 1.2 MB per view into the .rrd for no visible gain.
    img = cv2.cvtColor(cv2.resize(img, (intr["width"] // 2, intr["height"] // 2),
                                  interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB)
    W, H = float(intr["width"]), float(intr["height"])
    d = FRUSTUM_M

    def corner(u, v):
        return np.array([(u - intr["ppx"]) / intr["fx"] * d,
                         (v - intr["ppy"]) / intr["fy"] * d, d])

    # RDF: +x right, +y down, +z forward — so (0,0) is the TOP-LEFT pixel.
    cam = np.stack([corner(0, 0), corner(W, 0), corner(W, H), corner(0, H)])
    rr.log(f"{path}/preview",
           rr.Mesh3D(vertex_positions=cam.astype(np.float32),
                     triangle_indices=np.array([[0, 1, 2], [0, 2, 3]],
                                               dtype=np.uint32),
                     vertex_texcoords=np.array([[0, 0], [1, 0], [1, 1],
                                                [0, 1]], dtype=np.float32),
                     albedo_texture=img))


def log_overlay(stored: np.ndarray, replayed: np.ndarray) -> None:
    """Story 5, as two static channels in two colours.

    Static, not per-step: both are statements about the END of the run, and
    putting them on the slider would invite reading a step-7 stored cloud
    that never existed. Default-hidden by the blueprint; switching the
    subtree on is the whole interaction.
    """
    stored = np.asarray(stored, dtype=np.float32).reshape(-1, 3)
    replayed = np.asarray(replayed, dtype=np.float32).reshape(-1, 3)
    rr.log(OVERLAY_STORED,
           rr.Points3D(stored, colors=C_STORED, radii=R_OVERLAY * 1.6),
           static=True)
    rr.log(OVERLAY_REPLAYED,
           rr.Points3D(replayed, colors=C_REPLAYED, radii=R_OVERLAY),
           static=True)


# ------------------------------------------------------- THE PRIMITIVE CONTRACT
def log_primitives(primitives: Iterable[Mapping]) -> list[str]:
    """Render a fitter's output for the current step. Returns entity paths.

    This is the interface the spec calls load-bearing, so it is generous
    about input and strict about output: `pose` may be a 4x4, a
    `(center, R)` pair, or absent (identity); `params` keys have aliases
    where a fitter would plausibly use another word. Anything it cannot
    interpret raises here, in the producer, rather than logging a silently
    wrong shape into a file Anton will trust.

    The returned paths are what the caller CLEARS when a label stops being
    fitted (`clear_primitives`). This matters more than it looks: Rerun is
    latest-at, so a cylinder logged at step 12 and never again is still on
    screen at step 29 — it would look like a fit that was never made. A
    producer that walks steps in order must therefore diff these path sets
    step to step. `workspace.py` does.
    """
    paths = []
    for p in primitives:
        method, kind = p["method"], p["kind"]
        color = method_color(method)
        center, R = _pose(p.get("pose"))
        quat = _quat_from_mat(R)
        label = _leaf(p.get("label", kind))
        # One entity per (method, label) keeps show/hide at the granularity a
        # reviewer wants: kill one hallucinated plane, keep the other five.
        path = f"{PRIM}/{method}/{label}"
        # The hover/selection label carries the two per-primitive numbers the
        # contract records, so a shape that looks wrong can be interrogated
        # without leaving the viewer for the JSON.
        text = _label_text(p, label)
        par = p.get("params", {})
        if kind == "box":
            ext = np.asarray(par.get("extents", par.get("size")),
                             dtype=float).reshape(3)
            rr.log(path, rr.Boxes3D(
                centers=[center], half_sizes=[ext / 2.0],
                quaternions=[quat], colors=[color],
                fill_mode="MajorWireframe", radii=0.0012,
                labels=[text], show_labels=False))
        elif kind == "plane_patch":
            # A bounded rectangle, not an infinite plane: the extents are the
            # patch the fitter actually claims, and drawing more than that is
            # the exact hallucination this review is meant to catch. Solid,
            # because a wireframe rectangle seen edge-on is a line and the
            # question ("does this face hug the surface") needs a surface.
            ext = np.asarray(par.get("extents", par.get("size")),
                             dtype=float).reshape(2)
            rr.log(path, rr.Boxes3D(
                centers=[center],
                half_sizes=[(ext[0] / 2.0, ext[1] / 2.0, PLANE_T / 2.0)],
                quaternions=[quat], colors=[color], fill_mode="Solid",
                labels=[text], show_labels=False))
        elif kind == "cylinder":
            rr.log(path, rr.Cylinders3D(
                lengths=[float(par.get("length", par.get("height")))],
                radii=[float(par["radius"])],
                centers=[center], quaternions=[quat],
                colors=[color], fill_mode="MajorWireframe",
                line_radii=0.0012, labels=[text],
                show_labels=False))
        elif kind == "superquadric":
            V, F, N = superquadric_mesh(par["scale"], par["eps"])
            V = V @ R.T + center
            N = N @ R.T
            rr.log(path, rr.Mesh3D(
                vertex_positions=V.astype(np.float32),
                triangle_indices=F, vertex_normals=N.astype(np.float32),
                albedo_factor=(*color, SQ_ALPHA)))
        elif kind == "mesh":
            V = np.asarray(par.get("vertices", par.get("verts")),
                           dtype=np.float32).reshape(-1, 3)
            F = par.get("faces", par.get("triangles"))
            V = V @ R.T + center
            rr.log(path, rr.Mesh3D(
                vertex_positions=V,
                triangle_indices=(None if F is None else
                                  np.asarray(F, dtype=np.uint32)),
                albedo_factor=color))
        else:
            raise ValueError(f"unknown primitive kind {kind!r} — the closed "
                             f"set is box | cylinder | plane_patch | "
                             f"superquadric | mesh")
        paths.append(path)
    return paths


def clear_primitives(paths: Iterable[str]) -> None:
    """Retire entity paths at the current step (see `log_primitives`)."""
    for path in paths:
        rr.log(path, rr.Clear(recursive=True))


def open_method_channel(method: str) -> str:
    """Declare a fitter's channel so it exists in the tree even if it never
    draws anything.

    `planar` abstains on every step of two of the three runs. Without this
    marker its subtree simply would not be in the entity tree, and "the
    method refused" would be indistinguishable from "the method was never
    run" — which is the one thing an abstain-first contract must not lose.
    An empty point cloud renders nothing and costs nothing.
    """
    path = f"{PRIM}/{method}"
    rr.log(path, rr.Points3D(np.zeros((0, 3), dtype=np.float32)), static=True)
    return path


def log_decomposition_meta(method: str, **fields) -> None:
    """The non-geometric half of a `Decomposition`, on the method entity.

    `version`, `abstain`, `reason`, `coverage`, `residual_mm` have no shape
    to draw, but they are what tells a reviewer whether an empty channel is
    a refusal or a hole. As components on `/world/prim/<method>` they show
    up in the selection panel for the step under the slider, and no
    visualiser tries to draw them.
    """
    rr.log(f"{PRIM}/{method}", rr.AnyValues(**fields))


def superquadric_mesh(scale, eps, res: tuple[int, int] = SQ_RES):
    """Triangulate the EMS superquadric `(scale, eps)` into (V, F, N).

    The standard parametrisation, in the fitter's own convention: `scale`
    is SEMI-axes, `eps = (e1, e2)` are the shape exponents, and the surface
    is

        x = a1 c(eta)^e1 c(omega)^e2
        y = a2 c(eta)^e1 s(omega)^e2
        z = a3 s(eta)^e1

    with signed powers (`_spow`), eta in [-pi/2, pi/2] and omega wrapped
    over [-pi, pi). Vertices are in the primitive's own frame; the caller
    applies the pose.

    Exponents are clamped at 0.02: EMS returns values that get arbitrarily
    close to 0 for a hard-edged box, the signed power then collapses a
    whole quad row onto an edge, and the mesh grows zero-area triangles
    whose normals are NaN. 0.02 is already visually a sharp edge at our
    scale.
    """
    n_u, n_v = int(res[0]), int(res[1])
    a = np.asarray(scale, dtype=float).reshape(3)
    e1, e2 = (float(np.clip(v, 0.02, 2.0)) for v in np.asarray(eps).reshape(2))
    eta = np.linspace(-np.pi / 2, np.pi / 2, n_v)
    omega = np.linspace(-np.pi, np.pi, n_u, endpoint=False)
    E, O = np.meshgrid(eta, omega, indexing="ij")
    V = np.stack([a[0] * _spow(np.cos(E), e1) * _spow(np.cos(O), e2),
                  a[1] * _spow(np.cos(E), e1) * _spow(np.sin(O), e2),
                  a[2] * _spow(np.sin(E), e1)], axis=-1).reshape(-1, 3)

    i = np.arange(n_v - 1)[:, None]
    j = np.arange(n_u)[None, :]
    jn = (j + 1) % n_u                      # wrap the seam shut
    v00, v01 = i * n_u + j, i * n_u + jn
    v10, v11 = (i + 1) * n_u + j, (i + 1) * n_u + jn
    F = np.concatenate([np.stack([v00, v10, v11], -1).reshape(-1, 3),
                        np.stack([v00, v11, v01], -1).reshape(-1, 3)]
                       ).astype(np.uint32)

    # Vertex normals from area-weighted face normals. The analytic normal is
    # available but goes singular exactly where the exponents are small,
    # which is where the mesh needs shading most.
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for c in range(3):
        np.add.at(N, F[:, c], fn)
    n = np.linalg.norm(N, axis=1, keepdims=True)
    N = np.divide(N, n, out=np.zeros_like(N), where=n > 1e-12)
    return V, F, N


def _spow(x: np.ndarray, e: float) -> np.ndarray:
    """|x|^e with the sign of x — the superquadric's signed power."""
    return np.sign(x) * np.abs(x) ** e


def _leaf(label: str) -> str:
    """A label turned into ONE entity-path part.

    A `/` in a fitter's label would silently nest the primitive a level
    deeper and take it out of its method's subtree — i.e. out of the one
    toggle the reviewer uses.
    """
    s = str(label).strip().replace("/", "_") or "unnamed"
    return s


def _label_text(p: Mapping, label: str) -> str:
    bits = [label]
    if p.get("residual_mm"):
        bits.append(f"res {float(p['residual_mm']):.1f} mm")
    if p.get("support"):
        bits.append(f"n {int(p['support'])}")
    return "  ".join(bits)


# ------------------------------------------------------------- chain panel
def log_chain(slots: Mapping[str, str | Path]) -> None:
    """Story 4: the identity chain for the current step, one image per slot.

    Slot names become entity leaves and view titles, so a run with the
    chain trio and a run on the rgb+mask fallback are self-describing
    rather than silently different.
    """
    for slot, img in slots.items():
        rr.log(f"{CHAIN}/{slot}", rr.EncodedImage(path=str(img)))


# ------------------------------------------------------------------ blueprint
def blueprint(run: str,
              hidden: Sequence[str] = (OVERLAY,)) -> rrb.Blueprint:
    """The arranged workspace, saved into the file (story 1).

    Exactly TWO panes (Anton, 2026-09-10): the 3D scene on the left —
    frusta, the step-k cloud, the step-k camera lit in `C_CAM_ACTIVE` —
    and on the right one 2D view rooted at `ACTIVE`, showing only the
    image captured at the step under the slider. One time panel at the
    bottom pinned to the `step` timeline. `auto_views=False` so Rerun does
    not helpfully add a third view nobody asked for and shift every
    coordinate a screenshot test depends on.

    The chain images (`/chain`) are still logged — story 4's data is all
    there — but out of the layout: neither pane's origin reaches them, so
    they cost nothing on screen and stay one manual view away.

    Default visibility: cloud, cameras and every primitive method on; the
    stored/replayed overlay off, present in the tree with its eye closed
    so turning it on is one click.

    The one-click groups the tree gives a reviewer, by design:
    `/world/cam` — EVERYTHING the cameras draw: frusta, axis gizmos and
    rgb planes together. By the end of a scan those form a solid dome over
    the cup, and one eye-click has to leave nothing of it behind (Anton,
    2026-09-03). Then `/world/prim/<method>` (one fitter at a time) and
    `/world/overlay` (story 5).
    """
    scene = rrb.Spatial3DView(
        name=f"{run} — scene",
        origin=WORLD,
        contents="$origin/**",
        background=(18, 20, 26),
        line_grid=rrb.LineGrid3D(visible=True, spacing=0.05),
        overrides={h: rrb.EntityBehavior(visible=False) for h in hidden},
    )
    active = rrb.Spatial2DView(name="view @ step", origin=ACTIVE)
    return rrb.Blueprint(
        rrb.Horizontal(scene, active, column_shares=[2.4, 1.0]),
        # Paused, not playing. A file that starts animating the moment it
        # opens costs the reviewer a click before he can read anything, and
        # it makes every screenshot of the workspace a different screenshot.
        # `loop_mode="Off"` for the same reason: nothing moves unless the
        # slider is dragged.
        rrb.TimePanel(state="expanded", timeline=TIMELINE,
                      play_state="Paused", loop_mode="Off"),
        rrb.BlueprintPanel(state="expanded"),
        rrb.SelectionPanel(state="collapsed"),
        auto_views=False,
        auto_layout=False,
    )


# --------------------------------------------------------------- small math
def _pose(pose) -> tuple[np.ndarray, np.ndarray]:
    """(center, 3x3 rotation) out of any of the three accepted spellings."""
    if pose is None:
        return np.zeros(3), np.eye(3)
    if isinstance(pose, (tuple, list)) and len(pose) == 2:
        c, R = pose
        return (np.asarray(c, dtype=float).reshape(3),
                np.asarray(R, dtype=float).reshape(3, 3))
    T = np.asarray(pose, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"pose must be 4x4 or (center, R), got {T.shape}")
    return T[:3, 3].copy(), T[:3, :3].copy()


def _quat_from_mat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> xyzw, the order Rerun's `quaternions=` expects."""
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w, x, y, z = (0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
        q = [0.0, 0.0, 0.0]
        q[i], q[j], q[k] = 0.25 * s, (R[j, i] + R[i, j]) / s, \
            (R[k, i] + R[i, k]) / s
        w, (x, y, z) = (R[k, j] - R[j, k]) / s, q
    return (float(x), float(y), float(z), float(w))


def _quat_from_z(n: np.ndarray) -> tuple[float, float, float, float]:
    """Shortest rotation taking local +Z onto `n` — for slabs and cylinders."""
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, n)
    c = float(np.dot(z, n))
    if np.linalg.norm(v) < 1e-12:
        return (0.0, 0.0, 0.0, 1.0) if c > 0 else (1.0, 0.0, 0.0, 0.0)
    s = np.sqrt((1.0 + c) * 2.0)
    return (float(v[0] / s), float(v[1] / s), float(v[2] / s), float(s / 2.0))


# ---- viewplan additions (additive only) ------------------------------
# Six small functions for `vbti/inspection/viewplan/workspace.py`
# (docs/research/view-planning/poc/structure.md). Nothing above this line
# is touched: viewplan gets its own subtree, its own colour, and its own
# blueprint builder, so the record session's schema and this one cannot
# collide even though they may both be open on the same run.
VIEWPLAN = "/world/plan"        # new subtree: one eye-click kills the plan
VP_MESH = VIEWPLAN + "/mesh"    # this step's reconstructed mesh
VP_COVER = VIEWPLAN + "/coverage"  # samples coloured by measurability
VP_CAMS = VIEWPLAN + "/cam"     # /world/plan/cam/<view_id> — planned views
RENDERS = "/renders"            # /renders/<slot> — the 2D grid, milestone-only

#: Magenta — checked against every colour already in this module (neutral
#: cloud grey, fresh orange, table slate, stored/replayed blue/pink, cam
#: cyan, the three pinned fit hues): farthest in hue from all of them, so a
#: planned frustum is never mistaken for a recorded one on the same screen.
C_PLANNED = (255, 96, 210)

VP_PATCHES = VIEWPLAN + "/patches"  # samples coloured by OWNING view

#: One colour per planned view id, shared by that view's frustum and the
#: samples it owns (`covered_by` in the npz), so "which camera sees this
#: patch" is answered by colour alone. Qualitative 12-hue set tuned for the
#: dark viewer background; red is NOT in it (red means "uncovered" in the
#: coverage channel) and uncovered samples here are dim grey, so neither
#: channel's vocabulary leaks into the other. Wraps past 12.
VP_VIEW_COLORS = (
    (86, 180, 233), (230, 159, 0), (0, 158, 115), (240, 228, 66),
    (204, 121, 167), (213, 94, 0), (100, 143, 255), (220, 38, 127),
    (255, 176, 0), (120, 94, 240), (44, 160, 44), (23, 190, 207),
)
C_UNOWNED = (90, 90, 95)


def viewplan_view_color(view_id: int) -> tuple:
    """The pinned colour of planned view `view_id` (wraps past the palette)."""
    return VP_VIEW_COLORS[int(view_id) % len(VP_VIEW_COLORS)]


def log_viewplan_patches(points: np.ndarray, covered_by: np.ndarray
                         ) -> list[str]:
    """One entity PER OWNING VIEW: `/world/plan/patches/view_NN` (+ `unowned`).

    The complement of `log_viewplan_coverage`: coverage answers "how well is
    this patch seen", this answers "by WHICH camera". Each view's territory
    is its own entity so the reviewer can reveal them one at a time with the
    eye icons — `viewplan_blueprint(patch_paths=...)` hides them all by
    default for exactly that workflow. Colour matches the view's frustum
    (`viewplan_view_color`). `covered_by` is the npz column: view id per
    sample, -1 = unowned (dim grey, not red — red belongs to coverage).
    RETURNS the logged entity paths so the caller can retire stale ones
    when a later step's plan has fewer views.
    """
    P = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    owner = np.asarray(covered_by, dtype=int).reshape(-1)
    logged: list[str] = []
    for vid in np.unique(owner):
        sel = owner == vid
        if not sel.any():
            continue
        if vid < 0:
            path, color = f"{VP_PATCHES}/unowned", C_UNOWNED
        else:
            path, color = (f"{VP_PATCHES}/view_{int(vid):02d}",
                           viewplan_view_color(int(vid)))
        rr.log(path, rr.Points3D(P[sel], colors=np.tile(
            np.array(color, dtype=np.uint8), (int(sel.sum()), 1)),
            radii=R_CLOUD * 1.3))
        logged.append(path)
    return logged


def log_viewplan_mesh(vertices: np.ndarray, faces: np.ndarray,
                      vertex_normals: np.ndarray | None = None) -> None:
    """This step's reconstructed mesh, re-logged every step (it changes).

    Neutral translucent albedo — the mesh is the surface the plan reasons
    about, not a result to be admired, so it must not compete with the
    coverage colouring drawn over it (`log_viewplan_coverage`) or the
    accent magenta of the planned frusta.
    """
    V = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    F = np.asarray(faces, dtype=np.uint32).reshape(-1, 3)
    kw = {}
    if vertex_normals is not None:
        kw["vertex_normals"] = np.asarray(vertex_normals,
                                          dtype=np.float32).reshape(-1, 3)
    rr.log(VP_MESH, rr.Mesh3D(vertex_positions=V, triangle_indices=F,
                              albedo_factor=(200, 205, 215, 90), **kw))


def log_viewplan_coverage(points: np.ndarray, best_cos: np.ndarray,
                          cutoff_cos: float) -> None:
    """Surface samples coloured by measurability: red = uncovered, green
    ramp = covered, brighter with better (closer to orthogonal) incidence.

    `best_cos` is the per-sample best incidence cosine over the SELECTED
    views (0 for a sample no selected view reaches), `cutoff_cos` the
    variant's incidence-gate cosine — the same split `coverage_area_frac`
    is computed from, so the picture and the metric never disagree.
    """
    P = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    q = np.asarray(best_cos, dtype=float).reshape(-1)
    cutoff = float(cutoff_cos)
    covered = q >= cutoff
    colors = np.tile(np.array([214, 64, 64], dtype=np.uint8), (len(P), 1))
    if covered.any():
        span = max(1.0 - cutoff, 1e-9)
        t = np.clip((q[covered] - cutoff) / span, 0.0, 1.0)
        colors[covered] = np.stack([
            (70 - 30 * t) * 1.0,
            (140 + 105 * t),
            (50 - 30 * t),
        ], axis=1).clip(0, 255).astype(np.uint8)
    rr.log(VP_COVER, rr.Points3D(P, colors=colors, radii=R_CLOUD * 1.3))


def log_viewplan_camera(view_id: int, T_base_cam: np.ndarray, intr: Mapping,
                        label: str = "", image_path: str | Path | None = None,
                        color: tuple | None = None) -> str:
    """A planned view's frustum at `/world/plan/cam/<view_id>`.

    Default magenta; pass `color=viewplan_view_color(view_id)` to paint the
    frustum with its territory's patch colour instead.

    Same three-level pattern as `log_camera` (Transform3D + child Pinhole,
    optional textured preview via `_log_preview_quad`), so a planned camera
    and a recorded one read the same way in the tree and differ only by
    colour and subtree. RETURNS the entity path so the caller can diff it
    against last step's set and `clear_primitives` whatever is retired —
    plans change size step to step, unlike recorded frusta which are
    logged once and never move.
    """
    T = np.asarray(T_base_cam, dtype=float)
    K = np.array([[intr["fx"], 0.0, intr["ppx"]],
                  [0.0, intr["fy"], intr["ppy"]],
                  [0.0, 0.0, 1.0]], dtype=float)
    path = f"{VP_CAMS}/{int(view_id):02d}"
    rr.log(path, rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
    rr.log(f"{path}/frustum",
           rr.Pinhole(image_from_camera=K,
                      resolution=(intr["width"], intr["height"]),
                      camera_xyz=rr.ViewCoordinates.RDF,
                      image_plane_distance=FRUSTUM_M,
                      color=C_PLANNED if color is None else tuple(color)))
    if label:
        rr.log(path, rr.AnyValues(label=str(label)))
    if image_path is not None:
        _log_preview_quad(path, intr, image_path)
    return path


def log_viewplan_meta(variant: str, **fields) -> None:
    """The non-geometric half of a `StepPlan`, on the `/world/plan` entity.

    Same `log_decomposition_meta` pattern: abstain / reason / n_views /
    coverage have no shape to draw, but without them an empty step (no
    views, or an honest abstain) is indistinguishable from a step nobody
    ever planned.
    """
    rr.log(VIEWPLAN, rr.AnyValues(variant=str(variant), **fields))


def log_viewplan_render(slot: str, image_path: str | Path) -> None:
    """One milestone-step render into the 2D grid, `/renders/<slot>`.

    Slot names are stable view-id slots (`view_00`, `view_01`, ...), not
    per-step paths, so `viewplan_blueprint`'s grid has a fixed shape across
    the whole timeline and latest-at makes each cell simply show the most
    recent render logged for that slot — exactly the `log_chain` pattern.
    """
    rr.log(f"{RENDERS}/{slot}", rr.EncodedImage(path=str(image_path)))


VP_SAMPLES = VIEWPLAN + "/samples"  # probe subset of raw surface samples


def log_viewplan_sample_probes(points: np.ndarray, normals: np.ndarray,
                               ids: np.ndarray, arrow_m: float = 0.10,
                               cloud: np.ndarray | None = None,
                               patch_radius_m: float = 0.012) -> list[str]:
    """N raw sampled patches, one entity each: `/world/plan/samples/sNN`.

    The debug view of the sampler itself — each probe is one of the 3000
    area-uniform samples, drawn as the PATCH it stands on: every cloud
    point within `patch_radius_m` of the sample tinted the probe's colour
    (default 12 mm = `fit/common.py:NORMAL_RADIUS_M`, the radius whose PCA
    produced the normal — so the coloured disk is literally the set of
    points that VOTED for the arrow), plus the normal as an arrow of length
    `arrow_m`. With `arrow_m` = the variant's standoff the arrow TIP is
    exactly where that sample's candidate camera would sit and the
    candidate boresight is the arrow reversed — the first two links of the
    algorithm chain, drawn as geometry. RETURNS logged paths for retirement.
    """
    P = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    N = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    C = None if cloud is None else np.asarray(cloud, dtype=np.float32).reshape(-1, 3)
    logged: list[str] = []
    for j, i in enumerate(np.asarray(ids, dtype=int).reshape(-1)):
        path = f"{VP_SAMPLES}/s{j:02d}"
        color = viewplan_view_color(j)
        rr.log(path, rr.Points3D(P[i:i + 1], colors=[color],
                                 radii=R_CLOUD * 2.0))
        if C is not None and len(C):
            near = C[np.linalg.norm(C - P[i], axis=1) <= float(patch_radius_m)]
            if len(near):
                rr.log(f"{path}/patch", rr.Points3D(
                    near, colors=np.tile(np.array(color, dtype=np.uint8),
                                         (len(near), 1)),
                    radii=R_CLOUD * 1.4))
        rr.log(f"{path}/normal",
               rr.Arrows3D(origins=P[i:i + 1],
                           vectors=(N[i:i + 1] * float(arrow_m)),
                           colors=[color], radii=R_CLOUD * 0.5))
        rr.log(path, rr.AnyValues(sample_index=int(i)))
        logged.append(path)
    return logged


def viewplan_blueprint(run: str, variant: str,
                       render_slots: Sequence[str],
                       patch_paths: Sequence[str] = ()) -> rrb.Blueprint:
    """3D plan view left, milestone-render grid right, `step` timeline.

    Mirrors `blueprint()`'s shape (paused, no auto views, one time panel)
    rather than reusing it directly: the right-hand panel here is a
    RENDER grid keyed by view slot, not the fixed 2x2 identity chain, so
    the two blueprints are siblings in style, not one function with a
    branch — a viewplan reviewer never needs the chain panel and a record
    reviewer never needs 12 empty render cells.

    `patch_paths`: per-view patch entities to hide by default (Anton's
    review gesture is revealing territories one eye-click at a time, so
    they must START hidden — the override is blueprint state, exactly
    where per-entity visibility lives in rerun).
    """
    grid = rrb.Grid(
        contents=[rrb.Spatial2DView(name=s, origin=f"{RENDERS}/{s}")
                  for s in render_slots],
        grid_columns=3, name="renders")
    scene = rrb.Spatial3DView(
        name=f"{run} — {variant} plan",
        origin=WORLD,
        contents="$origin/**",
        background=(18, 20, 26),
        line_grid=rrb.LineGrid3D(visible=True, spacing=0.05),
        overrides={p: rrb.EntityBehavior(visible=False)
                   for p in patch_paths},
    )
    return rrb.Blueprint(
        rrb.Horizontal(scene, grid, column_shares=[3.0, 1.4]),
        rrb.TimePanel(state="expanded", timeline=TIMELINE,
                      play_state="Paused", loop_mode="Off"),
        rrb.BlueprintPanel(state="expanded"),
        rrb.SelectionPanel(state="collapsed"),
        auto_views=False,
        auto_layout=False,
    )
