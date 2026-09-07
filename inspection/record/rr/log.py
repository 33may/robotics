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
`inspection/fit/contract.py`::

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
PRIM = "/world/prim"            # /world/prim/<method>/<label>   (per step)
OVERLAY = "/world/overlay"      # stored vs replayed final cloud (static, off)
OVERLAY_STORED = OVERLAY + "/stored"
OVERLAY_REPLAYED = OVERLAY + "/replayed"
CHAIN = "/chain"                # /chain/<slot>, 2D images       (per step)

#: The accumulator voxel is 3 mm (`inspection/cell/geometry.py:46`), so a
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

#: Frustum length in metres. At the colour intrinsics this puts a ~120 mm
#: image plane at the end of each cone — readable, and small enough that
#: thirty of them at the end of a scrub still leave the 100 mm cup visible.
#: They live under one entity (`/world/cam`), so one eye-click in the
#: blueprint tree takes the whole thicket off (Anton, 2026-09-03).
FRUSTUM_M = 0.06

#: Colours are per METHOD and deterministic, so "the blue boxes" means the
#: same fitter in every run and in every screenshot. Hand-pinned for the
#: bank methods that exist (`inspection/fit`); hashed for the ones that do
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


def log_cloud(points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """The fused cloud as of the current step (story 2: it grows).

    Two entities, one cloud: `CLOUD` is the neutral backdrop the fits are
    judged against; `CLOUD_RGB` is the same points in their captured
    colours (`fused_colors.npy` / colored step PLYs), hidden by default —
    the eye icon IS the colour toggle. A step without colours logs an
    empty `CLOUD_RGB` so a scrub never shows a stale colouring.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rr.log(CLOUD, rr.Points3D(points, colors=C_CLOUD, radii=R_CLOUD))
    if colors is not None and len(colors) == len(points):
        rr.log(CLOUD_RGB, rr.Points3D(points, colors=np.asarray(colors),
                                      radii=R_CLOUD))
    else:
        rr.log(CLOUD_RGB, rr.Points3D(np.zeros((0, 3), dtype=np.float32)))


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


def log_camera(pose_id: int, T_base_cam: np.ndarray, intr: Mapping,
               image_path: str | Path | None = None,
               label: str | None = None) -> None:
    """A frustum at the pose this view actually shot from, wearing its rgb.

    Logged at the current step and never again, so it appears when its
    view is fused and persists for the rest of the scrub (story 2).

    `intr` is a session.json intrinsics block (`fx, fy, ppx, ppy, width,
    height`). `camera_xyz=RDF` is the OpenCV convention the cell's
    `deproject` uses (`geometry.py`), so the frustum points where the
    depth actually went.
    """
    T = np.asarray(T_base_cam, dtype=float)
    K = np.array([[intr["fx"], 0.0, intr["ppx"]],
                  [0.0, intr["fy"], intr["ppy"]],
                  [0.0, 0.0, 1.0]], dtype=float)
    path = f"{CAMS}/{pose_id:03d}"
    # Three levels, and each one earns its place. The POSE is on the camera
    # entity, so hiding `/world/cam` takes the axis gizmos with it. The
    # PINHOLE is on a child rather than on the camera itself, because a
    # pinhole makes its children image-plane space and the preview has to
    # stay a sibling in metres. The PREVIEW is a textured mesh in camera
    # coordinates, which the parent transform then places.
    rr.log(path, rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
    rr.log(f"{path}/frustum",
           rr.Pinhole(image_from_camera=K,
                      resolution=(intr["width"], intr["height"]),
                      camera_xyz=rr.ViewCoordinates.RDF,
                      image_plane_distance=FRUSTUM_M,
                      color=C_CAM))
    if image_path is not None:
        _log_preview_quad(path, intr, image_path)


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
def blueprint(run: str, chain_slots: Sequence[str],
              hidden: Sequence[str] = (OVERLAY,)) -> rrb.Blueprint:
    """The arranged workspace, saved into the file (story 1).

    3D dominant on the left, chain stacked on the right, one time panel at
    the bottom pinned to the `step` timeline. `auto_views=False` so Rerun
    does not helpfully add a fourth view nobody asked for and shift every
    coordinate a screenshot test depends on.

    Default visibility is the spec's: cloud, cameras, chain and every
    primitive method on; the stored/replayed overlay off, present in the
    tree with its eye closed so turning it on is one click.

    The one-click groups the tree gives a reviewer, by design:
    `/world/cam` — EVERYTHING the cameras draw: frusta, axis gizmos and
    rgb planes together. By the end of a scan those form a solid dome over
    the cup, and one eye-click has to leave nothing of it behind (Anton,
    2026-09-03). Then `/world/prim/<method>` (one fitter at a time) and
    `/world/overlay` (story 5).
    """
    # 2x2, not a column (Anton, 2026-09-03). The chain is four faces of one
    # moment — what the camera saw, what was asked for, what came back, what
    # survived — and a stack of four letterboxed 848x480 frames wastes most
    # of the panel on black bars. A grid gives each of them real pixels and
    # puts them side by side, which is how they are compared. Runs with only
    # the rgb+mask fallback fill two cells of the same grid rather than
    # getting a different layout.
    chain = rrb.Grid(
        contents=[rrb.Spatial2DView(name=s, origin=f"{CHAIN}/{s}")
                  for s in chain_slots],
        grid_columns=2, name="chain")
    scene = rrb.Spatial3DView(
        name=f"{run} — scene",
        origin=WORLD,
        contents="$origin/**",
        background=(18, 20, 26),
        line_grid=rrb.LineGrid3D(visible=True, spacing=0.05),
        overrides={h: rrb.EntityBehavior(visible=False) for h in hidden},
    )
    return rrb.Blueprint(
        rrb.Horizontal(scene, chain, column_shares=[3.2, 1.0]),
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
