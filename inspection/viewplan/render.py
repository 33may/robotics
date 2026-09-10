"""Offscreen evidence: what a planned camera would actually see.

No OpenGL, no pyrender (not installed, and a GPU context in a batch job is a
liability anyway). Every pixel here comes from the same
`o3d.t.geometry.RaycastingScene` the planner uses for occlusion, shaded in
numpy — so the render and the plan cannot disagree about the geometry, and a
full 848x480 frame costs ~15 ms.

This module exists to make one claim falsifiable. A plan says "view 3 covers
these 412 patches". The render draws view 3 **and splats those 412 patch
centres into it**. If the dots do not land on visible, well-lit surface, the
claim is false and you can see that it is false. The same pass returns the
number as `framed_frac` (roadmap M10), so the check is mechanical too, not
only eyeballed.

The render path was validated against reality, not against taste: rays built
from the recorded colour intrinsics and the recorded `T_base_cam` of
2408-cup3 step 005, cast at the step-029 mesh, put the mesh's hit-pixel bbox
within ~10 px of the SAM mask box recorded from the real photograph.

    from inspection.viewplan import render
    img, stats = render.render_view(mesh, T_base_cam, intr, cover_ids=ids,
                                    samples=pts)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["render_view", "coverage_panel", "render_plan", "save_png",
           "scene_of", "RENDER_SCALE"]

#: Renders are evidence, not measurements — half resolution is 424x240,
#: plenty to see whether a view frames a cup, and 4x cheaper.
RENDER_SCALE = 0.5

BACKGROUND = np.array([18, 20, 26], dtype=np.uint8)      # dark slate
KEY_LIGHT = np.array([0.3, 0.3, 0.9])
#: Wash for pixels whose range falls outside the D405 window — a view that is
#: too close or too far is visible at a glance instead of hidden in a metric.
OUT_OF_WINDOW_TINT = np.array([1.0, 0.35, 0.30])
#: Claimed-coverage dots.
COVER_DOT = np.array([255, 170, 40], dtype=np.uint8)
UNCOVERED = np.array([220, 40, 40], dtype=np.uint8)
MESH_GREY = np.array([0.62, 0.64, 0.68])


def _o3d():
    import open3d as o3d
    return o3d


def scene_of(m):
    """RaycastingScene over one legacy TriangleMesh — built once, reused."""
    o3d = _o3d()
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    return s


def _scaled_intr(intr: Dict[str, float], scale: float):
    W = max(int(round(float(intr["width"]) * scale)), 8)
    H = max(int(round(float(intr["height"]) * scale)), 8)
    return {"fx": float(intr["fx"]) * scale, "fy": float(intr["fy"]) * scale,
            "ppx": float(intr["ppx"]) * scale, "ppy": float(intr["ppy"]) * scale,
            "width": W, "height": H}


def _cast(scene, T: np.ndarray, ik: Dict[str, float]):
    """(t_hit, normals, dirs) for a full frame from `T` — OpenCV axes."""
    o3d = _o3d()
    W, H = ik["width"], ik["height"]
    R, t = T[:3, :3], T[:3, 3]
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d_cam = np.stack([(u - ik["ppx"]) / ik["fx"],
                      (v - ik["ppy"]) / ik["fy"], np.ones_like(u)], axis=-1)
    d = d_cam @ R.T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    rays = np.concatenate([np.broadcast_to(t, d.shape), d], -1).astype(np.float32)
    ans = scene.cast_rays(o3d.core.Tensor(rays))
    return ans["t_hit"].numpy(), ans["primitive_normals"].numpy(), d


def _shade(t_hit, nrm, dirs, window: Optional[Sequence[float]] = None,
           albedo: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Lambert headlight, double-sided. Returns (uint8 image, hit mask).

    `abs()` on every dot product is deliberate: an open Poisson sheet has
    arbitrary winding, and a correct view must not turn black because the
    mesher wound a triangle the other way.
    """
    hit = np.isfinite(t_hit)
    k = KEY_LIGHT / np.linalg.norm(KEY_LIGHT)
    shade = (0.15
             + 0.55 * np.abs(np.einsum("hwi,hwi->hw", nrm, dirs))
             + 0.30 * np.abs(nrm @ k))
    base = MESH_GREY if albedo is None else albedo
    if base.ndim == 1:
        rgb = shade[..., None] * base[None, None, :]
    else:
        rgb = shade[..., None] * base
    if window is not None:
        lo, hi = float(window[0]), float(window[1])
        bad = hit & ((t_hit < lo) | (t_hit > hi))
        rgb[bad] = rgb[bad] * 0.45 + 0.55 * OUT_OF_WINDOW_TINT
    img = np.zeros(hit.shape + (3,), dtype=np.uint8)
    img[:] = BACKGROUND
    img[hit] = np.clip(rgb[hit] * 255.0, 0, 255).astype(np.uint8)
    return img, hit


def _project(points: np.ndarray, T: np.ndarray, ik: Dict[str, float]):
    """(u, v, z) in the scaled image, index-preserving (no filtering)."""
    Ti = np.linalg.inv(T)
    pc = (Ti[:3, :3] @ np.asarray(points, float).reshape(-1, 3).T).T + Ti[:3, 3]
    z = pc[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return (ik["ppx"] + ik["fx"] * pc[:, 0] / safe,
            ik["ppy"] + ik["fy"] * pc[:, 1] / safe, z)


def _depth_agrees(t_hit: np.ndarray, uu, vv, dist, tol: float) -> np.ndarray:
    """Does the render's own depth buffer show this patch at this distance?

    The buffer is sampled over the 3x3 pixel neighbourhood, not the single
    nearest pixel. That is quantisation accounting, not leniency: at half
    resolution one pixel spans ~0.5 mm on the object, and a patch on a
    silhouette or a 55 deg slope legitimately lands between two pixel
    centres. An actually occluded patch is metres-of-mesh away from the
    buffer value and 3x3 does not rescue it.
    """
    H, W = t_hit.shape
    x0 = np.round(uu).astype(int)
    y0 = np.round(vv).astype(int)
    best = np.full(len(x0), np.inf)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            x = np.clip(x0 + dx, 0, W - 1)
            y = np.clip(y0 + dy, 0, H - 1)
            b = t_hit[y, x]
            e = np.where(np.isfinite(b), np.abs(b - dist), np.inf)
            best = np.minimum(best, e)
    return best < tol


def _splat(img: np.ndarray, uu, vv, colour, radius: int = 2) -> None:
    H, W = img.shape[:2]
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            if du * du + dv * dv > radius * radius:
                continue
            x = np.round(uu + du).astype(int)
            y = np.round(vv + dv).astype(int)
            ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            img[y[ok], x[ok]] = colour


def render_view(mesh_or_scene, T_base_cam, intr: Dict[str, float], *,
                scale: float = RENDER_SCALE,
                cover_ids: Optional[Sequence[int]] = None,
                samples: Optional[np.ndarray] = None,
                window: Sequence[float] = (0.07, 0.20),
                depth_tol_m: float = 0.005
                ) -> Tuple[np.ndarray, Dict[str, float]]:
    """Shaded frame from one planned pose, plus the numbers it proves.

    Returns `(image (H,W,3) uint8, stats)`. Returning the stats alongside the
    image is a deliberate departure from a bare `-> np.ndarray`: `frame_fill`
    and `framed_frac` both fall out of the very cast that made the picture,
    and a second signature that recomputed them would be a second raycast and
    a chance for the two to disagree.

    stats: `frame_fill` (fraction of pixels on the mesh),
    `framed_frac` (M10 — fraction of the view's *claimed* covered samples
    that project inside the frame AND survive a depth test against this
    render's own depth buffer), `n_claimed`, `t_min`/`t_max` (the range the
    view actually stands at).
    """
    T = np.asarray(T_base_cam, dtype=float)
    ik = _scaled_intr(intr, scale)
    scene = mesh_or_scene if hasattr(mesh_or_scene, "cast_rays") \
        else scene_of(mesh_or_scene)
    t_hit, nrm, dirs = _cast(scene, T, ik)
    img, hit = _shade(t_hit, nrm, dirs, window=window)

    stats = {"frame_fill": float(hit.mean()),
             "framed_frac": 1.0, "n_claimed": 0,
             "t_min": float(t_hit[hit].min()) if hit.any() else 0.0,
             "t_max": float(t_hit[hit].max()) if hit.any() else 0.0}

    if cover_ids is not None and samples is not None and len(cover_ids):
        ids = np.asarray(cover_ids, dtype=int)
        Q = np.asarray(samples, dtype=float)[ids]
        uu, vv, z = _project(Q, T, ik)
        inside = ((z > 1e-3) & (uu >= 0) & (uu < ik["width"]) &
                  (vv >= 0) & (vv < ik["height"]))
        # depth test against the render's own buffer: the claimed patch must
        # be the thing this camera actually sees at that pixel
        good = inside.copy()
        if inside.any():
            dist = np.linalg.norm(Q[inside] - T[:3, 3], axis=1)
            good[inside] = _depth_agrees(t_hit, uu[inside], vv[inside], dist,
                                         depth_tol_m)
        stats["n_claimed"] = int(len(ids))
        stats["framed_frac"] = float(good.mean())
        _splat(img, uu[inside], vv[inside], COVER_DOT)
    return img, stats


def save_png(img: np.ndarray, path) -> Path:
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(img)).save(path)
    return path


# --- coverage panel --------------------------------------------------------
def _square_crop(img: np.ndarray, cx: float) -> np.ndarray:
    """Cut the 848x480 letterbox down to a square about the principal point."""
    H, W = img.shape[:2]
    if W <= H:
        return img
    x0 = int(np.clip(round(cx) - H // 2, 0, W - H))
    return img[:, x0:x0 + H]


def _orbit_poses(center: np.ndarray, radius: float) -> List[np.ndarray]:
    """Four fixed look-at poses around the object — same every step, so the
    panels of step 3 and step 29 are directly comparable."""
    from inspection.viewplan.contract import look_at_cv
    out = []
    for az in (45.0, 135.0, 225.0, 315.0):
        a, e = np.deg2rad(az), np.deg2rad(30.0)
        pos = center + radius * np.array([np.cos(e) * np.cos(a),
                                          np.cos(e) * np.sin(a), np.sin(e)])
        out.append(look_at_cv(pos, center - pos))
    return out


def coverage_panel(m, arrays: Dict[str, np.ndarray], out_png,
                   intr: Optional[Dict[str, float]] = None,
                   cutoff_deg: float = 60.0,
                   scale: float = RENDER_SCALE) -> Path:
    """2x2 orbit of the mesh with every sample coloured by achieved incidence.

    Uncovered patches are drawn in a screaming red so the eye finds the hole
    before it finds the metric; covered ones ramp green->yellow from the
    cutoff to head-on. One PNG per step, fixed cameras, so the sequence reads
    as a film of the plan closing.
    """
    from inspection.viewplan.planner import DEFAULT_INTR
    intr = dict(intr) if intr else dict(DEFAULT_INTR)
    P = np.asarray(arrays["points"], float)
    best = np.asarray(arrays["best_cos"], float)
    ik0 = _scaled_intr(intr, scale)
    centre = np.asarray(m.get_center(), float)
    ext = np.asarray(m.get_max_bound()) - np.asarray(m.get_min_bound())
    # stand back just far enough that the object's bounding sphere fills 85 %
    # of the (square-cropped) frame — the same framing at step 2 and step 29,
    # so the panel sequence reads as the plan closing and not as a zoom.
    half_tan = 0.5 * ik0["height"] / ik0["fy"]
    radius = float(0.5 * np.linalg.norm(ext)) / max(0.85 * half_tan, 1e-6)

    cc = float(np.cos(np.deg2rad(cutoff_deg)))
    tt = np.clip((best - cc) / max(1.0 - cc, 1e-9), 0.0, 1.0)
    col = np.empty((len(P), 3), dtype=np.uint8)
    col[:] = UNCOVERED
    ok = best >= cc
    # green (barely at the cutoff) -> yellow-white (head-on)
    col[ok] = np.stack([60 + 195 * tt[ok], 190 + 60 * tt[ok],
                        60 + 40 * tt[ok]], axis=1).astype(np.uint8)

    scene = scene_of(m)
    ik = _scaled_intr(intr, scale)
    tiles = []
    for T in _orbit_poses(centre, radius):
        t_hit, nrm, dirs = _cast(scene, T, ik)
        img, _hit = _shade(t_hit, nrm, dirs, window=None)
        uu, vv, z = _project(P, T, ik)
        inside = (z > 1e-3) & (uu >= 0) & (uu < ik["width"]) & \
                 (vv >= 0) & (vv < ik["height"])
        # only draw samples this orbit camera can actually see
        vis = inside.copy()
        if inside.any():
            xi = np.clip(np.round(uu[inside]).astype(int), 0, ik["width"] - 1)
            yi = np.clip(np.round(vv[inside]).astype(int), 0, ik["height"] - 1)
            dist = np.linalg.norm(P[inside] - T[:3, 3], axis=1)
            buf = t_hit[yi, xi]
            vis[inside] = np.isfinite(buf) & (buf > dist - 0.004)
        idx = np.flatnonzero(vis)
        for c in np.unique(col[idx], axis=0):
            sel = idx[(col[idx] == c).all(1)]
            _splat(img, uu[sel], vv[sel], c, radius=2)
        tiles.append(_square_crop(img, ik0["ppx"]))
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    return save_png(grid, out_png)


# --- the per-plan pass the CLI calls --------------------------------------
def render_plan(variant_dir, plan_json: Dict[str, Any],
                intr: Optional[Dict[str, float]] = None,
                *, scale: float = RENDER_SCALE,
                coverage: bool = True) -> List[Dict[str, Any]]:
    """Render every view of a stored plan. Works from the dict, not the class.

    Reads the mesh and npz by the relative paths *inside* the plan, so the
    plan file stays the single source of truth about its own artifacts.
    Returns one dict per view — `view_id`, `render` (path relative to the
    variant dir), `frame_fill`, `framed_frac`, `n_claimed` — which the caller
    writes back into the JSON. A list of dicts rather than a list of paths:
    the numbers are the point, the files are the illustration.
    """
    from inspection.viewplan import mesh as meshmod
    from inspection.viewplan.planner import DEFAULT_INTR

    variant_dir = Path(variant_dir)
    intr = dict(intr) if intr else dict(DEFAULT_INTR)
    out: List[Dict[str, Any]] = []
    if plan_json.get("abstain") or not plan_json.get("views"):
        return out
    step = int(plan_json["step"])
    m = meshmod.load(variant_dir / plan_json["mesh"]["path"])
    npz = np.load(variant_dir / plan_json["samples"]["path"])
    pts = npz["points"]
    scene = scene_of(m)
    window = plan_json.get("params", {}).get("standoff_window_m", (0.07, 0.20))

    for v in plan_json["views"]:
        img, st = render_view(scene, np.asarray(v["T_base_cam"], float), intr,
                              scale=scale, cover_ids=v.get("covered", []),
                              samples=pts, window=window)
        rel = f"renders/step_{step:03d}_view_{int(v['view_id']):02d}.png"
        save_png(img, variant_dir / rel)
        out.append({"view_id": int(v["view_id"]), "render": rel, **st})

    if coverage:
        rel = f"renders/step_{step:03d}_coverage.png"
        coverage_panel(m, {k: npz[k] for k in npz.files}, variant_dir / rel,
                       intr=intr,
                       cutoff_deg=float(plan_json.get("params", {})
                                        .get("incidence_cutoff_deg", 60.0)),
                       scale=scale)
        out.append({"view_id": -1, "render": rel, "frame_fill": 0.0,
                    "framed_frac": 1.0, "n_claimed": 0})
    return out
