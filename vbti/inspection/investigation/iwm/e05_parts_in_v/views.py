"""e05 views (RQ5, RQ6): the view for a part, and for a set of parts, from the geometry of S/T/V.

For a part: centre and PCA box of its rows; normal by the e04 rule (RANSAC 3 mm + PCA on the
inliers over the part's FACE rows within 16 mm of its centre; the face = the nearest ancestor,
self included, with >= 300 rows of which >= 70 % lie on one plane), flipped toward the cameras
that detected it; standoff so the part fills half the frame, clamped to the D405 range 7-50 cm.
Candidates: the normal pose, tilts of 15/30 deg about the two in-plane axes, a pivot ring of 8
views at 30 deg, and the run's recorded steps. Scores from a ray cast of V (raw, no weighted
sum): visible share of the part's rows, occluded share, off-frame share, centring, median
incidence angle, pixel extent, distance. Ranking: visible >= 0.9 first, then incidence, then
centring.

Run:  p inspection/investigation/iwm/e05_parts_in_v/views.py --run 2209-compute-ray-demo5 --mm 1 [--mode agent]
Outputs  evidence/views.json, evidence/sets.json, evidence/queries.json,
         evidence/view_<run>_<part>.jpg, evidence/pivot_<run>_<part>.jpg, evidence/set_<run>_<name>.jpg
"""
import argparse
import json
from math import atan, tan

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree

from inspection.investigation.iwm.e01_all_masks import common as C
from inspection.investigation.iwm.e05_parts_in_v import common as C5
from inspection.investigation.iwm.e05_parts_in_v import lift as L
from inspection.investigation.iwm.e05_parts_in_v.stv import STV, Surface

D_MIN, D_MAX = 0.105, 0.50     # D405 at 720p: the loop's MIN_RANGE_M (view/rays.py), and the far end of its range
FILL = 0.5                     # the part fills this fraction of the frame width
FX, W = 647.465, 1280.0        # the colour camera (both runs)
FOV_H = 2 * atan(W / (2 * FX))
R16 = 0.016                    # the e04 normal region
MIN_FACE_ROWS = 300
PLANAR = 0.7
TILTS = (15, 30)
SHIFTS_CM = (2, 4)             # face-plane translations of the face-on pose, along the face's up and side axes
RING_N, RING_DEG = 8, 30
RW, RH = 640, 360              # score render size (half the capture)
VIS_VOX, OCC_VOX = 1.0, 2.0
CLEAR_M = 0.02                 # a pose is feasible when no surface row lies closer than this to the camera


def feasible(clearance_m):
    return bool(clearance_m >= CLEAR_M)
SHOW = ["LAN 1G", "Serial Port 2", "Chort", "USB 3.2 Label 1", "barcode", "QR code", "Front Face", "Heatsink"]
SETS = {"two LAN ports": ["LAN 1G", "LAN 10G"], "lower panel ports": ["LAN 1G", "LAN 10G", "COM / DIO", "COM / CAN", "MIC IN", "LINE OUT"],
        "port with its label": ["USB 3.2 Port 1", "USB 3.2 Label 1"], "sticker and LAN ports": ["Chort", "LAN 1G", "LAN 10G"],
        "four USB ports": ["USB 3.2 Port 1", "USB 3.2 Port 2", "USB 3.2 Port 3", "USB 3.2 Port 4"],
        "barcode and QR": ["barcode", "QR code"]}
QUERIES = {"lan": ["LAN ports", "LAN 1G port", "LAN 10G port", "ethernet ports", "Ethernet Port 1", "Ethernet Port 2"],
           "serial": ["Serial Ports Group", "Serial Port 1", "Serial Port 2", "serial number label", "serial number text"],
           "usb": ["Array of USB 3.2 Ports", "USB 3.2 Port 1", "USB 3.2 Port 2", "USB 3.2 Port 3", "USB 3.2 Port 4", "USB 3.0 Dual Port",
                   "USB 3.2 Label 1", "USB 3.2 Label 2", "USB 3.2 Label 3", "USB 3.2 Label 4", "USB 3.0 Port Label"],
           "hdmi": ["HDMI Port", "HDMI Port Label"], "sticker": ["sticker", "Chort label", "AI KIT sticker"],
           "label": ["Chort label", "product info label", "serial number label", "DC Input Label", "USB 3.2 Label 1", "USB 3.2 Label 2",
                     "USB 3.2 Label 3", "USB 3.2 Label 4", "USB 3.0 Port Label", "HDMI Port Label"],
           "barcode": ["barcode"], "qr": ["QR code"], "power": ["Power Button", "DC Input Terminal Block", "DC Input Label"],
           "audio": ["Audio Port 1", "Audio Port 2", "audio jacks", "microphone jack (MIC IN)", "line output jack (LINE OUT)"],
           "antenna": ["antenna ports", "antenna port 1", "antenna port 2"], "heatsink": ["Top Heatsink Face"],
           "logo": ["CropTeq logo", "logo"], "ce": ["CE mark"], "front face": ["Front Face", "front face", "lower front panel", "upper front panel"]}
#: the ray cast's normals are in the CAMERA frame, pointing away from the camera: on a face-on render of
#: the VERTICAL Front Face (base normal (0.76, 0.65, 0.03)) the rendered normal at the centre is (0.01, -0.01, 1.00).
#: (A +z face viewed from above cannot tell the frames apart - that test misled once.)
NORMAL_IN_CAM = {"value": True}


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


# ====================================================================== poses
def pose_for(centre, normal, up_axis, dist):
    """Camera at centre + dist * normal looking along -normal; up_axis (projected) becomes the image x."""
    z = -unit(np.asarray(normal, float))
    x = np.asarray(up_axis, float) - (np.asarray(up_axis, float) @ z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [0, 0, 1.0]) if abs(z[2]) < 0.9 else np.cross(z, [1.0, 0, 0])
    x = unit(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = np.asarray(centre, float) + dist * unit(np.asarray(normal, float))
    return T


def rot_about(axis, deg):
    k = unit(np.asarray(axis, float))
    a = np.radians(deg)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def tilted(T, centre, axis, deg):
    """The pose rotated about `axis` through `centre` by deg: same distance, still looking at the centre."""
    R = rot_about(axis, deg)
    T2 = np.eye(4)
    T2[:3, :3] = R @ T[:3, :3]
    T2[:3, 3] = centre + R @ (T[:3, 3] - centre)
    return T2


def shifted(T, axis, dist):
    """The pose translated by dist along `axis` (in the face plane): same orientation, still face-on,
    the part moves off-centre. Anton's shift idea: clear the table without tilting."""
    T2 = T.copy()
    T2[:3, 3] = T[:3, 3] + dist * unit(np.asarray(axis, float))
    return T2


def axis_deg(T, normal):
    """Angle between the camera's optical axis and the part normal: 0 for face-on and for every
    face-plane shift (image plane parallel to the face, no keystone), the tilt angle for a tilt."""
    look = -T[:3, 2]
    return float(np.degrees(np.arccos(np.clip(look @ unit(np.asarray(normal, float)), -1, 1))))


def standoff(extent_mm):
    s = max(extent_mm[0], extent_mm[1]) / 1000.0
    return float(np.clip(s / (2 * FILL * tan(FOV_H / 2)), D_MIN, D_MAX))


# ====================================================================== geometry of a part
def frame_of(stv, part):
    rows = stv.parts[part].rows
    P = stv.s.centres(rows)
    mu = P.mean(0)
    Q = P - mu
    lam, vec = np.linalg.eigh(Q.T @ Q)
    axes = vec[:, ::-1]                                   # columns: long, mid, short
    proj = Q @ axes
    ext = (np.percentile(proj, 98, axis=0) - np.percentile(proj, 2, axis=0)) * 1000
    return {"centre": mu, "axes": axes, "extent_mm": ext, "n_rows": int(len(rows))}


def ransac_pca(P, thr=0.003, seed=0):
    if len(P) < 10:
        return None, 0.0
    if len(P) > L.FIT_PTS:
        P = P[np.random.default_rng(seed).choice(len(P), L.FIT_PTS, replace=False)]
    o3d.utility.random.seed(seed)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    _, inl = pc.segment_plane(distance_threshold=thr, ransac_n=3, num_iterations=1000)
    Q = P[inl] - P[inl].mean(0)
    _, vec = np.linalg.eigh(Q.T @ Q)
    return unit(vec[:, 0]), len(inl) / len(P)


def face_of(stv, part):
    """(face part, plane normal, inlier ratio): the nearest ancestor, self included, that is planar."""
    for p in [part] + stv.ancestors(part):
        rows = stv.parts[p].rows
        if len(rows) < MIN_FACE_ROWS:
            continue
        n, ratio = ransac_pca(stv.s.centres(rows))
        if n is not None and ratio >= PLANAR:
            return p, n, ratio
    return None


def normal_of(stv, part, toward):
    """The e04 rule: RANSAC + PCA over the face's rows within R16 of the part's centre, toward the cameras."""
    fr = frame_of(stv, part)
    c = fr["centre"]
    face = face_of(stv, part)
    src, ratio, fp = "pca_short_axis", 0.0, None
    n = fr["axes"][:, 2]
    if face is not None:
        fp, n_face, ratio = face
        F = stv.s.centres(stv.parts[fp].rows)
        near = F[np.linalg.norm(F - c, axis=1) <= R16]
        if len(near) >= 50:
            n2, r2 = ransac_pca(near)
            n, ratio, src = (n2, r2, "face_region_16mm") if n2 is not None else (n_face, ratio, "face_plane")
        else:
            n, src = n_face, "face_plane"
    if n @ (np.asarray(toward) - c) < 0:
        n = -n
    return unit(n), fp, ratio, src


def cameras_of(stv, part, steps):
    """Mean position of the cameras that detected the part (else of all steps)."""
    ids = {int(v[0].split(".")[1]) for v in stv.parts[part].detections} if stv.parts[part].detections else set()
    pos = [s["T"][:3, 3] for s in steps if s["id"] in ids] or [s["T"][:3, 3] for s in steps]
    return np.mean(pos, axis=0), sorted(ids)


# ====================================================================== scoring
def score(stv, vbg, size, part_rows, T, K, w=RW, h=RH, part_normal=None):
    depth, normal = C5.cast(vbg, size, T, K, w, h)
    P = stv.s.centres(part_rows)
    u, v, z = C5.project(P, K, T)
    inframe = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    clearance = float(stv.s.kd.query(T[:3, 3])[0])
    out = {"n_rows": int(len(P)), "outside": round(float(1 - inframe.mean()), 3), "dist_m": round(float(np.linalg.norm(T[:3, 3] - P.mean(0))), 4),
           "clearance_m": round(clearance, 4), "feasible": feasible(clearance),
           "axis_deg": round(axis_deg(T, part_normal), 1) if part_normal is not None else None}
    hits, px = C5.hits_to_points(depth, K, T)
    if not len(hits) or not inframe.any():
        out.update({"visible": 0.0, "occluded": round(float(inframe.mean()), 3), "centring_x": 1.0, "centring_y": 1.0,
                    "incidence_med_deg": None, "px_w_frac": 0.0, "px_h_frac": 0.0, "n_px": 0})
        return out
    d, _ = cKDTree(hits).query(P, distance_upper_bound=VIS_VOX * size)
    out["visible"] = round(float(np.isfinite(d).mean()), 3)
    vi, ui = v[inframe].astype(int), u[inframe].astype(int)
    dr = depth[vi, ui]
    out["occluded"] = round(float(((dr > 0) & (z[inframe] - dr > OCC_VOX * size)).sum() / len(P)), 3)
    dP, _ = cKDTree(P).query(hits, distance_upper_bound=VIS_VOX * size)
    on = np.isfinite(dP)
    ppx = px[on]
    out["n_px"] = int(len(ppx))
    if len(ppx):
        cy, cx = ppx.mean(0)
        out["centring_x"] = round(float(abs(cx - w / 2) / w), 3)
        out["centring_y"] = round(float(abs(cy - h / 2) / h), 3)
        out["px_w_frac"] = round(float(np.ptp(ppx[:, 1]) / w), 3)
        out["px_h_frac"] = round(float(np.ptp(ppx[:, 0]) / h), 3)
        rays = hits[on] - T[:3, 3]
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        nrm = normal[ppx[:, 0], ppx[:, 1]]
        if NORMAL_IN_CAM["value"]:
            nrm = nrm @ T[:3, :3].T
        cosang = np.abs(np.einsum("ij,ij->i", rays, nrm)) / (np.linalg.norm(nrm, axis=1) + 1e-9)
        out["incidence_med_deg"] = round(float(np.degrees(np.arccos(np.clip(np.median(cosang), 0, 1)))), 1)
    else:
        out.update({"centring_x": 1.0, "centring_y": 1.0, "px_w_frac": 0.0, "px_h_frac": 0.0, "incidence_med_deg": None})
    return out


def rank_key(s):
    """Infeasible last; then visible >= 0.9 first (else by visible); then the optical-axis angle to the
    part normal (keystone: 0 for face-on and shifts); then the per-pixel incidence; then centring."""
    inc = s["incidence_med_deg"] if s["incidence_med_deg"] is not None else 90.0
    ax = s.get("axis_deg")
    return (not s.get("feasible", True), s["visible"] < 0.9, -s["visible"] if s["visible"] < 0.9 else 0.0,
            round(ax) if ax is not None else 0, inc, max(s["centring_x"], s["centring_y"]))


def normal_frame_check(stv, vbg, size, part, T, K):
    """Decide whether the ray-cast normals come in the camera or the base frame: compare the
    rendered normal at the part's pixels with the RANSAC plane normal of the part in both frames."""
    depth, normal = C5.cast(vbg, size, T, K, RW, RH)
    P = stv.s.centres(stv.parts[part].rows)
    n_plane, _ = ransac_pca(P)
    hits, px = C5.hits_to_points(depth, K, T)
    dP, _ = cKDTree(P).query(hits, distance_upper_bound=VIS_VOX * size)
    nrm = normal[px[np.isfinite(dP), 0], px[np.isfinite(dP), 1]]
    nrm = nrm[np.linalg.norm(nrm, axis=1) > 0.5]
    med = unit(np.median(nrm, axis=0))
    a_base = np.degrees(np.arccos(abs(med @ n_plane)))
    a_cam = np.degrees(np.arccos(abs((T[:3, :3] @ med) @ n_plane)))
    # diagnostic only: the frame is fixed above (a |dot| on a grazing view misled this check once)
    return {"angle_if_base": round(float(a_base), 1), "angle_if_cam": round(float(a_cam), 1), "normal_in_cam": NORMAL_IN_CAM["value"]}


# ====================================================================== the view for a part / a set
def candidates(centre, n, up, d, steps):
    T0 = pose_for(centre, n, up, d)
    side = np.cross(n, up)
    cands = [("normal", T0)]
    for deg in TILTS:
        for name, axis in (("x", up), ("y", side)):
            cands.append((f"tilt+{deg}{name}", tilted(T0, centre, axis, deg)))
            cands.append((f"tilt-{deg}{name}", tilted(T0, centre, axis, -deg)))
    for cm in SHIFTS_CM:
        for name, axis in (("up", up), ("side", side)):
            cands.append((f"shift+{cm}{name}", shifted(T0, axis, cm / 100)))
            cands.append((f"shift-{cm}{name}", shifted(T0, axis, -cm / 100)))
    for k in range(RING_N):
        az = 2 * np.pi * k / RING_N
        cands.append((f"pivot{k}", tilted(T0, centre, np.cos(az) * up + np.sin(az) * side, RING_DEG)))
    for s in steps:
        cands.append((f"step{s['id']}", s["T"]))
    return cands


def view_for(stv, vbg, size, part, steps, K):
    fr = frame_of(stv, part)
    toward, det_steps = cameras_of(stv, part, steps)
    n, face, ratio, src = normal_of(stv, part, toward)
    d = standoff(fr["extent_mm"])
    cands = candidates(fr["centre"], n, fr["axes"][:, 0], d, steps)
    rows = stv.parts[part].rows
    scored = [{"name": name, "T": T.tolist(), **score(stv, vbg, size, rows, T, K, part_normal=n)} for name, T in cands]
    computed = [s for s in scored if not s["name"].startswith("step")]
    best = min(computed, key=rank_key)
    det = [s for s in scored if s["name"] in {f"step{i}" for i in det_steps}]
    best_det = min(det, key=rank_key) if det else None
    return {"part": part, "name": stv.parts[part].name, "path": stv.path(part), "depth": stv.depth(part),
            "centre": [round(float(x), 4) for x in fr["centre"]], "normal": [round(float(x), 4) for x in n],
            "normal_source": src, "face": face, "face_name": stv.parts[face].name if face is not None else None,
            "planar": round(float(ratio), 3), "extent_mm": [round(float(x), 1) for x in fr["extent_mm"]],
            "standoff_m": round(d, 4), "detecting_steps": det_steps,
            "normal_score": scored[0], "best": best, "best_detecting_step": best_det,
            "computed_beats_step": (rank_key(best) <= rank_key(best_det)) if best_det else None,
            "all": scored}


def view_for_set(stv, vbg, size, parts, steps, K):
    rows = np.unique(np.concatenate([stv.parts[p].rows for p in parts]))
    P = stv.s.centres(rows)
    c = P.mean(0)
    Q = P - c
    _, vec = np.linalg.eigh(Q.T @ Q)
    axes = vec[:, ::-1]
    ext = (np.percentile(Q @ axes, 98, axis=0) - np.percentile(Q @ axes, 2, axis=0)) * 1000
    toward = np.mean([cameras_of(stv, p, steps)[0] for p in parts], axis=0)
    n, ratio = ransac_pca(P)
    src = "union_plane"
    if n is None or ratio < PLANAR:
        n = unit(np.mean([normal_of(stv, p, toward)[0] for p in parts], axis=0))
        src = "mean_of_part_normals"
    if n @ (toward - c) < 0:
        n = -n
    d = standoff(ext)
    cands = candidates(c, n, axes[:, 0], d, steps)
    scored = []
    for name, T in cands:
        per = {p: score(stv, vbg, size, stv.parts[p].rows, T, K) for p in parts}
        union = score(stv, vbg, size, rows, T, K, part_normal=n)
        s = {"name": name, "T": T.tolist(), "min_visible": round(min(v["visible"] for v in per.values()), 3),
             "visible": round(min(v["visible"] for v in per.values()), 3), "centring_x": union["centring_x"], "centring_y": union["centring_y"],
             "incidence_med_deg": union["incidence_med_deg"], "dist_m": union["dist_m"], "axis_deg": union["axis_deg"],
             "clearance_m": union["clearance_m"], "feasible": union["feasible"],
             "per_part": {stv.parts[p].name: v["visible"] for p, v in per.items()}}
        scored.append(s)
    computed = [s for s in scored if not s["name"].startswith("step")]
    best = min(computed, key=rank_key)
    best_step = min([s for s in scored if s["name"].startswith("step")], key=rank_key)
    return {"parts": [stv.parts[p].name for p in parts], "centre": [round(float(x), 4) for x in c], "normal_source": src,
            "planar": round(float(ratio), 3), "extent_mm": [round(float(x), 1) for x in ext], "standoff_m": round(d, 4),
            "best": best, "best_step": best_step,
            "exists_80": bool(max((s["min_visible"] for s in computed if s["feasible"]), default=0.0) >= 0.8), "all": scored}


# ====================================================================== figures
def render_tile(stv, vbg, size, rows, T, K, title, w=RW, h=RH, crop=None):
    depth, _, col = C5.cast(vbg, size, T, K, w, h, color=True)
    hits, px = C5.hits_to_points(depth, K, T)
    P = stv.s.centres(rows)
    m = np.zeros((h, w), bool)
    if len(hits):
        dP, _ = cKDTree(P).query(hits, distance_upper_bound=VIS_VOX * size)
        on = np.isfinite(dP)
        m[px[on, 0], px[on, 1]] = True
    img = C.outlines(col, [m], [(255, 0, 255)], thick=2, dim=0.0)
    img[h // 2 - 1:h // 2 + 1, :] = (img[h // 2 - 1:h // 2 + 1, :] * 0.5 + 60).astype(np.uint8)
    img[:, w // 2 - 1:w // 2 + 1] = (img[:, w // 2 - 1:w // 2 + 1] * 0.5 + 60).astype(np.uint8)
    return C.label(img, title, size=13)


def fmt(s):
    inc = f"{s['incidence_med_deg']:.0f}°" if s["incidence_med_deg"] is not None else "–"
    ax = f" axis {s['axis_deg']:.0f}°" if s.get("axis_deg") is not None else ""
    return (f"vis {s['visible']:.2f} occ {s['occluded']:.2f}{ax} inc {inc} ctr {max(s['centring_x'], s['centring_y']):.2f} d {s['dist_m']:.2f} m"
            + ("" if s.get("feasible", True) else f" INFEASIBLE ({100 * s['clearance_m']:.1f} cm free)"))


def step_crop(steps, step_id, centre, K_full):
    st = next((s for s in steps if s["id"] == step_id), None)
    if st is None:
        return None
    u, v, z = C5.project(centre[None], K_full, st["T"])
    if not (z[0] > 0):
        return None
    h, w = st["rgb"].shape[:2]
    x0, y0 = int(np.clip(u[0] - 160, 0, w - 320)), int(np.clip(v[0] - 120, 0, h - 240))
    img = st["rgb"][y0:y0 + 240, x0:x0 + 320].copy()
    return img


def figure_part(stv, vbg, size, res, steps, K, K_full, path):
    rows = stv.parts[res["part"]].rows
    tiles = [render_tile(stv, vbg, size, rows, np.array(res["normal_score"]["T"]), K, f"normal pose: {fmt(res['normal_score'])}")]
    if res["best"]["name"] != "normal":
        tiles.append(render_tile(stv, vbg, size, rows, np.array(res["best"]["T"]), K, f"best {res['best']['name']}: {fmt(res['best'])}"))
    if res["best_detecting_step"]:
        bd = res["best_detecting_step"]
        tiles.append(render_tile(stv, vbg, size, rows, np.array(bd["T"]), K, f"detecting {bd['name']}: {fmt(bd)}"))
        crop = step_crop(steps, int(bd["name"][4:]), np.array(res["centre"]), K_full)
        if crop is not None:
            crop = C.resize(crop, RW)
            tiles.append(C.label(crop, f"{bd['name']}'s real frame around the part", size=13))
    hmax = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, hmax - t.shape[0]), (0, 0), (0, 0)), constant_values=255) for t in tiles]
    g = C.title_band(C.grid(tiles, cols=2), f"{res['name']} ({res['path']}): extent {res['extent_mm'][0]:.0f}×{res['extent_mm'][1]:.0f} mm, "
                                            f"normal from {res['normal_source']}, standoff {res['standoff_m']:.2f} m")
    Image.fromarray(g).save(path, quality=88)


def figure_pivot(stv, vbg, size, res, K, path):
    rows = stv.parts[res["part"]].rows
    tiles = [render_tile(stv, vbg, size, rows, np.array(s["T"]), K, f"{s['name']}: {fmt(s)}")
             for s in res["all"] if s["name"] == "normal" or s["name"].startswith("pivot")]
    g = C.title_band(C.grid(tiles, cols=3), f"{res['name']}: the normal pose and the pivot ring ({RING_DEG}° around the normal)")
    Image.fromarray(g).save(path, quality=85)


def figure_set(stv, vbg, size, name, parts, res, K, path):
    T = np.array(res["best"]["T"])
    depth, _, col = C5.cast(vbg, size, T, K, RW, RH, color=True)
    hits, px = C5.hits_to_points(depth, K, T)
    masks, cols = [], []
    pal = C._palette(len(parts), seed=5)
    for i, p in enumerate(parts):
        m = np.zeros((RH, RW), bool)
        if len(hits):
            dP, _ = cKDTree(stv.s.centres(stv.parts[p].rows)).query(hits, distance_upper_bound=VIS_VOX * size)
            on = np.isfinite(dP)
            m[px[on, 0], px[on, 1]] = True
        masks.append(m)
        cols.append(tuple(int(x) for x in pal[i]))
    img = C.outlines(col, masks, cols, thick=2, dim=0.0)
    per = ", ".join(f"{k} {v:.2f}" for k, v in res["best"]["per_part"].items())
    g = C.title_band(C.label(img, f"best {res['best']['name']}: min visible {res['best']['min_visible']:.2f}; {per}", size=12),
                     f"set '{name}': {len(parts)} parts, normal from {res['normal_source']}, standoff {res['standoff_m']:.2f} m")
    Image.fromarray(g).save(path, quality=88)


# ====================================================================== main
def load_final(run, mm, mode):
    vbg, size = C5.V(run, mm)
    surf = Surface.from_vbg(vbg, size, C5.trunc_m(size))
    d = C5.DATA / run / f"{mm}mm" / f"keep_{mode}" / f"after_{C5.ORDER[run][-1]}"
    if not d.exists():
        d = C5.DATA / run / f"{mm}mm" / "keep_baseline" / f"after_{C5.ORDER[run][-1]}"
    return STV.load(d, surf), vbg, size, d


def find(stv, name):
    hits = stv.search(name)
    return hits[0] if hits else None


def queries_check(stv):
    out = {}
    names = {p: stv.parts[p].name for p in stv.parts}
    for q, expected in QUERIES.items():
        hits = {names[p] for p in stv.search(q)}
        exp = {e for e in expected if e in names.values()}
        if not exp:
            continue
        tp = len(hits & exp)
        out[q] = {"expected": sorted(exp), "hits": sorted(hits), "precision": round(tp / len(hits), 2) if hits else None,
                  "recall": round(tp / len(exp), 2)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=list(C5.ORDER))
    ap.add_argument("--mm", type=int, default=1)
    ap.add_argument("--mode", default="agent")
    ap.add_argument("--all-parts", action="store_true", default=True)
    a = ap.parse_args()
    views_out, sets_out, queries_out = {}, {}, {}
    for run in a.runs:
        stv, vbg, size, src = load_final(run, a.mm, a.mode)
        steps, intr, scale = C5.run_steps(run)
        K = C5.K_of(intr, 0.5)
        K_full = C5.K_of(intr)
        face = max(stv.parts[stv.root].children, key=lambda p: len(stv.parts[p].rows))     # the largest face, diagnostic only
        det_view = stv.parts[face].detections[0][0] if stv.parts[face].detections else C5.ORDER[run][0]
        chk = normal_frame_check(stv, vbg, size, face, next(s["T"] for s in steps if s["id"] == int(det_view.split(".")[1])), K)
        print(f"{run}: normals from the ray cast are in the {'camera' if chk['normal_in_cam'] else 'base'} frame {chk}")
        results = []
        for p in sorted(stv.parts):
            if p == stv.root or len(stv.parts[p].rows) < 20:
                continue
            r = view_for(stv, vbg, size, p, steps, K)
            results.append(r)
            print(f"  P{p:<3} {stv.parts[p].name[:26]:26s} {r['extent_mm'][0]:5.0f} mm n={r['normal_source'][:12]:12s} d {r['standoff_m']:.2f} | "
                  f"normal {fmt(r['normal_score'])} | best {r['best']['name']:10s} | step {fmt(r['best_detecting_step']) if r['best_detecting_step'] else '-'} "
                  f"| beats {r['computed_beats_step']}")
        n_cmp = [r for r in results if r["computed_beats_step"] is not None]
        summary = {"parts": len(results), "compared": len(n_cmp), "computed_beats_or_matches_step": sum(r["computed_beats_step"] for r in n_cmp),
                   "normal_pose_infeasible": sum(not r["normal_score"]["feasible"] for r in results),
                   "best_kind": {k: sum(r["best"]["name"].rstrip("0123456789+-xyupside").startswith(k) for r in results)
                                 for k in ("normal", "tilt", "shift", "pivot")},
                   "shift_rescues": sum(1 for r in results if not r["normal_score"]["feasible"]
                                        and any(s["name"].startswith("shift") and s["feasible"] and s["visible"] >= 0.9 for s in r["all"])),
                   "candidates_infeasible_share": round(float(np.mean([not s["feasible"] for r in results for s in r["all"] if not s["name"].startswith("step")])), 3),
                   "normal_visible_ge_0.9": sum(r["normal_score"]["visible"] >= 0.9 for r in results),
                   "normal_centred_le_0.1": sum(max(r["normal_score"]["centring_x"], r["normal_score"]["centring_y"]) <= 0.1 for r in results),
                   "normal_incidence_median": float(np.median([r["normal_score"]["incidence_med_deg"] for r in results if r["normal_score"]["incidence_med_deg"] is not None])),
                   "tilt30_visible_drop_median": float(np.median([r["normal_score"]["visible"] - min(s["visible"] for s in r["all"] if s["name"].startswith("tilt+30") or s["name"].startswith("tilt-30")) for r in results])),
                   "normal_frame_check": chk, "stv": str(src)}
        views_out[run] = {"summary": summary, "parts": [{k: v for k, v in r.items() if k != "all"} for r in results],
                          "all_candidates": {str(r["part"]): [{k: v for k, v in s.items() if k != "T"} for s in r["all"]] for r in results}}
        print(f"  summary {json.dumps({k: v for k, v in summary.items() if k not in ('normal_frame_check', 'stv')})}")
        # figures for the chosen parts
        for q in SHOW:
            p = find(stv, q)
            if p is None or p == stv.root:
                continue
            r = next((x for x in results if x["part"] == p), None)
            if r is None:
                continue
            figure_part(stv, vbg, size, r, steps, K, K_full, C5.EVID / f"view_{run}_P{p}.jpg")
        p = find(stv, "LAN 1G") or find(stv, "USB 3.2 Port 1") or find(stv, "QR code")
        if p is not None:
            r = next((x for x in results if x["part"] == p), None)
            if r is not None:
                figure_pivot(stv, vbg, size, r, K, C5.EVID / f"pivot_{run}_P{p}.jpg")
        # sets
        for name, qs in SETS.items():
            parts = [find(stv, q) for q in qs]
            parts = [p for p in parts if p is not None and p != stv.root]
            if len(parts) < 2:
                continue
            r = view_for_set(stv, vbg, size, parts, steps, K)
            sets_out[f"{run}: {name}"] = {k: v for k, v in r.items() if k != "all"}
            print(f"  set {name}: best {r['best']['name']} min visible {r['best']['min_visible']:.2f} exists_80 {r['exists_80']} | step {r['best_step']['name']} {r['best_step']['min_visible']:.2f}")
            figure_set(stv, vbg, size, name, parts, r, K, C5.EVID / f"set_{run}_{name.replace(' ', '_')}.jpg")
        queries_out[run] = queries_check(stv)
    json.dump(views_out, open(C5.EVID / "views.json", "w"), indent=1)
    json.dump(sets_out, open(C5.EVID / "sets.json", "w"), indent=1)
    json.dump(queries_out, open(C5.EVID / "queries.json", "w"), indent=1)
    for run, q in queries_out.items():
        print(run, "queries:", {k: (v["precision"], v["recall"]) for k, v in q.items()})


if __name__ == "__main__":
    main()
