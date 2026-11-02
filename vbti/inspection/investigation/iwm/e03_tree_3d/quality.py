"""e03 turn 2: how well do the e02 masks land on the geometry?

Per frame, three figures in evidence/quality/:
  <f>_depth.jpg  mask outlines on RGB | on the depth map | mask pixels without depth
  <f>_3d.jpg     node points painted on the run's fused cloud: capture camera, side,
                 top, and the distance of every node point to the fused surface
  <f>_views.jpg  the nodes carried through 3D into other steps of the same run
                 (occlusion-tested against that step's depth); first panel = the
                 frame itself (round trip); where the target has its own e02 tree,
                 its own masks are drawn as white outlines
Numbers: evidence/projection_quality.json.

Run:  p vbti/inspection/investigation/iwm/e03_tree_3d/quality.py [--frames c5.6 ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation
from scipy.spatial import cKDTree

from inspection.cell.geometry import project_to_pixels
from inspection.investigation.iwm.e01_all_masks import common as C
from inspection.investigation.iwm.e03_tree_3d.project import DATA, E02, load_tree, session_intrinsics
from inspection.record.run import Run

HERE = Path(__file__).resolve().parent
EVID = HERE / "evidence" / "quality"
OUT = DATA / "quality"
PALETTE = [(30, 160, 255), (255, 140, 0), (40, 215, 110), (255, 60, 170), (160, 90, 255),
           (250, 220, 30), (0, 220, 220), (255, 80, 60), (150, 230, 40), (255, 130, 210),
           (0, 120, 200), (200, 100, 0), (0, 160, 90), (190, 0, 120), (110, 60, 200),
           (190, 170, 0), (0, 160, 160), (200, 50, 40), (100, 170, 20), (230, 170, 230)]
PANEL_W = 560
OCCL_TOL_M = 0.025          # a carried point is visible if within 25 mm of that view's depth
BLEED_MM = 8.0
MAX_ANGLE = 70              # carry only into views that see the face at most this oblique


# ------------------------------------------------------------------ helpers
def assign_colors(nodes):
    """One colour per node (parents first), so no two nodes of a frame share one."""
    out = {0: (200, 200, 200)}
    for k, i in enumerate(node_order(nodes)):
        out[i] = PALETTE[k % len(PALETTE)]
    return out


def turbo(x):
    """Small turbo-like colormap without matplotlib state: x in [0,1] -> uint8 rgb."""
    import matplotlib
    return (matplotlib.colormaps["turbo"](np.clip(x, 0, 1))[..., :3] * 255).astype(np.uint8)


def crop(img, box):
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]


def fit_w(img, w=PANEL_W):
    return C.resize(img, w)


def pad_to(img, h, bg=(255, 255, 255), center=False):
    if img.shape[0] >= h:
        return img[:h]
    out = np.full((h, img.shape[1], 3), bg, np.uint8)
    y = (h - img.shape[0]) // 2 if center else 0
    out[y:y + img.shape[0]] = img
    return out


def letterbox(img, w=440, h=330, bg=(255, 255, 255)):
    """Fit inside w x h keeping aspect, centred."""
    from PIL import Image
    im = Image.fromarray(img)
    k = min(w / im.width, h / im.height)
    im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.BILINEAR)
    out = np.full((h, w, 3), bg, np.uint8)
    y, x = (h - im.height) // 2, (w - im.width) // 2
    out[y:y + im.height, x:x + im.width] = np.array(im)
    return out


def main_normal(f):
    """Normal of the frame's largest side, pointing at the capture camera."""
    side = max((i for i in f.nodes if f.nodes[i]["depth"] == 1), key=lambda i: f.nodes[i]["n_points"])
    S = f.P[f.Pn == side]
    c = S.mean(0)
    n = np.linalg.svd(S[:: max(1, len(S) // 20000)] - c, full_matrices=False)[2][2]
    return (n if n @ (f.T[:3, 3] - c) > 0 else -n), c, side


def view_angle_deg(f, g):
    n, c, _ = main_normal(f)
    r = g.T[:3, 3] - c
    return float(np.degrees(np.arccos(np.clip(n @ r / np.linalg.norm(r), -1, 1))))


def row(panels, pad=6):
    h = max(p.shape[0] for p in panels)
    return C.grid([pad_to(p, h) for p in panels], cols=len(panels), pad=pad)


def node_order(nodes):
    """Parents first, so children are painted on top."""
    return sorted((i for i in nodes if i != 0), key=lambda i: (nodes[i]["depth"], -nodes[i]["area"]))


def legend(nodes, colors, w):
    """A strip naming every node in its colour."""
    from PIL import Image, ImageDraw
    items = [(i, nodes[i]["name"] or f"#{i}") for i in node_order(nodes)]
    font = C._font(14)
    im = Image.new("RGB", (w, 22), "white")
    d = ImageDraw.Draw(im)
    x, y, lines = 8, 4, [[]]
    for i, name in items:
        tw = d.textlength(name, font=font) + 30
        if x + tw > w - 8:
            lines.append([])
            x = 8
        lines[-1].append((x, i, name))
        x += tw
    im = Image.new("RGB", (w, 22 * len(lines) + 4), "white")
    d = ImageDraw.Draw(im)
    for r, line in enumerate(lines):
        for x, i, name in line:
            y = 4 + 22 * r
            d.rectangle([x, y + 2, x + 14, y + 16], fill=colors[i])
            d.text((x + 20, y), name, fill="black", font=font)
    return np.array(im)


# ------------------------------------------------------------------ frame data
class F:
    def __init__(self, frame):
        self.frame = frame
        self.id = frame["id"]
        p = C.frame_paths(frame)
        step = json.load(open(p["step"]))
        self.rot = int(step.get("rgb_rotation_deg", 0) or 0)
        self.T = np.array(step["T_base_cam"])
        self.run_dir = C.RUNS_DIR / frame["run"]
        self.intr, self.scale = session_intrinsics(self.run_dir)
        self.rgb = C.load_rgb(frame)
        self.depth = np.load(p["depth"])                     # upright, like rgb
        self.tree, self.masks = load_tree(frame)
        d = DATA / frame["id"] / "full"
        z = np.load(d / "nodes.npz")
        self.P, self.Pc, self.Pn = z["points"], z["colors"], z["node"]
        self.nodes = {n["id"]: n for n in json.load(open(d / "nodes.json"))["nodes"]}
        self.colors = assign_colors(self.nodes)
        self.box = C.expand_box(self.nodes[0]["bbox"], self.rgb.shape[:2], 0.10)
        run = Run.load(self.run_dir)
        fused = run.fused()
        self.cloud, self.cloud_cols = (fused[0], fused[1]) if fused else (np.zeros((0, 3)), None)

    def masks_list(self):
        ids = node_order(self.nodes)
        return ids, [self.masks[i] for i in ids], [self.colors[i] for i in ids]


# ------------------------------------------------------------------ figure 1
def fig_depth(f):
    ids, ms, cols = f.masks_list()
    rgb_c = crop(f.rgb, f.box)
    mc = [crop(m, f.box) for m in ms]
    a = C.overlay(rgb_c, mc, alpha=0.35, colors=cols, thick=2)
    dep = crop(f.depth, f.box).astype(np.float32) * f.scale
    obj = crop(f.masks[0], f.box) & (dep > 0)
    lo, hi = np.percentile(dep[obj], [2, 98]) if obj.any() else (0.1, 0.5)
    dimg = turbo((dep - lo) / max(hi - lo, 1e-3))
    dimg[dep == 0] = (0, 0, 0)
    b = C.outlines(dimg, mc, cols, thick=2, dim=0.0)
    gray = (rgb_c.mean(2, keepdims=True).repeat(3, 2) * 0.55 + 90).astype(np.uint8)
    hole = np.zeros(dep.shape, bool)
    for m in mc:
        hole |= m & (dep == 0)
    c = gray.copy()
    c[hole] = (235, 30, 30)
    c = C.outlines(c, mc, cols, thick=1, dim=0.0)
    a = C.label(fit_w(a), "masks on RGB", size=16)
    b = C.label(fit_w(b), f"masks on depth ({lo * 100:.1f}-{hi * 100:.1f} cm)", size=16)
    c = C.label(fit_w(c), "red = mask pixel without depth", size=16)
    img = row([a, b, c])
    img = np.concatenate([img, legend(f.nodes, f.colors, img.shape[1])], 0)
    return C.title_band(img, f"{f.id} · {f.frame['run']} step {f.frame['step']} — masks against depth")


# ------------------------------------------------------------------ figure 2
def look_at(eye, target, up=(0, 0, 1)):
    """T_base_cam (OpenCV: z forward, y down) for a virtual camera."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, np.array(up, float))
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, np.array([1.0, 0, 0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = eye
    return T


def fit_f(points, T, size=(1600, 1200), fill=0.8):
    Ti = np.linalg.inv(T)
    pc = points @ Ti[:3, :3].T + Ti[:3, 3]
    pc = pc[pc[:, 2] > 0.02]
    x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    ex = max(np.percentile(np.abs(x - np.median(x)), 99.5), 1e-3)
    ey = max(np.percentile(np.abs(y - np.median(y)), 99.5), 1e-3)
    return min(size[0] / 2 * fill / ex, size[1] / 2 * fill / ey), np.median(x), np.median(y)


def splat(points, colors, T, f=1400.0, size=(1600, 1200), r=2, bias=None, c=(0.0, 0.0)):
    """Painter's z-buffer of disks; returns image and the pixel bbox of `points`."""
    W, H = size
    Ti = np.linalg.inv(T)
    pc = points @ Ti[:3, :3].T + Ti[:3, 3]
    z = pc[:, 2] - (bias if bias is not None else 0)
    keep = pc[:, 2] > 0.02
    u = (f * (pc[:, 0] / pc[:, 2] - c[0]) + W / 2)
    v = (f * (pc[:, 1] / pc[:, 2] - c[1]) + H / 2)
    keep &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, cols = u[keep].astype(int), v[keep].astype(int), z[keep], colors[keep]
    order = np.argsort(-z)                                       # far first, near overwrites
    u, v, cols = u[order], v[order], cols[order]
    img = np.full((H, W, 3), (18, 20, 26), np.uint8)
    offs = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1) if dx * dx + dy * dy <= r * r]
    for dx, dy in offs:
        uu, vv = np.clip(u + dx, 0, W - 1), np.clip(v + dy, 0, H - 1)
        img[vv, uu] = cols
    return img


def proj_bbox(points, T, f=1400.0, size=(1600, 1200), margin=0.12, c=(0.0, 0.0)):
    W, H = size
    Ti = np.linalg.inv(T)
    pc = points @ Ti[:3, :3].T + Ti[:3, 3]
    pc = pc[pc[:, 2] > 0.02]
    u, v = f * (pc[:, 0] / pc[:, 2] - c[0]) + W / 2, f * (pc[:, 1] / pc[:, 2] - c[1]) + H / 2
    x0, x1 = np.percentile(u, [0.5, 99.5])
    y0, y1 = np.percentile(v, [0.5, 99.5])
    mx, my = (x1 - x0) * margin + 20, (y1 - y0) * margin + 20
    return (int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(W, x1 + mx)), int(min(H, y1 + my)))


def paint_nodes(f, k=0.7):
    """Each node point painted with the DEEPEST node containing it (children on top)."""
    cols = f.Pc.astype(np.float32).copy()
    depth = np.array([f.nodes[n]["depth"] for n in f.Pn])
    sel = f.Pn != 0
    pal = np.array([f.colors.get(n, (200, 200, 200)) for n in f.Pn], np.float32)
    cols[sel] = cols[sel] * (1 - k) + pal[sel] * k
    return cols.astype(np.uint8), depth


def side_of(f, i):
    while f.nodes[i]["parent"] not in (0, None):
        i = f.nodes[i]["parent"]
    return i


def plane_dist_mm(f):
    """|distance| of every node point to a robust plane fitted on its side's points."""
    out = np.full(len(f.P), np.nan)
    sides = {side_of(f, i) for i in f.nodes if i != 0}
    side_of_pt = np.array([side_of(f, n) if n != 0 else -1 for n in f.Pn])
    for sd in sides:
        S = f.P[f.Pn == sd]
        keep = np.ones(len(S), bool)
        for _ in range(3):
            c = S[keep].mean(0)
            n = np.linalg.svd(S[keep][:: max(1, keep.sum() // 20000)] - c, full_matrices=False)[2][2]
            r = np.abs((S - c) @ n)
            keep = r < max(np.percentile(r, 80), 0.002)
        sel = side_of_pt == sd
        out[sel] = np.abs((f.P[sel] - c) @ n) * 1000
    return out


def fig_3d(f, fused_kd):
    sel = f.Pn != 0
    P = f.P[sel]
    pcols, pdepth = paint_nodes(f)
    pcols, pdepth = pcols[sel], pdepth[sel]
    center = f.P[f.Pn == 0].mean(0)
    near = np.linalg.norm(f.cloud - center, axis=1) < 0.25 if len(f.cloud) else np.zeros(0, bool)
    cloud = f.cloud[near]
    ccol = ((f.cloud_cols[near] if f.cloud_cols is not None else np.full((near.sum(), 3), 150))
            * 0.55).astype(np.uint8)
    d_mm = fused_kd.query(P)[0] * 1000 if fused_kd is not None else np.full(len(P), np.nan)
    pl_mm = plane_dist_mm(f)[sel]
    plane_col = turbo(np.clip(pl_mm / 10.0, 0, 1) * 0.85 + 0.1)

    cam = f.T[:3, 3]
    dist = np.linalg.norm(cam - center)
    ang = np.deg2rad(65)
    rel = cam - center
    Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    views = [("capture camera", f.T),
             ("side, 65° around the vertical", look_at(center + Rz @ rel, center)),
             ("top", look_at(center + np.array([0.0, 0.0, dist]), center, up=(1, 0, 0)))]
    panels = []
    for title, T in views:
        if title == "capture camera" and f.rot == 180:          # upright like the RGB
            T = T @ np.diag([-1, -1, 1, 1])
        allp = np.concatenate([cloud, P])
        allc = np.concatenate([ccol, pcols])
        bias = np.concatenate([np.zeros(len(cloud)), 1e-4 * pdepth])
        fz, cx, cy = fit_f(P, T)
        img = splat(allp, allc, T, f=fz, bias=bias, c=(cx, cy))
        x0, y0, x1, y1 = proj_bbox(P, T, f=fz, c=(cx, cy))
        panels.append(C.label(fit_w(img[y0:y1, x0:x1], 420), title, size=15))
    # distance to the node's own side plane, from the side view
    T = views[1][1]
    fz, cx, cy = fit_f(P, T)
    img = splat(np.concatenate([cloud, P]), np.concatenate([(ccol * 0.6).astype(np.uint8), plane_col]), T,
                f=fz, c=(cx, cy))
    x0, y0, x1, y1 = proj_bbox(P, T, f=fz, c=(cx, cy))
    panels.append(C.label(fit_w(img[y0:y1, x0:x1], 420), "off its side's plane: blue 0 → red ≥10 mm",
                          size=13))
    h = max(p.shape[0] for p in panels)
    img = C.grid([pad_to(p, h, (18, 20, 26), center=True) for p in panels], cols=4, pad=6)
    img = np.concatenate([img, legend(f.nodes, f.colors, img.shape[1])], 0)
    title = f"{f.id} — node points painted on the run's fused cloud (dimmed)"
    if f.frame.get("recorded_mask") is False:
        title += "  ·  NOTE: this step was not fused into the run cloud"
    return C.title_band(img, title), d_mm, pl_mm


# ------------------------------------------------------------------ figure 3
def carry(f, g):
    """Carry f's node points into view g. Returns {node: mask in g (upright)}, stats."""
    uv = project_to_pixels(f.P, g.T, g.intr)
    Ti = np.linalg.inv(g.T)
    zc = (f.P @ Ti[:3, :3].T + Ti[:3, 3])[:, 2]
    front = zc > 1e-3
    Pn, zc = f.Pn[front], zc[front]
    H, W = g.depth.shape
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    if g.rot == 180:
        u, v = W - 1 - u, H - 1 - v
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    dg = np.zeros(len(u))
    dg[inside] = g.depth[v[inside], u[inside]] * g.scale
    known = inside & (dg > 0)
    vis = known & (np.abs(zc - dg) < OCCL_TOL_M)
    # closing radius grows when g sees the object bigger than f did
    ratio = np.linalg.norm(f.T[:3, 3] - f.P.mean(0)) / max(np.linalg.norm(g.T[:3, 3] - f.P.mean(0)), 1e-3)
    rad = int(np.ceil(max(ratio, 1.0))) + 1
    out, stats = {}, {}
    for i in f.nodes:
        s = Pn == i
        m = np.zeros((H, W), bool)
        m[v[s & vis], u[s & vis]] = True
        if m.any():
            m = binary_closing(binary_dilation(m, iterations=1), iterations=rad)
        out[i] = m
        n_known = int((s & known).sum())
        stats[i] = {"visible_share": round(float((s & vis).sum()) / max(int(s.sum()), 1), 3),
                    "occluded_share_of_known": round(1 - float((s & vis).sum()) / max(n_known, 1), 3)}
    return out, stats


def own_tree(g):
    d = E02 / g.id / "full"
    return (d / "tree_final.json").exists()


def match_own(carried, g, f):
    """For a target with its own e02 tree: best-IoU own node per carried node."""
    _, gm = load_tree(g.frame)
    res = {}
    for i, m in carried.items():
        if i == 0 or not m.any():
            continue
        best, bi = 0.0, None
        for j, mj in gm.items():
            inter = (m & mj).sum()
            if inter:
                iou = inter / (m | mj).sum()
                if iou > best:
                    best, bi = iou, j
        if bi is None:
            res[i] = {"iou": 0.0}
            continue
        ca = np.argwhere(m).mean(0)
        cb = np.argwhere(gm[bi]).mean(0)
        off_px = float(np.linalg.norm(ca - cb))
        zs = g.depth[gm[bi]] * g.scale
        z = float(np.median(zs[zs > 0])) if (zs > 0).any() else np.nan
        res[i] = {"own_node": int(bi), "own_name": g.nodes[bi]["name"] if bi in g.nodes else None,
                  "iou": round(float(best), 3), "centroid_offset_px": round(off_px, 1),
                  "centroid_offset_mm": round(off_px * z / g.intr["fx"] * 1000, 1)}
    return res


def fig_views(f, frames_by_step, max_targets=4):
    ids = node_order(f.nodes)
    run = Run.load(f.run_dir)
    steps = sorted(int(p.name) for p in (f.run_dir / "steps").iterdir() if (p / "rgb.png").exists())
    cand = []
    for s in steps:
        g = frames_by_step.get(s) or Target(f.frame["run"], s)
        if g.depth is None:
            continue
        carried, stats = carry(f, g)
        share = stats[0]["visible_share"]
        cand.append((s, g, carried, stats, share, view_angle_deg(f, g)))
    self_c = [c for c in cand if c[0] == f.frame["step"]]
    others = sorted([c for c in cand if c[0] != f.frame["step"] and c[4] >= 0.25 and c[5] <= MAX_ANGLE],
                    key=lambda c: -c[4])
    skipped = {str(c[0]): {"visible_share": c[4], "angle_deg": round(c[5], 1)}
               for c in cand if c[0] != f.frame["step"] and c not in others}
    chosen = self_c + others[:max_targets]
    panels, numbers = [], {}
    for s, g, carried, stats, share, angle in chosen:
        box = C.expand_box(_bbox(carried[0]) if carried[0].any() else (0, 0, g.rgb.shape[1], g.rgb.shape[0]),
                           g.rgb.shape[:2], 0.12)
        ms = [crop(carried[i], box) for i in ids]
        img = C.overlay(crop(g.rgb, box), ms, alpha=0.35, colors=[f.colors[i] for i in ids], thick=2)
        rec = {"step": s, "object_visible_share": share, "angle_to_face_deg": round(angle, 1),
               "nodes": {str(i): stats[i] for i in ids}}
        if s != f.frame["step"] and hasattr(g, "masks"):
            own = [crop(m, box) for k, m in g.masks.items() if k != 0]
            img = C.outlines(img, own, [(255, 255, 255)] * len(own), thick=1, dim=0.0)
            rec["match_own_tree"] = {str(k): v for k, v in match_own(carried, g, f).items()}
        dist = np.linalg.norm(g.T[:3, 3] - f.P[f.Pn == 0].mean(0)) * 100
        tag = "round trip (own view)" if s == f.frame["step"] else f"→ step {s}"
        tag += f" · {dist:.0f} cm · {angle:.0f}° · {100 * share:.0f}% visible"
        if s != f.frame["step"] and hasattr(g, "masks"):
            tag += " · white = own e02 masks"
        panels.append(C.label(letterbox(img, 460, 330), tag, size=13))
        numbers[str(s)] = rec
    cols = min(len(panels), 3)
    h = max(p.shape[0] for p in panels)
    img = C.grid([pad_to(p, h) for p in panels], cols=cols, pad=6)
    img = np.concatenate([img, legend(f.nodes, f.colors, img.shape[1])], 0)
    numbers["skipped"] = skipped
    return C.title_band(img, f"{f.id} — nodes carried through 3D into the other steps of its run "
                             f"(≤{MAX_ANGLE}° to the face)"), numbers


def _bbox(m):
    ys, xs = np.nonzero(m)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


class Target:
    """A step of a run without an e02 tree: only rgb, depth, pose."""
    def __init__(self, run, step):
        d = C.RUNS_DIR / run / "steps" / f"{step:03d}"
        self.id = f"{run}:{step}"
        self.depth = np.load(d / "depth_aligned.npy") if (d / "depth_aligned.npy").exists() else None
        if self.depth is None:
            return
        s = json.load(open(d / "step.json"))
        self.T = np.array(s["T_base_cam"])
        self.rot = int(s.get("rgb_rotation_deg", 0) or 0)
        self.intr, self.scale = session_intrinsics(C.RUNS_DIR / run)
        from PIL import Image
        self.rgb = np.array(Image.open(d / "rgb.png").convert("RGB"))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="*")
    args = ap.parse_args()
    EVID.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    all_frames = {fr["id"]: F(fr) for fr in C.frames()}
    todo = [all_frames[fr["id"]] for fr in C.frames(args.frames)]
    numbers = {}
    for f in todo:
        print(f"== {f.id}", flush=True)
        kd = cKDTree(f.cloud) if len(f.cloud) else None
        from PIL import Image
        Image.fromarray(fig_depth(f)).save(EVID / f"{f.id}_depth.jpg", quality=90)
        img, d_mm, pl_mm = fig_3d(f, kd)
        Image.fromarray(img).save(EVID / f"{f.id}_3d.jpg", quality=90)
        same_run = {g.frame["step"]: g for g in all_frames.values() if g.frame["run"] == f.frame["run"]}
        img, views = fig_views(f, same_run)
        Image.fromarray(img).save(EVID / f"{f.id}_views.jpg", quality=90)
        sel = f.Pn != 0
        per_node = {}
        for i in node_order(f.nodes):
            s = f.Pn[sel] == i
            nd = f.nodes[i]
            per_node[str(i)] = {"name": nd["name"], "depth": nd["depth"],
                                "px_with_depth": nd["px_with_depth"],
                                "off_side_plane_gt_8mm": (round(float((pl_mm[s] > BLEED_MM).mean()), 3)
                                                          if s.any() else None),
                                "off_side_plane_median_mm": (round(float(np.median(pl_mm[s])), 1)
                                                             if s.any() else None),
                                "median_to_fused_mm": (round(float(np.median(d_mm[s])), 1)
                                                       if s.any() and kd is not None else None)}
        numbers[f.id] = {"run": f.frame["run"], "step": f.frame["step"],
                         "fused": bool(f.frame.get("recorded_mask", True)),
                         "nodes": per_node, "views": views}
    path = HERE / "evidence" / "projection_quality.json"
    old = json.load(open(path)) if path.exists() else {}
    old.update(numbers)
    json.dump(old, open(path, "w"), indent=1, ensure_ascii=False)
    C.run_record(OUT, [{"id": f.id, "run": f.frame["run"], "step": f.frame["step"]} for f in todo],
                 {"name": "carry node points via T_base_cam + colour intrinsics, occlusion vs target depth"},
                 {"occlusion_tol_m": OCCL_TOL_M, "bleed_mm": BLEED_MM})


if __name__ == "__main__":
    main()
