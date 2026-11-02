"""e03: interactive viewer — the tree's nodes on the run's cloud (Viser, in the browser).

Look: the run's cloud (slightly dimmed, true colours); every shown node paints its own
points in its level's colour, sits in a transparent box with bright edges and carries a
name label with a level-coloured marker.

Drill-down (default):
  - the sides are shown; hover = yellow outline, click = open it;
  - an open node turns into a faint shell and lets every hover and click PASS THROUGH to
    its children (its click binding is removed, so the browser never picks it);
  - click inside the open node where no child is -> close it (back one level); click
    inside an ancestor's box -> back to that ancestor. That part is picked on the
    server: a scene-wide click gives the ray, the ray is tested against the boxes of
    the open path, the deepest one hit wins.
Right panel: name, size, parents, children, the source frame crop with the node and its
children painted, Back / Clear.

Run:  p inspection/investigation/iwm/e03_tree_3d/viewer.py --frames c5.6 [c5.7 c9.6] [--port 8080]
then open http://localhost:8080
"""
import argparse
import itertools
import json
import time

import numpy as np
import trimesh
import viser

from inspection.investigation.iwm.e01_all_masks import common as C
from inspection.investigation.iwm.e03_tree_3d.project import DATA, load_tree
from inspection.record.run import Run

#: one saturated colour per level (1 sides, 2 things, 3 parts, 4+ details) + its marker
LEVEL_COLORS = [(30, 160, 255), (255, 140, 0), (40, 215, 110), (255, 60, 170),
                (160, 90, 255), (250, 220, 30)]
LEVEL_MARK = ["🟦", "🟧", "🟩", "🟥", "🟪", "🟨"]
FILL, OPEN_FILL, DIM_FILL = 0.32, 0.05, 0.10
EDGE_PX, OPEN_EDGE_PX = 2.5, 1.5
TINT, CLOUD_DIM = 0.6, 0.7
BG = (18, 20, 26)
GREY = (120, 125, 135)


def level_color(depth):
    return LEVEL_COLORS[(depth - 1) % len(LEVEL_COLORS)]


def level_mark(depth):
    return LEVEL_MARK[(depth - 1) % len(LEVEL_MARK)]


def oriented_box(P, pad=0.002, lo_q=1.0, hi_q=99.0):
    """PCA box around points: (center (3,), R (3x3 columns=axes), extents (3,))."""
    c = P.mean(0)
    Q = P - c
    if len(P) > 20000:
        Q = Q[:: len(P) // 20000]
    _, _, vt = np.linalg.svd(Q, full_matrices=False)
    R = vt.T
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    proj = (P - c) @ R
    lo, hi = np.percentile(proj, lo_q, axis=0), np.percentile(proj, hi_q, axis=0)
    ext = np.maximum(hi - lo, 0.003) + 2 * pad
    center = c + R @ ((lo + hi) / 2)
    return center, R, ext


def box_mesh(center, R, ext):
    m = trimesh.creation.box(extents=ext)
    v = (R @ m.vertices.T).T + center
    return v.astype(np.float32), m.faces.astype(np.int32)


def box_edges(center, R, ext):
    """The 12 edges of the box as (12, 2, 3)."""
    signs = list(itertools.product((-1, 1), repeat=3))
    corner = {s: center + R @ (np.array(s) * ext / 2) for s in signs}
    segs = [(corner[a], corner[b]) for a, b in itertools.combinations(signs, 2)
            if sum(x != y for x, y in zip(a, b)) == 1]
    return np.array(segs, np.float32)


def ray_hits_box(o, d, center, R, ext):
    """Slab test in the box frame; distance along the ray or None."""
    o2, d2 = R.T @ (o - center), R.T @ d
    half = ext / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-half - o2) / d2, (half - o2) / d2
    t1 = np.where(np.isnan(t1), -np.inf, t1)
    t2 = np.where(np.isnan(t2), np.inf, t2)
    tmin, tmax = np.minimum(t1, t2).max(), np.maximum(t1, t2).min()
    return float(max(tmin, 0.0)) if tmax >= max(tmin, 0.0) else None


def tint(cols, color, k):
    return (cols.astype(np.float32) * (1 - k) + np.array(color, np.float32) * k).astype(np.uint8)


class FrameData:
    def __init__(self, frame, variant):
        self.frame = frame
        self.id = frame["id"]
        d = DATA / frame["id"] / variant
        z = np.load(d / "nodes.npz")
        self.points, self.colors, self.node_of = z["points"], z["colors"], z["node"]
        self.meta = json.load(open(d / "nodes.json"))
        self.nodes = {n["id"]: n for n in self.meta["nodes"]}
        self.tree, self.masks = load_tree(frame, variant)
        self.rgb = C.load_rgb(frame)
        step = json.load(open(C.frame_paths(frame)["step"]))
        self.T_base_cam = np.array(step["T_base_cam"])
        run = Run.load(C.RUNS_DIR / frame["run"])
        fused = run.fused()
        self.cloud = fused[0].astype(np.float32) if fused else np.zeros((0, 3), np.float32)
        self.cloud_colors = fused[1] if fused and fused[1] is not None else None

    def parents(self, i):
        chain = []
        p = self.nodes[i]["parent"]
        while p is not None:
            chain.append(p)
            p = self.nodes[p]["parent"]
        return chain

    def crop(self, i, max_w=640):
        """Source frame around the node: the node lightly painted, its children painted
        in their level colour, its parents outlined."""
        nd = self.nodes[i]
        chain = self.parents(i)
        anchor = chain[0] if chain else i
        x0, y0, x1, y1 = C.expand_box(self.nodes[anchor]["bbox"], self.rgb.shape[:2], 0.08)
        cut = lambda k: self.masks[k][y0:y1, x0:x1]            # noqa: E731
        img = C.overlay(self.rgb[y0:y1, x0:x1], [cut(i)], alpha=0.25,
                        colors=[level_color(nd["depth"])], thick=2)
        kids = [k for k in nd["children"] if k in self.masks]
        if kids:
            img = C.overlay(img, [cut(k) for k in kids], alpha=0.5,
                            colors=[level_color(self.nodes[k]["depth"]) for k in kids], thick=1)
        if chain:
            img = C.outlines(img, [cut(k) for k in chain],
                             [level_color(self.nodes[k]["depth"]) for k in chain], thick=1, dim=0.0)
        img = C.label(img, f"#{i} {nd['name']}  ·  {self.id} step {self.frame['step']}", size=15)
        return C.resize(img, max_w) if img.shape[1] > max_w else img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    frames = [FrameData(f, args.variant) for f in C.frames(args.frames)]
    by_id = {fd.id: fd for fd in frames}

    server = viser.ViserServer(port=args.port, label="IWM tree in 3D")
    server.gui.configure_theme(dark_mode=True, control_layout="collapsible",
                               control_width="large", show_logo=False,
                               brand_color=LEVEL_COLORS[0])
    server.scene.set_up_direction("+z")
    server.scene.set_background_image(np.full((4, 4, 3), BG, np.uint8))
    maxd = max(n["depth"] for fd in frames for n in fd.nodes.values())

    handles = {}          # (frame id, node id) -> dict of scene handles + geometry
    clouds = {}
    state = {"stack": [], "selected": None}

    # ---------------------------------------------------------------- GUI
    with server.gui.add_folder("Tree"):
        drill_cb = server.gui.add_checkbox("drill-down", initial_value=True)
        path_md = server.gui.add_markdown("")
        legend_md = server.gui.add_markdown("  ".join(
            f"{level_mark(d)} depth {d}" for d in range(1, maxd + 1)))
    with server.gui.add_folder("Overview (drill-down off)", expand_by_default=False):
        level_cb = {d: server.gui.add_checkbox(f"{level_mark(d)} depth {d}", initial_value=d <= 2)
                    for d in range(1, maxd + 1)}
    with server.gui.add_folder("Look", expand_by_default=False):
        names_cb = server.gui.add_checkbox("names", initial_value=True)
        fill_sl = server.gui.add_slider("box fill", 0.0, 0.8, 0.01, FILL)
        tint_sl = server.gui.add_slider("paint parts", 0.0, 1.0, 0.05, TINT)
        dim_sl = server.gui.add_slider("cloud brightness", 0.2, 1.0, 0.05, CLOUD_DIM)
        cloud_size = server.gui.add_slider("cloud point size (mm)", 0.5, 4.0, 0.25, 1.25)
    with server.gui.add_folder("Frames"):
        frame_cb = {fd.id: server.gui.add_checkbox(f"{fd.id}  {fd.frame['run']} step {fd.frame['step']}",
                                                   initial_value=(k == 0))
                    for k, fd in enumerate(frames)}
        cloud_cb = server.gui.add_checkbox("run cloud", initial_value=True)
    with server.gui.add_folder("Selection"):
        sel_md = server.gui.add_markdown("_hover a side, click to open it_")
        back_btn = server.gui.add_button("◀ Back")
        clear_btn = server.gui.add_button("Clear")
        sel_img = server.gui.add_image(np.full((8, 8, 3), BG, np.uint8), label="source frame")

    # ---------------------------------------------------------------- scene
    runs_loaded = {}
    for fd in frames:
        run = fd.frame["run"]
        if run in runs_loaded:                 # one cloud per run, shared by its frames
            clouds[fd.id] = runs_loaded[run]
        elif len(fd.cloud):
            base = fd.cloud_colors if fd.cloud_colors is not None else np.full((len(fd.cloud), 3), 160, np.uint8)
            h = server.scene.add_point_cloud(
                f"/cloud/{run}", fd.cloud, (base * CLOUD_DIM).astype(np.uint8),
                point_size=cloud_size.value / 1000, point_shape="circle")
            clouds[fd.id] = runs_loaded[run] = {"h": h, "base": base}
        for i, nd in fd.nodes.items():
            if i == 0 or nd["n_points"] < 30:
                continue
            sel = fd.node_of == i
            P, cols = fd.points[sel], fd.colors[sel]
            center, R, ext = oriented_box(P)
            v, f = box_mesh(center, R, ext)
            col = level_color(nd["depth"])
            vis = nd["depth"] == 1
            root = f"/nodes/{fd.id}/{i}"
            hb = server.scene.add_mesh_simple(f"{root}/box", v, f, color=col, opacity=FILL,
                                              flat_shading=True, side="double", visible=vis)
            he = server.scene.add_line_segments(f"{root}/edges", box_edges(center, R, ext), col,
                                                thickness=EDGE_PX, thickness_units="screen",
                                                visible=vis)
            top = center + R @ np.array([0, 0, ext[2] / 2]) + np.array([0, 0, 0.004])
            hl = server.scene.add_label(f"{root}/label", f"{level_mark(nd['depth'])} {nd['name'] or f'#{i}'}",
                                        position=top, visible=vis)
            # the part's own points, painted in its level colour; deeper = a bit bigger,
            # so children draw over their parent's paint
            hp = server.scene.add_point_cloud(f"{root}/points", P, tint(cols, col, TINT),
                                              point_size=0.0016 + 0.0002 * nd["depth"],
                                              point_shape="circle", visible=vis)

            def _on_click(event, fd=fd, i=i):
                state["node_click"] = time.time()
                print(f"node click {fd.id} #{i} {fd.nodes[i]['name']}", flush=True)
                select(fd, i, push=True)
            hb.on_click(_on_click)
            handles[(fd.id, i)] = {"box": hb, "edges": he, "label": hl, "points": hp,
                                   "P_cols": cols, "depth": nd["depth"], "color": col,
                                   "geom": (center, R, ext), "cb": _on_click, "bound": True}

    # ---------------------------------------------------------------- behaviour
    def open_path():
        sel = state["selected"]
        path = [(fd.id, i) for fd, i in state["stack"]]
        return path + [(sel[0].id, sel[1])] if sel else path

    def shown_set():
        """Drill-down: the sides, plus the children of every node on the open path."""
        shown = {(fd.id, i) for fd in frames for i, nd in fd.nodes.items() if nd["depth"] == 1}
        for fid, i in open_path():
            shown.update((fid, c) for c in by_id[fid].nodes[i]["children"])
            shown.add((fid, i))
        return shown

    def role(key):
        """open | ancestor | child (of the open node) | other | None (no selection)."""
        sel = state["selected"]
        if not sel or not drill_cb.value:
            return None
        if key == (sel[0].id, sel[1]):
            return "open"
        if key in open_path():
            return "ancestor"
        if key[0] == sel[0].id and key[1] in sel[0].nodes[sel[1]]["children"]:
            return "child"
        return "other"

    def refresh():
        shown = shown_set() if drill_cb.value else None
        path = set(open_path())
        for key, h in handles.items():
            fid, i = key
            if shown is not None:
                on = key in shown and frame_cb[fid].value
            else:
                on = level_cb[h["depth"]].value and frame_cb[fid].value
            r = role(key)
            # --- look
            if r == "open":
                fill, edge, ecol, px = OPEN_FILL, OPEN_EDGE_PX, (235, 235, 235), False
            elif r == "ancestor":
                fill, edge, ecol, px = 0.0, OPEN_EDGE_PX, GREY, False
            elif r == "other":
                fill, edge, ecol, px = DIM_FILL * fill_sl.value / FILL, EDGE_PX * 0.6, GREY, False
            else:                                   # child of the open node, or nothing open
                fill, edge, ecol, px = fill_sl.value, EDGE_PX, h["color"], True
            h["box"].visible = on and fill > 0.001
            h["box"].opacity = max(fill, 0.001)
            h["edges"].visible = on
            if h.get("ecol") != ecol:
                h["edges"].colors = np.tile(np.array(ecol, np.uint8), (12, 2, 1))
                h["ecol"] = ecol
            h["edges"].thickness = edge
            h["points"].visible = on and px
            h["label"].visible = on and names_cb.value and r != "ancestor"
            # --- clicks: nodes on the open path let everything pass through
            want = on and key not in path
            if want != h["bound"]:
                (h["box"].on_click(h["cb"]) if want else h["box"].remove_click_callback(h["cb"]))
                h["bound"] = want
        sel = state["selected"]
        if drill_cb.value:
            crumbs = [f"{level_mark(by_id[f].nodes[i]['depth'])} {by_id[f].nodes[i]['name']}"
                      for f, i in open_path()]
            path_md.content = ("**" + "  ›  ".join(crumbs) + "**  \n_click inside the open box, "
                               "away from its children, to go back_") if crumbs \
                else "_hover a side, click to open it_"
        else:
            path_md.content = "_overview: tick depths below_"
        shown_runs = {fd.frame["run"] for fd in frames if frame_cb[fd.id].value}
        for fd in frames:
            if fd.id in clouds:
                clouds[fd.id]["h"].visible = cloud_cb.value and fd.frame["run"] in shown_runs

    def select(fd, i, push):
        """push=True is a click on a node: the open path becomes the node's ancestry."""
        if push:
            state["stack"] = [(fd, a) for a in reversed(fd.parents(i)) if a != 0]
        state["selected"] = (fd, i)
        nd = fd.nodes[i]
        chain = fd.parents(i)
        ext = nd.get("extent_mm")
        kids = [k for k in nd["children"] if k in fd.nodes]
        lines = [f"### {level_mark(nd['depth'])} {nd['name']}",
                 f"`#{i}` · {fd.id} · `{fd.frame['run']}` step {fd.frame['step']} · depth {nd['depth']}",
                 f"{ext[0]:.0f} × {ext[1]:.0f} × {ext[2]:.0f} mm · {nd['n_points']} points" if ext else "",
                 (f'text: "{nd["text"]}"' if nd.get("text") else ""),
                 "**parents:** " + (" › ".join(fd.nodes[k]["name"] for k in reversed(chain) if k != 0) or "—"),
                 "**children:** " + (", ".join(f"{level_mark(fd.nodes[k]['depth'])} {fd.nodes[k]['name']}"
                                               for k in kids) or "— (a leaf)")]
        sel_md.content = "  \n".join(l for l in lines if l)
        sel_img.image = fd.crop(i)
        refresh()

    def back():
        if state["stack"]:
            fd, i = state["stack"].pop()
            select(fd, i, push=False)
        else:
            clear()

    def clear():
        state["selected"] = None
        state["stack"].clear()
        sel_md.content = "_hover a side, click to open it_"
        sel_img.image = np.full((8, 8, 3), BG, np.uint8)
        refresh()

    @server.scene.on_click()
    def _(event):
        """Clicks that no part caught: inside an open box -> back to that level."""
        time.sleep(0.05)                                   # let a node click land first
        print("scene click", event.ray_origin, event.ray_direction, flush=True)
        if time.time() - state.get("node_click", 0) < 0.5 or not drill_cb.value:
            return
        o, d = np.array(event.ray_origin), np.array(event.ray_direction)
        hit = [k for k in open_path() if ray_hits_box(o, d, *handles[k]["geom"]) is not None] \
            if all(k in handles for k in open_path()) else []
        if not hit:
            return
        deepest = hit[-1]                                   # open path is ordered root -> open
        if deepest == open_path()[-1]:
            back()                                          # inside the open box: up one level
        else:
            while open_path()[-1] != deepest:               # inside an ancestor: back to it
                back()

    back_btn.on_click(lambda _: back())
    clear_btn.on_click(lambda _: clear())
    for cb in list(level_cb.values()) + list(frame_cb.values()) + [names_cb, cloud_cb, drill_cb]:
        cb.on_update(lambda _: refresh())
    fill_sl.on_update(lambda _: refresh())

    @tint_sl.on_update
    def _(_):
        for h in handles.values():
            h["points"].colors = tint(h["P_cols"], h["color"], tint_sl.value)

    @dim_sl.on_update
    def _(_):
        for c in runs_loaded.values():
            c["h"].colors = (c["base"] * dim_sl.value).astype(np.uint8)

    @cloud_size.on_update
    def _(_):
        for c in runs_loaded.values():
            c["h"].point_size = cloud_size.value / 1000

    refresh()

    # camera: start where the first frame's camera was, looking at the object
    fd0 = frames[0]
    target = fd0.points[fd0.node_of == 0].mean(0) if (fd0.node_of == 0).any() else fd0.cloud.mean(0)
    cam_pos = fd0.T_base_cam[:3, 3]
    cam_pos = target + (cam_pos - target) * 2.6 + np.array([0, 0, 0.12])

    @server.on_client_connect
    def _(client):
        try:
            client.camera.up_direction = np.array([0.0, 0.0, 1.0])
            client.camera.position = cam_pos
            client.camera.look_at = target
            client.camera.fov = np.deg2rad(40)
        except Exception as e:                       # noqa: BLE001
            print("camera setup:", e)

    print(f"viewer on http://localhost:{args.port}  frames {[fd.id for fd in frames]}  "
          f"nodes {len(handles)}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
