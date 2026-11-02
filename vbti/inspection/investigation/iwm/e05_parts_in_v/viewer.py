"""e05 viewer: walk the coloured V of one run (S/T/V after all views), search parts, see the
view computed for a part or a set of parts.

Scene: e04's marching-cubes mesh of V in grey; the labelled surface rows as a coloured point
cloud (one colour per part at the chosen tree level); step cameras as frustums; the proposed
view of the selected part as a magenta frustum carrying V's render from that pose.
GUI: level slider (colour by the ancestor at that depth), view filter (parts detected from that
view only), search box, part selector (voxels light up, info, its detecting frame's crop, the
proposed view; "look from proposed" moves this browser's camera there), a set query
("P3, P7, P9" -> the view for the set), "look from step". Click the surface -> the part under
the click and its ancestor chain.

Run:  p vbti/inspection/investigation/iwm/e05_parts_in_v/viewer.py [--run 2209-compute-ray-demo5] [--mm 1] [--mode agent] [--port 8083]
      p .../viewer.py --render-only        writes evidence/labels_<run>_<mm>mm.jpg (capture poses + top + side)
"""
import argparse
import json
import time

import numpy as np
import open3d as o3d
import trimesh
import viser
import viser.transforms as vtf
from PIL import Image

from inspection.investigation.iwm.e01_all_masks import common as C
from inspection.investigation.iwm.e04_voxel_scene.build import OUT as E04_OUT
from inspection.investigation.iwm.e05_parts_in_v import common as C5
from inspection.investigation.iwm.e05_parts_in_v import lift as L
from inspection.investigation.iwm.e05_parts_in_v import views as Vw
from inspection.investigation.iwm.e05_parts_in_v.stv import STV, Surface

POINT = 0.0014


def load_all(run, mm, mode):
    vbg, size = C5.V(run, mm)
    surf = Surface.from_vbg(vbg, size, C5.trunc_m(size))
    d = C5.DATA / run / f"{mm}mm" / f"keep_{mode}" / f"after_{C5.ORDER[run][-1]}"
    if not d.exists():
        d = C5.DATA / run / f"{mm}mm" / "keep_baseline" / f"after_{C5.ORDER[run][-1]}"
    stv = STV.load(d, surf)
    mesh = o3d.io.read_triangle_mesh(str(E04_OUT / run / "scene" / f"{mm:g}mm" / "mesh.ply"))
    steps, intr, scale = C5.run_steps(run)
    return vbg, size, surf, stv, mesh, steps, intr, d


def palette(stv):
    ids = sorted(stv.parts)
    pal = C._palette(len(ids), seed=3)
    return {p: np.array(pal[i], np.uint8) for i, p in enumerate(ids)}


def ancestor_at(stv, part, depth):
    chain = [part] + stv.ancestors(part)          # deepest first
    chain = chain[::-1]                           # root first
    return chain[min(depth, len(chain) - 1)]


def colour_rows(stv, pal, level, view_filter):
    rows = np.nonzero(stv.label != -1)[0]
    lab = stv.label[rows]
    keep = np.ones(len(rows), bool)
    if view_filter != "all":
        ok = {p for p in stv.parts if any(v == view_filter for v, _ in stv.parts[p].detections)}
        ok |= {d for p in ok for d in stv.descendants(p)}
        keep = np.isin(lab, list(ok))
    rows, lab = rows[keep], lab[keep]
    shown = np.array([ancestor_at(stv, int(l), level) for l in np.unique(lab)])
    remap = dict(zip(np.unique(lab).tolist(), shown.tolist()))
    cols = np.stack([pal[remap[int(l)]] for l in lab]) if len(lab) else np.zeros((0, 3), np.uint8)
    return rows, cols


def label_renders(run, mm, mode, path):
    """For the report: V's labels from each capture pose, from the top and from the side."""
    from inspection.investigation.iwm.e05_parts_in_v.keep import label_figure
    vbg, size, surf, stv, mesh, steps, intr, _ = load_all(run, mm, mode)
    views = [C5.load_view(v) for v in C5.ORDER[run]]
    P = surf.centres(stv.parts[stv.root].rows)
    c = P.mean(0)
    ext = np.ptp(P, axis=0).max()
    K = C5.K_of(intr)
    extra = []
    for name, n, up in (("top", np.array([0, 0, 1.0]), np.array([1.0, 0, 0])), ("side", np.array([0, -1.0, 0]), np.array([1.0, 0, 0]))):
        T = Vw.pose_for(c, n, up, max(0.25, ext * 1.1))
        extra.append({"id": name, "T": T, "K": K, "rot": 0})
    label_figure(stv, vbg, size, views + extra, path, f"{run} at {mm} mm, {mode}: V's labels after {len(views)} views")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="2209-compute-ray-demo5")
    ap.add_argument("--mm", type=int, default=1)
    ap.add_argument("--mode", default="agent")
    ap.add_argument("--port", type=int, default=8083)
    ap.add_argument("--render-only", action="store_true")
    args = ap.parse_args()
    if args.render_only:
        for run in C5.ORDER:
            label_renders(run, args.mm, args.mode, C5.EVID / f"labels_{run}_{args.mm}mm.jpg")
            print("wrote", C5.EVID / f"labels_{run}_{args.mm}mm.jpg")
        return

    vbg, size, surf, stv, mesh, steps, intr, src = load_all(args.run, args.mm, args.mode)
    pal = palette(stv)
    K_half, K_full = C5.K_of(intr, 0.5), C5.K_of(intr)
    flipped = {st["id"]: int(json.load(open(C.RUNS_DIR / args.run / "steps" / f"{st['id']:03d}" / "step.json"))
                             .get("rgb_rotation_deg", 0) or 0) == 180 for st in steps}
    max_depth = max(stv.depth(p) for p in stv.parts)
    views_cache = {}

    def view_of(fid):
        if fid not in views_cache:
            views_cache[fid] = C5.load_view(fid)
        return views_cache[fid]

    rs = o3d.t.geometry.RaycastingScene()
    rs.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(dark_mode=True)
    server.scene.set_up_direction("+z")
    F = np.asarray(mesh.triangles)
    grey = (np.asarray(mesh.vertex_colors) * 255 * 0.55 + 40).clip(0, 255).astype(np.uint8)
    tm = trimesh.Trimesh(np.asarray(mesh.vertices), np.concatenate([F, F[:, ::-1]]), vertex_colors=grey, process=False)
    server.scene.add_mesh_trimesh("/V", tm, cast_shadow=False, receive_shadow=False)

    fov = 2 * np.arctan(intr["height"] / 2 / intr["fy"])
    aspect = intr["width"] / intr["height"]
    frust = []
    for st in steps:
        frust.append(server.scene.add_camera_frustum(f"/cams/{st['id']}", fov=fov, aspect=aspect, scale=0.04,
                                                     wxyz=vtf.SO3.from_matrix(st["T"][:3, :3]).wxyz, position=st["T"][:3, 3],
                                                     image=st["rgb"][::8, ::8], color=(255, 200, 0), visible=False))

    with server.gui.add_folder("Parts"):
        level_sl = server.gui.add_slider("tree level", 0, max_depth, 1, max_depth)
        view_dd = server.gui.add_dropdown("detected from", ["all"] + C5.ORDER[args.run], initial_value="all")
        search_tx = server.gui.add_text("search", initial_value="")
        search_bt = server.gui.add_button("search")
        hits_md = server.gui.add_markdown("")
        names = [f"P{p} {'· ' * stv.depth(p)}{stv.parts[p].name}" for p in sorted(stv.parts, key=lambda p: stv.path(p))]
        part_dd = server.gui.add_dropdown("part", ["-"] + names, initial_value="-")
        info_md = server.gui.add_markdown(f"**{args.run}** · {args.mm} mm · {args.mode} · {len(stv.parts)} parts · "
                                          f"{(stv.label != -1).sum():,} labelled rows · {len(stv.conflicts)} conflicts")
        crop_img = server.gui.add_image(np.full((8, 8, 3), 30, np.uint8), label="detecting frame")
    with server.gui.add_folder("View for the part"):
        view_md = server.gui.add_markdown("_select a part_")
        render_img = server.gui.add_image(np.full((8, 8, 3), 30, np.uint8), label="V from the proposed pose")
        look_bt = server.gui.add_button("look from proposed")
    with server.gui.add_folder("View for a set"):
        set_tx = server.gui.add_text("parts (P ids, comma separated)", initial_value="")
        set_bt = server.gui.add_button("view for set")
        set_md = server.gui.add_markdown("")
    with server.gui.add_folder("V"):
        cams_cb = server.gui.add_checkbox("step cameras", initial_value=False)
        look_dd = server.gui.add_dropdown("look from step", ["-"] + [str(st["id"]) for st in steps], initial_value="-")
    with server.gui.add_folder("Clicked point"):
        hit_md = server.gui.add_markdown("_click the surface_")

    state = {"part": None, "proposed": None}

    def recolour():
        rows, cols = colour_rows(stv, pal, int(level_sl.value), view_dd.value)
        server.scene.add_point_cloud("/parts", surf.centres(rows).astype(np.float32), cols, point_size=POINT, point_shape="rounded")

    recolour()
    level_sl.on_update(lambda _: recolour())
    view_dd.on_update(lambda _: recolour())

    @cams_cb.on_update
    def _(_):
        for f in frust:
            f.visible = cams_cb.value

    def move_camera(client, T):
        R = T[:3, :3]
        with client.atomic():
            client.camera.fov = fov
            client.camera.position = T[:3, 3]
            client.camera.wxyz = vtf.SO3.from_matrix(R).wxyz
            client.camera.look_at = T[:3, 3] + R[:, 2] * 0.2
            client.camera.up_direction = -R[:, 1]

    @look_dd.on_update
    def _(event):
        if look_dd.value == "-" or event.client is None:
            return
        st = next(x for x in steps if str(x["id"]) == look_dd.value)
        T = st["T"].copy()
        if flipped[st["id"]]:
            T[:3, :3] = T[:3, :3] @ np.diag([-1, -1, 1])
        move_camera(event.client, T)

    @search_bt.on_click
    def _(_):
        hits = stv.search(search_tx.value)
        hits_md.content = ("\n".join(f"- P{p} {stv.path(p)} ({len(stv.parts[p].rows)} rows)" for p in hits[:20]) or "_no hits_")
        if hits:
            part_dd.value = next(n for n in names if n.startswith(f"P{hits[0]} "))

    def select(part):
        state["part"] = part
        p = stv.parts[part]
        server.scene.add_point_cloud("/selected", surf.centres(p.rows).astype(np.float32),
                                     np.tile(np.array([[255, 255, 255]], np.uint8), (len(p.rows), 1)), point_size=POINT * 1.6)
        det = ", ".join(f"{v}#{n}" for v, n in p.detections) or "none"
        info_md.content = (f"**P{part} {p.name}**  \n{stv.path(part)}  \n{len(p.rows):,} rows · depth {stv.depth(part)} · "
                           f"children {len(p.children)} · detections {det}")
        if p.detections:
            v = view_of(p.detections[0][0])
            nd = v["nodes"].get(p.detections[0][1])
            if nd is not None:
                rgb = L._upright(v["rgb"], v["rot"])
                m = L._upright(v["masks"][nd["id"]], v["rot"])
                h, w = rgb.shape[:2]
                x0, y0, x1, y1 = C.expand_box(nd["bbox"], (h, w), 0.3)
                crop_img.image = C.resize(C.outlines(rgb[y0:y1, x0:x1], [m[y0:y1, x0:x1]], [(0, 255, 0)], thick=2, dim=0.1), 480)
        r = Vw.view_for(stv, vbg, size, part, steps, K_half)
        T = np.array(r["best"]["T"])
        state["proposed"] = T
        _, _, col = C5.cast(vbg, size, T, K_half, Vw.RW, Vw.RH, color=True)
        render_img.image = col
        server.scene.add_camera_frustum("/proposed", fov=fov, aspect=aspect, scale=0.05, wxyz=vtf.SO3.from_matrix(T[:3, :3]).wxyz,
                                        position=T[:3, 3], image=col[::2, ::2], color=(255, 0, 255))
        server.scene.add_line_segments("/proposed/normal", np.array([[r["centre"], np.array(r["centre"]) + np.array(r["normal"]) * 0.05]]),
                                       np.array([[[255, 0, 255]] * 2]), line_width=3.0)
        bd = r["best_detecting_step"]
        view_md.content = "\n".join([
            f"**extent** {r['extent_mm'][0]:.0f} × {r['extent_mm'][1]:.0f} mm · **normal** from {r['normal_source']}"
            + (f" (face P{r['face']} {r['face_name']}, planar {r['planar']:.2f})" if r['face'] is not None else ""),
            f"**standoff** {r['standoff_m']:.2f} m · best candidate **{r['best']['name']}**: {Vw.fmt(r['best'])}",
            f"normal pose: {Vw.fmt(r['normal_score'])}",
            f"detecting {bd['name']}: {Vw.fmt(bd)}" if bd else "_no recorded step detected this part_",
            f"computed beats the step: **{r['computed_beats_step']}**"])

    @part_dd.on_update
    def _(_):
        if part_dd.value == "-":
            return
        select(int(part_dd.value.split()[0][1:]))

    @look_bt.on_click
    def _(event):
        if state["proposed"] is not None and event.client is not None:
            move_camera(event.client, state["proposed"])

    @set_bt.on_click
    def _(event):
        try:
            parts = [int(x.strip().lstrip("Pp#")) for x in set_tx.value.split(",") if x.strip()]
            parts = [p for p in parts if p in stv.parts]
        except ValueError:
            parts = []
        if len(parts) < 2:
            set_md.content = "_give at least two part ids_"
            return
        r = Vw.view_for_set(stv, vbg, size, parts, steps, K_half)
        T = np.array(r["best"]["T"])
        state["proposed"] = T
        _, _, col = C5.cast(vbg, size, T, K_half, Vw.RW, Vw.RH, color=True)
        render_img.image = col
        server.scene.add_camera_frustum("/proposed", fov=fov, aspect=aspect, scale=0.05, wxyz=vtf.SO3.from_matrix(T[:3, :3]).wxyz,
                                        position=T[:3, 3], image=col[::2, ::2], color=(255, 0, 255))
        rows = np.unique(np.concatenate([stv.parts[p].rows for p in parts]))
        server.scene.add_point_cloud("/selected", surf.centres(rows).astype(np.float32),
                                     np.tile(np.array([[255, 255, 255]], np.uint8), (len(rows), 1)), point_size=POINT * 1.6)
        per = ", ".join(f"{k} {v:.2f}" for k, v in r["best"]["per_part"].items())
        set_md.content = (f"**{', '.join(r['parts'])}**  \nbest **{r['best']['name']}**: min visible {r['best']['min_visible']:.2f} "
                          f"({per}) · standoff {r['standoff_m']:.2f} m · normal from {r['normal_source']}  \n"
                          f"a pose with every part > 80 % visible exists: **{r['exists_80']}** · best step {r['best_step']['name']} {r['best_step']['min_visible']:.2f}")

    @server.scene.on_click()
    def _(event):
        o = np.array(event.ray_origin, float)
        d = np.array(event.ray_direction, float)
        ans = rs.cast_rays(o3d.core.Tensor(np.concatenate([o, d])[None].astype(np.float32)))
        t = float(ans["t_hit"].numpy()[0])
        if not np.isfinite(t):
            hit_md.content = "_no surface under the click_"
            return
        p = o + t * d
        row = int(surf.rows_near(p[None], 2.0)[0])
        chain = stv.part_of(row) if row >= 0 else []
        server.scene.add_icosphere("/hit", radius=0.002, color=(255, 0, 255), position=p)
        if not chain:
            hit_md.content = f"({p[0] * 1000:.0f}, {p[1] * 1000:.0f}, {p[2] * 1000:.0f}) mm · row {row} · _no part here_"
            return
        hit_md.content = (f"({p[0] * 1000:.0f}, {p[1] * 1000:.0f}, {p[2] * 1000:.0f}) mm · row {row}  \n"
                          + "  \n".join(f"{'· ' * stv.depth(q)}P{q} {stv.parts[q].name}" for q in chain[::-1]))
        part_dd.value = next(n for n in names if n.startswith(f"P{chain[0]} "))

    print(f"e05 viewer on http://localhost:{args.port}  run {args.run}  {args.mm} mm  {args.mode}  ({src})", flush=True)
    while True:
        time.sleep(10)


if __name__ == "__main__":
    main()
