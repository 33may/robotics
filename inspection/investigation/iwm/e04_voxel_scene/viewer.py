"""e04 viewer: walk around V of one run, switch voxel size, click the surface.

The surface is the marching-cubes mesh of V (vertex colours). A click casts the browser's
ray into that mesh (Open3D RaycastingScene) and shows:
  - the hit point, its voxel index (floor(p / size)) and the surface normal (a line)
  - every step that saw that point: projected into the step, its captured depth within
    SAW_TOL_M of the point, plus the angle between the step's ray and the normal
  - the crop of the best such step (most face-on) with the point marked
Step cameras can be drawn as frustums.

Run:  p inspection/investigation/iwm/e04_voxel_scene/viewer.py [--run 2209-compute-ray-demo5]
      [--scope scene] [--port 8082]
"""
import argparse
import time

import numpy as np
import open3d as o3d
import trimesh
import viser
import viser.transforms as vtf

from inspection.investigation.iwm.e04_voxel_scene.build import OUT, device, load_steps
from inspection.investigation.iwm.e04_voxel_scene.normals.normals import Surface
from inspection.investigation.iwm.e04_voxel_scene.render import SIZES, load_vbg

SAW_TOL_M = 0.008
METHODS = ["triangle (N0a)", "TSDF gradient (N0b)", "weighted PCA (N1)", "RANSAC plane (N2)", "smoothed TSDF (N3)"]
CROP = 90                       # half size of the evidence crop, px


def load_mesh(run, scope, mm):
    m = o3d.io.read_triangle_mesh(str(OUT / run / scope / f"{mm:g}mm" / "mesh.ply"))
    m.compute_triangle_normals()
    return m


def saw(p, n, steps, intr, scale):
    out = []
    for st in steps:
        Ti = np.linalg.inv(st["T"])
        pc = Ti[:3, :3] @ p + Ti[:3, 3]
        if pc[2] <= 1e-3:
            continue
        u = int(round(intr["ppx"] + intr["fx"] * pc[0] / pc[2]))
        v = int(round(intr["ppy"] + intr["fy"] * pc[1] / pc[2]))
        h, w = st["depth"].shape
        if not (0 <= u < w and 0 <= v < h):
            continue
        z = [pc[2]]
        d = st["depth"][v, u] * scale
        if d <= 0 or abs(d - z[0]) > SAW_TOL_M:
            continue
        ray = p - st["T"][:3, 3]
        ang = float(np.degrees(np.arccos(np.clip(-(ray / np.linalg.norm(ray)) @ n, -1, 1))))
        out.append({"step": st["id"], "u": u, "v": v, "dist_cm": float(np.linalg.norm(ray) * 100),
                    "angle": ang, "err_mm": float((d - z[0]) * 1000)})
    return sorted(out, key=lambda r: r["angle"])


def crop_of(st, u, v, flipped):
    img = st["rgb"].copy()                       # raw orientation (as integrated)
    h, w, _ = img.shape
    y0, y1, x0, x1 = max(0, v - CROP), min(h, v + CROP), max(0, u - CROP), min(w, u + CROP)
    c = img[y0:y1, x0:x1].copy()
    cv, cu = v - y0, u - x0
    c[max(0, cv - 1):cv + 2, max(0, cu - 12):cu + 13] = (255, 0, 255)
    c[max(0, cv - 12):cv + 13, max(0, cu - 1):cu + 2] = (255, 0, 255)
    return c[::-1, ::-1] if flipped else c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="2209-compute-ray-demo5")
    ap.add_argument("--scope", default="scene", choices=["scene", "object"])
    ap.add_argument("--port", type=int, default=8082)
    args = ap.parse_args()
    import json
    from inspection.investigation.iwm.e01_all_masks import common as C
    steps, intr, scale = load_steps(args.run)
    flipped = {st["id"]: int(json.load(open(C.RUNS_DIR / args.run / "steps" / f"{st['id']:03d}" / "step.json"))
                             .get("rgb_rotation_deg", 0) or 0) == 180 for st in steps}

    meshes, scenes = {}, {}
    for mm in SIZES:
        m = load_mesh(args.run, args.scope, mm)
        meshes[mm] = m
        rs = o3d.t.geometry.RaycastingScene()
        rs.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
        scenes[mm] = rs

    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(dark_mode=True)
    server.scene.set_up_direction("+z")
    handles = {}
    for mm, m in meshes.items():
        # marching-cubes winding is inconsistent (~30 % of a face's triangles point away from the camera
        # that saw it) and the browser culls back faces -> see-through holes. Draw both windings.
        F = np.asarray(m.triangles)
        tm = trimesh.Trimesh(np.asarray(m.vertices), np.concatenate([F, F[:, ::-1]]),
                             vertex_colors=(np.asarray(m.vertex_colors) * 255).astype(np.uint8), process=False)
        handles[mm] = server.scene.add_mesh_trimesh(f"/V/{mm:g}mm", tm, visible=False,
                                                    cast_shadow=False, receive_shadow=False)
    cur = {"mm": 2.0}
    handles[cur["mm"]].visible = True

    frust = []
    for st in steps:
        R = st["T"][:3, :3]
        wxyz = vtf.SO3.from_matrix(R).wxyz
        fov = 2 * np.arctan(intr["height"] / 2 / intr["fy"])
        img = st["rgb"][::8, ::8]
        frust.append(server.scene.add_camera_frustum(f"/cams/{st['id']}", fov=fov, aspect=intr["width"] / intr["height"],
                                                     scale=0.04, wxyz=wxyz, position=st["T"][:3, 3], image=img,
                                                     color=(255, 200, 0), visible=False))
        server.scene.add_label(f"/cams/{st['id']}/label", f"step {st['id']}", visible=False)

    with server.gui.add_folder("V"):
        size_dd = server.gui.add_dropdown("voxel size (mm)", [f"{mm:g}" for mm in SIZES], initial_value="2")
        cams_cb = server.gui.add_checkbox("step cameras", initial_value=False)
        look_dd = server.gui.add_dropdown("look from step", ["-"] + [str(st["id"]) for st in steps], initial_value="-")
        info = server.gui.add_markdown(f"**{args.run}** · scope {args.scope}")
    with server.gui.add_folder("Normal"):
        meth_dd = server.gui.add_dropdown("method", list(METHODS), initial_value="RANSAC plane (N2)")
        rad_sl = server.gui.add_slider("radius (mm)", 4, 30, 1, 16)
        sig_sl = server.gui.add_slider("TSDF blur sigma (voxels)", 1, 4, 1, 2)
        all_md = server.gui.add_markdown("")
    with server.gui.add_folder("Clicked point"):
        hit_md = server.gui.add_markdown("_click the surface_")
        crop_img = server.gui.add_image(np.full((8, 8, 3), 30, np.uint8), label="best step")

    def stats_md(mm):
        m = meshes[mm]
        return f"**{args.run}** · scope {args.scope} · {mm:g} mm · {len(m.triangles) / 1e3:.0f} k triangles"

    info.content = stats_md(cur["mm"])

    @size_dd.on_update
    def _(_):
        mm = float(size_dd.value)
        for k, h in handles.items():
            h.visible = (k == mm)
        cur["mm"] = mm
        info.content = stats_md(mm)

    @look_dd.on_update
    def _(event):
        if look_dd.value == "-":
            return
        st = next(x for x in steps if str(x["id"]) == look_dd.value)
        T = st["T"]
        R = T[:3, :3] @ (np.diag([-1, -1, 1]) if flipped[st["id"]] else np.eye(3))   # upright like the frame
        fov = 2 * np.arctan(intr["height"] / 2 / intr["fy"])
        clients = [event.client] if event.client is not None else list(server.get_clients().values())
        for client in clients:                                   # only the browser that asked
            with client.atomic():
                client.camera.fov = fov
                client.camera.position = T[:3, 3]
                client.camera.wxyz = vtf.SO3.from_matrix(R).wxyz
                client.camera.look_at = T[:3, 3] + R[:, 2] * 0.2
                client.camera.up_direction = -R[:, 1]

    @cams_cb.on_update
    def _(_):
        for f in frust:
            f.visible = cams_cb.value
        for st in steps:
            server.scene.add_label(f"/cams/{st['id']}/label", f"step {st['id']}", visible=cams_cb.value)

    surfaces = {}
    last = {}

    def surface(mm):
        if mm not in surfaces:
            surfaces[mm] = Surface(load_vbg(args.run, args.scope, mm, device()), mm / 1000)
        return surfaces[mm]

    def normal_of(method, p, tri_n, cam_toward, mm):
        r, sg = rad_sl.value / 1000, int(sig_sl.value)
        sf = surface(mm)
        n = {"triangle (N0a)": lambda: tri_n,
             "TSDF gradient (N0b)": lambda: sf.N0[sf.kd.query(p)[1]],
             "weighted PCA (N1)": lambda: sf.n1(p, r)[0],
             "RANSAC plane (N2)": lambda: sf.n2(p, r)[0],
             "smoothed TSDF (N3)": lambda: sf.n3(p, sg)}[method]()
        if n is None:
            return None
        n = np.asarray(n, float)
        return n if n @ cam_toward > 0 else -n

    def show():
        if not last:
            return
        mm, p, tri_n, toward = last["mm"], last["p"], last["tri"], last["toward"]
        n = normal_of(meth_dd.value, p, tri_n, toward, mm)
        if n is None:
            hit_md.content = "_too few surface points for this method here_"
            return
        server.scene.add_icosphere("/hit/p", radius=0.003, color=(255, 0, 255), position=p)
        server.scene.add_line_segments("/hit/n", np.array([[p, p + n * 0.05]]), np.array([[[255, 0, 255]] * 2]),
                                       line_width=4.0)
        uses_r = meth_dd.value in ("weighted PCA (N1)", "RANSAC plane (N2)")
        server.scene.add_icosphere("/hit/region", radius=rad_sl.value / 1000 if uses_r else 1e-6,
                                   color=(255, 0, 255), opacity=0.15, position=p)
        vox = np.floor(p / (mm / 1000)).astype(int)
        seen = saw(p, n, steps, intr, scale)
        lines = [f"**point** ({p[0] * 1000:.1f}, {p[1] * 1000:.1f}, {p[2] * 1000:.1f}) mm, base frame",
                 f"**voxel** {tuple(int(x) for x in vox)} at {mm:g} mm",
                 f"**normal** ({n[0]:+.2f}, {n[1]:+.2f}, {n[2]:+.2f}), {meth_dd.value}",
                 f"**seen by {len(seen)} step(s)**, most face-on first:"]
        for r in seen:
            lines.append(f"- step {r['step']}: {r['angle']:.0f}° to the normal, {r['dist_cm']:.0f} cm, "
                         f"depth off by {r['err_mm']:+.1f} mm")
        hit_md.content = "\n".join(lines)
        rows = []
        for m in METHODS:
            nm = normal_of(m, p, tri_n, toward, mm)
            rows.append(f"- {m}: " + ("n/a" if nm is None else
                        f"{np.degrees(np.arccos(np.clip(nm @ n, -1, 1))):.1f}° from the shown one"))
        all_md.content = f"**all methods** (radius {rad_sl.value:g} mm, sigma {sig_sl.value:g}):\n" + "\n".join(rows)
        if seen:
            st = next(s for s in steps if s["id"] == seen[0]["step"])
            crop_img.image = crop_of(st, seen[0]["u"], seen[0]["v"], flipped[st["id"]])

    for h in (meth_dd, rad_sl, sig_sl):
        h.on_update(lambda _: show())

    @server.scene.on_click()
    def _(event):
        mm = cur["mm"]
        o = np.array(event.ray_origin, float)
        d = np.array(event.ray_direction, float)
        ans = scenes[mm].cast_rays(o3d.core.Tensor(np.concatenate([o, d])[None].astype(np.float32)))
        t = float(ans["t_hit"].numpy()[0])
        if not np.isfinite(t):
            hit_md.content = "_no surface under the click_"
            return
        p = o + t * d
        last.update(mm=mm, p=p, tri=ans["primitive_normals"].numpy()[0].astype(float), toward=-d)
        show()
        print("click", mm, p.round(4), flush=True)

    print(f"e04 viewer on http://localhost:{args.port}  run {args.run}  scope {args.scope}", flush=True)
    while True:
        time.sleep(10)


if __name__ == "__main__":
    main()
