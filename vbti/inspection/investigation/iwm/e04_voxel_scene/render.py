"""e04: look at V by ray casting it (the same operation view selection will use).

Per run and scope:
  evidence/<run>_<scope>_sweep.jpg
     rows = voxel sizes; columns = colour from the reference capture camera, normals from
     that camera, colour from the side, normals from the top, colour + normals of a
     close-up (LAN ports on compute, the product label on the box)
  evidence/<run>_<scope>_selfcheck.jpg
     V ray-cast from each step's OWN pose vs the depth that step captured (object pixels):
     |rendered - captured| as a heat map
  evidence/render_stats.json  per size and step: median / p90 |diff| mm, share > 5 mm,
     share of object pixels V renders at all

Run:  p vbti/inspection/investigation/iwm/e04_voxel_scene/render.py [--runs ...] [--scopes object scene]
"""
import argparse
import json

import numpy as np
import open3d as o3d
import open3d.core as o3c

from inspection.investigation.iwm.e01_all_masks import common as C
from inspection.investigation.iwm.e03_tree_3d.quality import look_at, letterbox, turbo
from inspection.investigation.iwm.e04_voxel_scene.build import (HERE, MIN_TRUNC_M, OUT, RUNS, TRUNC_VOXELS,
                                                                clean_depth, device, intrinsic_tensor, load_steps)

SIZES = [10, 5, 3, 2, 1]
REF = {"2209-compute-ray-demo5": {"step": 6, "frame": "c5.6", "focus": [4, 5, 10], "focus_name": "LAN ports"},
       "2209-box-ray-demo1": {"step": 4, "frame": "b1.4", "focus": list(range(1, 12)), "focus_name": "product label"}}
W, H = 640, 360           # render size (half the capture), intrinsics scaled
PW, PH = 380, 260          # panel size
DARK = (24, 26, 32)


def load_vbg(run, scope, mm, dev):
    return o3d.t.geometry.VoxelBlockGrid.load(str(OUT / run / scope / f"{mm:g}mm" / "vbg.npz")).to(dev)


def cast(vbg, size_m, T_base_cam, K, w=W, h=H):
    trunc = max(TRUNC_VOXELS, MIN_TRUNC_M / size_m)
    ext = o3c.Tensor(np.linalg.inv(T_base_cam), o3c.float64)
    r = vbg.ray_cast(block_coords=vbg.hashmap().key_tensor(), intrinsic=K, extrinsic=ext, width=w, height=h,
                     render_attributes=["depth", "normal", "color"], depth_scale=1.0, depth_min=0.05,
                     depth_max=1.0, weight_threshold=1.0, trunc_voxel_multiplier=trunc, range_map_down_factor=8)
    depth = r["depth"].cpu().numpy()[..., 0]
    normal = r["normal"].cpu().numpy()
    color = r["color"].cpu().numpy()
    hit = depth > 0
    col = np.where(hit[..., None], np.clip(color, 0, 1) * 255, DARK).astype(np.uint8)
    nrm = np.where(hit[..., None], (np.clip(normal, -1, 1) + 1) * 127.5, DARK).astype(np.uint8)
    return depth, col, nrm


def K_scaled(intr, s=0.5):
    return o3c.Tensor([[intr["fx"] * s, 0, intr["ppx"] * s], [0, intr["fy"] * s, intr["ppy"] * s], [0, 0, 1]],
                      o3c.float64)


def K_virtual(f=520.0, w=W, h=H):
    return o3c.Tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], o3c.float64)


def focus_point(run):
    ref = REF[run]
    z = np.load(C.ROOT / "inspection" / "data" / "iwm" / "e03_tree_3d" / ref["frame"] / "full" / "nodes.npz")
    sel = np.isin(z["node"], ref["focus"])
    return z["points"][sel].mean(0), np.ptp(z["points"][sel], 0).max()


def cameras(run, steps):
    ref = next(s for s in steps if s["id"] == REF[run]["step"])
    T = ref["T"]
    center, size = focus_point(run)
    cam = T[:3, 3]
    obj_c = np.array(center)
    side_dir = cam - obj_c
    side_dir[2] = 0
    side_dir /= np.linalg.norm(side_dir)
    rot = np.array([[0.42, -0.906, 0], [0.906, 0.42, 0], [0, 0, 1]])   # 65 deg around z
    side_eye = obj_c + rot @ side_dir * 0.30 + np.array([0, 0, 0.12])
    top_eye = obj_c + np.array([0, 0, 0.35])
    close_dir = (cam - center) / np.linalg.norm(cam - center)
    close_eye = center + close_dir * max(0.08, size * 3.0)
    up = T[:3, 1] if _is_flipped(run, ref["id"]) else -T[:3, 1]   # image-up of the operator's upright frame
    return {"ref": ref, "side": look_at(side_eye, obj_c), "top": look_at(top_eye, obj_c, up=(1, 0, 0)),
            "close": look_at(close_eye, center, up=up if abs(up @ close_dir) < 0.9 else (0, 0, 1))}


def panel(img, title):
    return C.label(letterbox(img, PW, PH, DARK), title, size=13)


def sweep_figure(run, scope, steps, intr, dev, stats):
    cams = cameras(run, steps)
    ref = cams["ref"]
    Kc = K_scaled(intr)
    Kv = K_virtual()
    Kz = K_virtual(f=900.0)
    rows = []
    for mm in SIZES:
        vbg = load_vbg(run, scope, mm, dev)
        s = mm / 1000
        _, c_ref, n_ref = cast(vbg, s, ref["T"], Kc)
        _, c_side, _ = cast(vbg, s, cams["side"], Kv)
        _, _, n_top = cast(vbg, s, cams["top"], Kv)
        _, c_cl, n_cl = cast(vbg, s, cams["close"], Kz)
        flip = _is_flipped(run, ref["id"])
        c_ref = c_ref[::-1, ::-1] if flip else c_ref
        n_ref = n_ref[::-1, ::-1] if flip else n_ref
        st = next(x for x in stats if x["scope"] == scope and x["voxel_mm"] == mm)
        tag = f"{mm:g} mm · {st['voxels_observed'] / 1e6:.2f} M voxels · {st['mesh_triangles'] / 1e3:.0f} k tri"
        rowp = [panel(c_ref, f"{tag}  |  colour, capture step {ref['id']}"), panel(n_ref, "normals, same camera"),
                panel(c_side, "colour, side"), panel(n_top, "normals, top"),
                panel(c_cl, f"close-up: {REF[run]['focus_name']}"), panel(n_cl, "close-up normals")]
        rows.append(C.grid(rowp, cols=6, pad=4, bg=(255, 255, 255)))
        del vbg
    img = C.grid(rows, cols=1, pad=6, bg=(255, 255, 255))
    return C.title_band(img, f"{run} · scope {scope} · V ray-cast at five voxel sizes "
                             f"(normals: red = +x, green = +y, blue = +z of base, as RGB)")


def _is_flipped(run, step_id):
    s = json.load(open(C.RUNS_DIR / run / "steps" / f"{step_id:03d}" / "step.json"))
    return int(s.get("rgb_rotation_deg", 0) or 0) == 180


def selfcheck(run, scope, steps, intr, scale, dev):
    Kc = K_scaled(intr)
    res, rows = [], []
    for mm in SIZES:
        vbg = load_vbg(run, scope, mm, dev)
        panels = []
        for st in steps:
            if st["mask"] is None:
                continue
            d_rend, _, _ = cast(vbg, mm / 1000, st["T"], Kc)
            cap = clean_depth(st["depth"], scale, st["mask"])[::2, ::2].astype(np.float64) * scale
            obj = st["mask"][::2, ::2] & (cap > 0)
            both = obj & (d_rend > 0)
            diff = np.abs(d_rend - cap) * 1000
            dv = diff[both]
            res.append({"run": run, "scope": scope, "voxel_mm": mm, "step": st["id"],
                        "rendered_share": round(float(both.sum() / max(obj.sum(), 1)), 3),
                        "median_mm": round(float(np.median(dv)), 2) if dv.size else None,
                        "p90_mm": round(float(np.percentile(dv, 90)), 2) if dv.size else None,
                        "share_gt_5mm": round(float((dv > 5).mean()), 3) if dv.size else None})
            heat = np.full((*diff.shape, 3), 235, np.uint8)
            heat[both] = turbo(np.clip(diff[both] / 10.0, 0, 1))
            heat[obj & ~both] = (255, 0, 255)
            if _is_flipped(run, st["id"]):
                heat = heat[::-1, ::-1]
            r = res[-1]
            panels.append(C.label(letterbox(heat, 300, 190, (235, 235, 235)),
                                  f"{mm:g} mm · step {st['id']} · med {r['median_mm']} mm · "
                                  f"{100 * r['rendered_share']:.0f}% rendered", size=11))
        rows.append(C.grid(panels, cols=len(panels), pad=3, bg=(255, 255, 255)))
        del vbg
    img = C.grid(rows, cols=1, pad=5, bg=(255, 255, 255))
    return C.title_band(img, f"{run} · {scope}: V from each step's own pose vs that step's captured depth "
                             f"(turbo 0-10 mm, magenta = object pixel V does not render)"), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--scopes", nargs="*", default=["object", "scene"])
    args = ap.parse_args()
    from PIL import Image
    dev = device()
    bstats = json.load(open(HERE / "evidence" / "build_stats.json"))
    path = HERE / "evidence" / "render_stats.json"
    allres = json.load(open(path)) if path.exists() else []
    for run in args.runs:
        steps, intr, scale = load_steps(run)
        stats = [s for s in bstats if s["run"] == run]
        for scope in args.scopes:
            img = sweep_figure(run, scope, steps, intr, dev, stats)
            Image.fromarray(img).save(HERE / "evidence" / f"{run}_{scope}_sweep.jpg", quality=88)
            img, res = selfcheck(run, scope, steps, intr, scale, dev)
            Image.fromarray(img).save(HERE / "evidence" / f"{run}_{scope}_selfcheck.jpg", quality=88)
            allres = [r for r in allres if not (r["run"] == run and r["scope"] == scope)] + res
            for mm in SIZES:
                rr = [r for r in res if r["voxel_mm"] == mm and r["median_mm"] is not None]
                print(f"{run[5:]:20s} {scope:6s} {mm:>3g} mm  rendered {np.mean([r['rendered_share'] for r in rr]):.2f}"
                      f"  median {np.median([r['median_mm'] for r in rr]):.2f} mm  p90 "
                      f"{np.median([r['p90_mm'] for r in rr]):.2f}  >5mm {np.mean([r['share_gt_5mm'] for r in rr]):.3f}",
                      flush=True)
    json.dump(allres, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
