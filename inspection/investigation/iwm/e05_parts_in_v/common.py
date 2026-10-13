"""e05 common: the views (e02 trees + poses), V from e04, and a full-resolution ray cast.

Orientation rule (e03/e04): rgb / depth / masks on disk are upright; T_base_cam and the
intrinsics are raw; on rgb_rotation_deg == 180 steps everything is flipped back to raw
before any geometry.

Smoke:  p inspection/investigation/iwm/e05_parts_in_v/common.py c5.6
"""
import json
import sys
from pathlib import Path

import numpy as np
import open3d.core as o3c

from inspection.investigation.iwm.e01_all_masks import common as C
from inspection.investigation.iwm.e04_voxel_scene.build import (BLOCK_RES, MIN_TRUNC_M, TRUNC_VOXELS, device,
                                                                 load_steps)
from inspection.investigation.iwm.e04_voxel_scene.render import load_vbg

HERE = Path(__file__).resolve().parent
EVID = HERE / "evidence"
DATA = C.ROOT / "inspection" / "data" / "iwm" / "e05_parts_in_v"
E02 = C.ROOT / "inspection" / "data" / "iwm" / "e02_agent_tree"
W, H = 1280, 720
DARK = (24, 26, 32)
#: the views of each run in capture order (the first one's root becomes the object)
ORDER = {"2209-compute-ray-demo5": ["c5.2", "c5.6", "c5.7"], "2209-box-ray-demo1": ["b1.1", "b1.4"]}
FRAMES = {f["id"]: (f["run"], f["step"]) for f in C.frames()}


def trunc_m(size_m):
    return max(TRUNC_VOXELS * size_m, MIN_TRUNC_M)


def V(run, mm, dev=None):
    """(vbg on the device, voxel size in m) of the e04 scene-scope grid."""
    return load_vbg(run, "scene", mm, dev or device()), mm / 1000.0


def K_of(intr, s=1.0):
    return np.array([[intr["fx"] * s, 0, intr["ppx"] * s], [0, intr["fy"] * s, intr["ppy"] * s], [0, 0, 1.0]])


def load_tree(fid, variant="full", tag="final"):
    """(tree json, {node id: bool mask HxW}) as e02 left them: upright."""
    d = E02 / fid / variant
    tree = json.load(open(d / f"tree_{tag}.json"))
    masks = C.load_packed(d / f"masks_{tag}.npz")
    ids = json.load(open(d / f"masks_{tag}_ids.json"))
    return tree, {int(i): m for i, m in zip(ids, masks)}


def load_view(fid):
    run, step = FRAMES[fid]
    f = {"id": fid, "run": run, "step": step}
    p = C.frame_paths(f)
    st = json.load(open(p["step"]))
    rot = int(st.get("rgb_rotation_deg", 0) or 0)
    T = np.array(st["T_base_cam"], dtype=float)
    sess = json.load(open(C.RUNS_DIR / run / "session.json"))
    intr = sess["intrinsics"]["color"]
    scale = float(sess["depth_scale_m_per_unit"])
    depth = np.load(p["depth"])
    rgb = C.load_rgb(f)
    tree, masks = load_tree(fid)
    if rot == 180:
        depth, rgb = depth[::-1, ::-1], rgb[::-1, ::-1]
        masks = {i: np.ascontiguousarray(m[::-1, ::-1]) for i, m in masks.items()}
    nodes = {n["id"]: n for n in tree["nodes"]}
    return {"id": fid, "run": run, "step": step, "T": T, "K": K_of(intr), "intr": intr, "scale": scale,
            "depth": np.ascontiguousarray(depth), "rgb": np.ascontiguousarray(rgb), "rot": rot,
            "tree": tree, "nodes": nodes, "masks": masks, "cam": T[:3, 3].copy()}


def cast(vbg, size_m, T, K, w=W, h=H, color=False):
    """Ray cast V from camera pose T (base <- cam) with intrinsics K: (depth m HxW, normal HxWx3[, colour uint8])."""
    ext = o3c.Tensor(np.linalg.inv(T), o3c.float64)
    Kt = o3c.Tensor(np.asarray(K, dtype=np.float64), o3c.float64)
    attrs = ["depth", "normal"] + (["color"] if color else [])
    r = vbg.ray_cast(block_coords=vbg.hashmap().key_tensor(), intrinsic=Kt, extrinsic=ext, width=int(w),
                     height=int(h), render_attributes=attrs, depth_scale=1.0, depth_min=0.03,
                     depth_max=1.0, weight_threshold=0.5, trunc_voxel_multiplier=max(TRUNC_VOXELS, MIN_TRUNC_M / size_m),
                     range_map_down_factor=8)
    depth = r["depth"].cpu().numpy()[..., 0]
    normal = r["normal"].cpu().numpy()
    if not color:
        return depth, normal
    col = np.where((depth > 0)[..., None], np.clip(r["color"].cpu().numpy(), 0, 1) * 255, DARK).astype(np.uint8)
    return depth, normal, col


def hits_to_points(depth, K, T):
    """Pixels with depth -> (points (M,3) in the base frame, px (M,2) as (v, u))."""
    v, u = np.nonzero(depth > 0)
    z = depth[v, u].astype(np.float64)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pc = np.column_stack([x, y, z])
    return pc @ T[:3, :3].T + T[:3, 3], np.column_stack([v, u])


def project(pts, K, T):
    """Base-frame points -> (u, v, z) in the camera of pose T."""
    pc = (pts - T[:3, 3]) @ T[:3, :3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = pc[:, 0] / z * K[0, 0] + K[0, 2]
        v = pc[:, 1] / z * K[1, 1] + K[1, 2]
    return u, v, z


def run_steps(run):
    """e04's step list (id, T, depth, rgb, mask, address), raw orientation."""
    steps, intr, scale = load_steps(run)
    return steps, intr, scale


if __name__ == "__main__":
    fid = sys.argv[1] if len(sys.argv) > 1 else "c5.6"
    v = load_view(fid)
    vbg, size = V(v["run"], 2)
    d, n = cast(vbg, size, v["T"], v["K"])
    print(f"{fid}: run {v['run']} step {v['step']} rot {v['rot']} nodes {len(v['nodes'])} masks {len(v['masks'])}")
    print(f"V 2 mm: blocks {vbg.hashmap().size()}  cast hit share {(d > 0).mean():.3f}  "
          f"depth median {np.median(d[d > 0]):.3f} m")
