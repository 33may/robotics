"""e07: V as it grows during the run — one TSDF, the steps integrated in capture order, the surface
extracted after each step (e04's settings: scene scope, same truncation, weight threshold, crop).

So capture k shows the geometry of steps 0..k only (step 0 = the survey that boots V).
Cached per (run, mm) in vbti/inspection/data/iwm/e07_update_loop/<run>/<mm>mm/V_after_step<id>.ply.
"""
import numpy as np
import open3d as o3d
import open3d.core as o3c

from inspection.cell.geometry import MAX_RANGE_M
from inspection.investigation.iwm.e01_all_masks import common as C
from inspection.investigation.iwm.e04_voxel_scene import build as B

OUT = C.ROOT / "inspection" / "data" / "iwm" / "e07_update_loop"


def grow(run, mm, steps, intr, scale, dev=None):
    """{step id: legacy TriangleMesh of V after that step}, built once and cached."""
    d = OUT / run / f"{mm}mm"
    paths = {st["id"]: d / f"V_after_step{st['id']}.ply" for st in steps}
    if all(p.exists() for p in paths.values()):
        return {i: o3d.io.read_triangle_mesh(str(p)) for i, p in paths.items()}
    d.mkdir(parents=True, exist_ok=True)
    dev = dev or B.device()
    size = mm / 1000.0
    K = B.intrinsic_tensor(intr)
    trunc_mult = max(B.TRUNC_VOXELS, B.MIN_TRUNC_M / size)
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)), voxel_size=size, block_resolution=B.BLOCK_RES,
        block_count=60000 if size >= 0.002 else 200000, device=dev)
    cloud = np.load(C.RUNS_DIR / run / "fused" / "cloud.npy")
    box = o3d.geometry.AxisAlignedBoundingBox(cloud.min(0) - [B.SCENE_MARGIN_M, B.SCENE_MARGIN_M, 0.03],
                                              cloud.max(0) + [B.SCENE_MARGIN_M, B.SCENE_MARGIN_M, 0.10])
    out = {}
    for st in steps:
        dd = B.clean_depth(st["depth"], scale, None)
        depth = o3d.t.geometry.Image(o3c.Tensor(dd.astype(np.uint16))).to(dev)
        color = o3d.t.geometry.Image(o3c.Tensor(st["rgb"].astype(np.uint8))).to(dev)
        ext = o3c.Tensor(np.linalg.inv(st["T"]), o3c.float64)
        coords = vbg.compute_unique_block_coordinates(depth, K, ext, 1.0 / scale, MAX_RANGE_M, trunc_voxel_multiplier=trunc_mult)
        vbg.integrate(coords, depth, color, K, K, ext, 1.0 / scale, MAX_RANGE_M, trunc_voxel_multiplier=trunc_mult)
        mesh = vbg.extract_triangle_mesh(weight_threshold=B.MESH_WEIGHT).to_legacy().crop(box)
        o3d.io.write_triangle_mesh(str(paths[st["id"]]), mesh)
        out[st["id"]] = mesh
    return out
