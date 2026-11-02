"""e04: build the voxel geometry V of a run (Open3D TSDF VoxelBlockGrid).

Every captured step's aligned depth + rgb is integrated with its recorded T_base_cam
(as-is, the per-view offset is NOT corrected). Voxels live in the robot base frame.

Scopes:
  object : depth outside the step's object mask (mask.png) is zeroed
  scene  : full depth up to MAX_RANGE_M; the result is cropped to the workspace box

Per (run, scope, voxel size) -> vbti/inspection/data/iwm/e04_voxel_scene/<run>/<scope>/<mm>mm/
  vbg.npz      the grid (Open3D save format)
  mesh.ply     marching-cubes surface, vertex colours
  stats.json   blocks, active voxels, memory, build time, mesh size

Run:  p vbti/inspection/investigation/iwm/e04_voxel_scene/build.py [--runs ...] [--sizes 1 2 3 5 10]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.core as o3c
from PIL import Image

from inspection.cell.geometry import MAX_RANGE_M, MIN_RANGE_M
from inspection.investigation.iwm.e01_all_masks import common as C

HERE = Path(__file__).resolve().parent
OUT = C.ROOT / "inspection" / "data" / "iwm" / "e04_voxel_scene"
RUNS = ["2209-compute-ray-demo5", "2209-box-ray-demo1"]
TRUNC_VOXELS = 4.0          # truncation = 4 voxels (Open3D default trunc_voxel_multiplier is 8)
MIN_TRUNC_M = 0.004         # never truncate tighter than the depth noise at 1-2 mm voxels
BLOCK_RES = 16
SCENE_MARGIN_M = 0.12      # scene crop = object extent + this, in x and y
MESH_WEIGHT = 0.5          # weight counts views and the test is ">": 1.0 drops every voxel seen by ONE view


def device():
    return o3c.Device("CUDA:0") if o3c.cuda.is_available() else o3c.Device("CPU:0")


def load_steps(run):
    rd = C.RUNS_DIR / run
    sess = json.load(open(rd / "session.json"))
    intr = sess["intrinsics"]["color"]
    scale = float(sess["depth_scale_m_per_unit"])
    steps = []
    for sd in sorted((rd / "steps").iterdir()):
        s = json.load(open(sd / "step.json"))
        if s.get("outcome") != "captured":
            continue
        depth = np.load(sd / "depth_aligned.npy")
        rgb = np.array(Image.open(sd / "rgb.png").convert("RGB"))
        mask = np.array(Image.open(sd / "mask.png")) > 0 if (sd / "mask.png").exists() else None
        if int(s.get("rgb_rotation_deg", 0) or 0) == 180:   # disk is upright, pose + intrinsics are raw
            depth, rgb = depth[::-1, ::-1], rgb[::-1, ::-1]
            mask = mask[::-1, ::-1] if mask is not None else None
        steps.append({"id": s["step_id"], "T": np.array(s["T_base_cam"]), "depth": np.ascontiguousarray(depth),
                      "rgb": np.ascontiguousarray(rgb), "mask": None if mask is None else np.ascontiguousarray(mask),
                      "address": s["view"].get("address")})
    return steps, intr, scale


def intrinsic_tensor(intr):
    return o3c.Tensor([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]], o3c.float64)


def clean_depth(depth, scale, mask=None):
    z = depth.astype(np.float64) * scale
    d = depth.copy()
    d[(z < MIN_RANGE_M) | (z > MAX_RANGE_M) | (depth == 65535)] = 0
    if mask is not None:
        d[~mask] = 0
    return d


def build(run, scope, size_m, steps, intr, scale, dev):
    K = intrinsic_tensor(intr)
    trunc_mult = max(TRUNC_VOXELS, MIN_TRUNC_M / size_m)
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)), voxel_size=size_m, block_resolution=BLOCK_RES,
        block_count=60000 if size_m >= 0.002 else 200000, device=dev)
    t0 = time.time()
    used = []
    for st in steps:
        if scope == "object" and st["mask"] is None:
            continue
        d = clean_depth(st["depth"], scale, st["mask"] if scope == "object" else None)
        depth = o3d.t.geometry.Image(o3c.Tensor(d.astype(np.uint16))).to(dev)
        color = o3d.t.geometry.Image(o3c.Tensor(st["rgb"].astype(np.uint8))).to(dev)  # uint16 depth needs uint8 colour
        ext = o3c.Tensor(np.linalg.inv(st["T"]), o3c.float64)
        coords = vbg.compute_unique_block_coordinates(depth, K, ext, 1.0 / scale, MAX_RANGE_M,
                                                      trunc_voxel_multiplier=trunc_mult)
        vbg.integrate(coords, depth, color, K, K, ext, 1.0 / scale, MAX_RANGE_M, trunc_voxel_multiplier=trunc_mult)
        used.append(st["id"])
    if dev.get_type() == o3c.Device.DeviceType.CUDA:
        o3c.cuda.synchronize()
    t_build = time.time() - t0
    t1 = time.time()
    mesh = vbg.extract_triangle_mesh(weight_threshold=MESH_WEIGHT).to_legacy()
    t_mesh = time.time() - t1
    if scope == "scene":
        # NOT the cell WORKSPACE box: it cuts the compute unit at x = 0.05 (the object reaches x = -0.036).
        # Crop to the object's own extent (the run's fused cloud) plus a margin.
        cloud = np.load(C.RUNS_DIR / run / "fused" / "cloud.npy")
        lo = cloud.min(0) - [SCENE_MARGIN_M, SCENE_MARGIN_M, 0.03]
        hi = cloud.max(0) + [SCENE_MARGIN_M, SCENE_MARGIN_M, 0.10]
        mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(lo, hi))
    w = vbg.attribute("weight").cpu().numpy()
    n_blocks = int(vbg.hashmap().size())
    active = int((w[:n_blocks] > 0).sum()) if n_blocks else 0
    stats = {"run": run, "scope": scope, "voxel_mm": round(size_m * 1000, 3), "steps_used": used,
             "trunc_voxels": round(trunc_mult, 2), "trunc_mm": round(trunc_mult * size_m * 1000, 2),
             "blocks": n_blocks, "voxels_allocated": n_blocks * BLOCK_RES ** 3, "voxels_observed": active,
             "memory_mb": round(n_blocks * BLOCK_RES ** 3 * 5 * 4 / 1e6, 1),
             "build_s": round(t_build, 2), "mesh_s": round(t_mesh, 2),
             "mesh_vertices": len(mesh.vertices), "mesh_triangles": len(mesh.triangles),
             "device": str(dev)}
    return vbg, mesh, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=RUNS)
    ap.add_argument("--scopes", nargs="*", default=["object", "scene"])
    ap.add_argument("--sizes", nargs="*", type=float, default=[10, 5, 3, 2, 1])
    args = ap.parse_args()
    dev = device()
    allstats = []
    for run in args.runs:
        steps, intr, scale = load_steps(run)
        for scope in args.scopes:
            for mm in args.sizes:
                out = OUT / run / scope / f"{mm:g}mm"
                out.mkdir(parents=True, exist_ok=True)
                vbg, mesh, stats = build(run, scope, mm / 1000, steps, intr, scale, dev)
                vbg.save(str(out / "vbg.npz"))
                o3d.io.write_triangle_mesh(str(out / "mesh.ply"), mesh)
                json.dump(stats, open(out / "stats.json", "w"), indent=1)
                allstats.append(stats)
                print(f"{run[5:]:20s} {scope:6s} {mm:>4g} mm  blocks {stats['blocks']:6d}  observed "
                      f"{stats['voxels_observed']:>9d}  {stats['memory_mb']:>7.1f} MB  build {stats['build_s']:5.2f}s"
                      f"  mesh {stats['mesh_triangles']:>8d} tri", flush=True)
                del vbg
    (HERE / "evidence").mkdir(exist_ok=True)
    path = HERE / "evidence" / "build_stats.json"
    prev = json.load(open(path)) if path.exists() else []
    keep = [s for s in prev if (s["run"], s["scope"], s["voxel_mm"]) not in
            {(a["run"], a["scope"], a["voxel_mm"]) for a in allstats}]
    json.dump(keep + allstats, open(path, "w"), indent=1)
    C.run_record(OUT / "build", [{"run": r} for r in args.runs], {"name": "Open3D VoxelBlockGrid TSDF"},
                 {"sizes_mm": args.sizes, "scopes": args.scopes, "trunc_voxels": TRUNC_VOXELS,
                  "min_trunc_m": MIN_TRUNC_M, "block_res": BLOCK_RES, "max_range_m": MAX_RANGE_M})


if __name__ == "__main__":
    main()
