"""Preview Antonio's Isaac USD assembly (UR5e + RG2) in meshcat, and measure
the gripper in the flange frame (-> TCP candidate + collision envelope box).

Usage (from repo root, robo env active):
    p inspection/cell/usd_preview.py            # serve + keep running for review
    p inspection/cell/usd_preview.py --print    # just print measurements, no viewer

Needs: pip install usd-core (already in robo env).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from pxr import Usd, UsdGeom

DEFAULT_USD = (
    Path(__file__).resolve().parents[1]
    / "data/isaac_models/ur5e_new_gripper_rg2/ur5e_new_rg2.usd"
)
FLANGE_PRIM = "/ur5e/wrist_3_link/flange"
GRIPPER_PRIMS = ["/ur5e/RG2_gripper_edit", "/ur5e/Gripper"]


def triangulate(counts, indices):
    """Fan-triangulate polygon faces -> (M, 3) int array."""
    tris, i = [], 0
    for c in counts:
        for k in range(1, c - 1):
            tris.append((indices[i], indices[i + k], indices[i + k + 1]))
        i += c
    return np.array(tris, dtype=np.int64)


def _local_T(pos, rot):
    """Joint local frame -> 4x4 (column convention)."""
    from pxr import Gf

    m = Gf.Matrix4d().SetRotate(Gf.Quatd(rot))
    T = np.array(m).T
    T[:3, 3] = [pos[0], pos[1], pos[2]]
    return T


def _axis_rot(axis, angle_deg):
    """Rotation about a joint's local axis (X/Y/Z), 4x4."""
    a = np.deg2rad(angle_deg or 0.0)
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    i = "XYZ".index(axis or "Z")
    j, k = (i + 1) % 3, (i + 2) % 3
    T[j, j], T[j, k], T[k, j], T[k, k] = c, -s, s, c
    return T


def solve_articulation(stage):
    """The gripper's INTERNAL part arrangement as saved is coherent (CAD assembly);
    what's stale in raw USD is only the assembly-level FixedJoint weld to the arm
    (Isaac applies joints at sim start). NOTE: do NOT re-solve the finger linkage
    from joint states — `state:angular:physics:position` is an unset 0.0 default,
    not the assembled angle; per-body solving explodes the four-bar.
    Returns {subtree_root: 4x4 correction} from cross-assembly welds only."""
    from pxr import UsdPhysics

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    def world_T(path):
        return np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T

    corrections = {}
    it = Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))
    for prim in it:
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        j = UsdPhysics.FixedJoint(prim)
        b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
        if not b0 or not b1:
            continue
        root0 = "/" + "/".join(str(b0[0]).strip("/").split("/")[:2])
        root1 = "/" + "/".join(str(b1[0]).strip("/").split("/")[:2])
        if root0 == root1:
            continue  # internal weld, arrangement already coherent
        L0 = _local_T(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get())
        L1 = _local_T(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get())
        T_target = world_T(str(b0[0])) @ L0 @ np.linalg.inv(L1)
        corrections[root1] = T_target @ np.linalg.inv(world_T(str(b1[0])))
    return corrections


def collect_meshes(stage, skip_collisions=True, slide_mm=0.0):
    """Yield (path, world_vertices Nx3, faces Mx3) for every Mesh prim."""
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    corrections = solve_articulation(stage)
    if slide_mm and corrections:
        # manual alignment tweak: slide the welded tool along the flange axis
        # (+ = outward along flange X, ur_description convention)
        Tf = np.array(
            UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
                stage.GetPrimAtPath(FLANGE_PRIM))).T
        d = Tf[:3, 0] / np.linalg.norm(Tf[:3, 0])
        T_slide = np.eye(4)
        T_slide[:3, 3] = d * (slide_mm / 1000.0)
        corrections = {k: T_slide @ C for k, C in corrections.items()}
    # longest body path first so meshes match their most specific rigid body
    body_paths = sorted(corrections, key=len, reverse=True)
    # Isaac marks geometry instanceable; plain Traverse() skips instance internals
    it = Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))
    for prim in it:
        path = str(prim.GetPath())
        if path.startswith("/Render"):
            continue
        if skip_collisions and "/collisions" in path:
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = np.array(mesh.GetPointsAttr().Get() or [], dtype=float)
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if pts.size == 0 or not counts:
            continue
        T = np.array(cache.GetLocalToWorldTransform(prim)).T  # row-major -> column
        for bp in body_paths:  # apply the joint solve the raw stage doesn't
            if path.startswith(bp):
                T = corrections[bp] @ T
                break
        world = (T[:3, :3] @ pts.T).T + T[:3, 3]
        yield path, world * scale, triangulate(counts, indices)


def flange_frame_bounds(stage, slide_mm=0.0):
    """AABB of the gripper subtree expressed in the flange frame."""
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    flange = stage.GetPrimAtPath(FLANGE_PRIM)
    if not flange:
        return None
    T_wf = np.array(cache.GetLocalToWorldTransform(flange)).T
    T_fw = np.linalg.inv(T_wf)
    results = {}
    for root in GRIPPER_PRIMS:
        prim = stage.GetPrimAtPath(root)
        if not prim:
            continue
        pts_all = []
        for path, world, _ in collect_meshes(stage, skip_collisions=True, slide_mm=slide_mm):
            if path.startswith(root):
                homo = np.hstack([world / scale, np.ones((len(world), 1))])
                pts_all.append((T_fw @ homo.T).T[:, :3] * scale)
        if pts_all:
            pts = np.vstack(pts_all)
            results[root] = (pts.min(axis=0), pts.max(axis=0))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", type=Path, default=DEFAULT_USD)
    ap.add_argument("--print", action="store_true", help="measurements only")
    ap.add_argument("--collisions", action="store_true", help="show collision meshes too")
    ap.add_argument("--slide-mm", type=float, default=-22.0,
                    help="nudge the welded tool along the flange axis (+ = outward)")
    args = ap.parse_args()

    stage = Usd.Stage.Open(str(args.usd))
    print(f"stage: {args.usd}  (metersPerUnit={UsdGeom.GetStageMetersPerUnit(stage)})")

    bounds = flange_frame_bounds(stage, slide_mm=args.slide_mm)
    if bounds:
        print("\n== gripper AABB in FLANGE frame (meters) ==")
        for root, (lo, hi) in bounds.items():
            size = hi - lo
            print(f"{root}")
            print(f"   min  [{lo[0]:+.4f} {lo[1]:+.4f} {lo[2]:+.4f}]")
            print(f"   max  [{hi[0]:+.4f} {hi[1]:+.4f} {hi[2]:+.4f}]")
            print(f"   size [{size[0]: .4f} {size[1]: .4f} {size[2]: .4f}]")
            fwd = int(np.argmax(hi))  # flange frame: +X out of the tool face (ur_description)
            print(f"   -> fingertip reach along flange {'XYZ'[fwd]}: {hi[fwd]:.4f} m "
                  f"(open fingers; TCP from closed state or datasheet)")
    else:
        print("WARNING: flange prim not found; no measurements")

    if args.print:
        return 0

    import meshcat
    import meshcat.geometry as g

    vis = meshcat.Visualizer()
    print(f"\nmeshcat URL: {vis.url()}")
    n = 0
    for path, world, faces in collect_meshes(
            stage, skip_collisions=not args.collisions, slide_mm=args.slide_mm):
        name = path.strip("/").replace("/", "|")
        color = 0xCC4422 if "ripper" in path or "RG2" in path else 0x8A8A8A
        vis["usd"][name].set_object(
            g.TriangularMeshGeometry(world, faces), g.MeshLambertMaterial(color=color)
        )
        n += 1
    print(f"rendered {n} meshes — arm gray, gripper red. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
