# e04: the voxel geometry V of a run: build, navigate, resolution

**Opened:** 2026-10-12. **Rung:** a new base under rungs 4–5. V is the geometry of the
IWM structure S / T / V (`project/iwm-structure-s-t-v`). This experiment builds V only:
no labels, no tree, no merging.

## Question

What does a voxel grid built from one run's views look like, how do we move around it
and query it, and what resolution keeps the parts we will later label (ports, labels,
stickers of ~5 mm and up)?

## Why

S / T / V is settled. The two main uses, geometry-aware view selection and semantic
search with best subpart view, both stand on V. They need a surface with normals at one
resolution. Before detections are written into V (e05), we need to know the container
itself: what it shows, what it loses, and which operations on it work.

## Input

Recorded runs, read-only:
- `2209-compute-ray-demo5`: 9 steps; the c5.x frames from e01–e03.
- `2209-box-ray-demo1`: flat faces and labels.

Per step: `depth_aligned.npy`, `rgb.png`, `T_base_cam`, `rgb_rotation_deg`, `mask.png`.
Per run: `session.json` (colour intrinsics, depth scale). Today's fused point cloud
(`fused/cloud.npy`, 3 mm point averaging) is the reference to compare against.

## What we do

1. **Build V.**
   - Integrate every step's aligned depth and colour into an Open3D 0.19 TSDF voxel
     grid (`VoxelBlockGrid`, signed distance + weight + colour per voxel).
   - Use the camera pose `T_base_cam`. On 180° steps, rotate the image back to raw.
   - Voxels live in the robot base frame.
   - Two scopes: the object only (step masks) and the scene around it (table included,
     cropped to the workspace).
2. **Sweep the resolution:** voxel size 1, 2, 3, 5 and 10 mm. Per size, record:
   - voxel count and memory;
   - build time;
   - the surface (mesh) extracted from V.
3. **Navigate V.** Try the operations e05 and view selection will need:
   - **ray cast** from a camera pose: depth, normal and colour images of V;
   - **point query:** which voxel, its distance to the surface, its normal;
   - **region crop:** the voxels inside a box;
   - **surface extraction:** mesh or points.
4. **Viewer.**
   - Viser, like e03: V's surface at a chosen resolution, with a resolution switch.
   - Click a point on the surface → its voxel index, its normal, and the steps that saw
     it.

## What we look at

- **V against the fused cloud,** at each resolution, rendered from the capture camera,
  the side and the top.
- **Rendered vs captured depth:** V ray-cast from each step's own pose, minus the
  depth that step captured.
- **Normals rendered as colour** on the faces: are flat faces flat, and do port recesses
  show?
- **Close-up of a port array and a label** at each resolution: down to which size do
  they stay visible as shapes?

## What would count as a no

- **At the size the details need (≤ 3 mm), the surface breaks up.** Holes, double walls
  from the per-view offset, or ports smeared flat.
- **Normals on flat faces are too noisy** to face a detail.
- **The ray-cast depth from a step's own pose differs from that step's captured depth**
  by more than a part is wide.

## Done when

Anton has moved around V in the viewer at several resolutions, and the report shows the
sweep. We then choose the resolution e05 builds on.

## Needs

No GPU, no robot. Open3D 0.19 and Viser 1.1.1, both in the robo env.
