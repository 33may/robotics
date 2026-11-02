# e04 / normals: a reliable surface normal at a point of V

**Opened:** 2026-10-12. A sub-experiment of e04 (the voxel geometry V).

## Question

How do we get a surface normal at a chosen point of V that is reliable enough to face a
detail with the camera ("pivot on a detail")? The noisy reconstruction should not swing
it from one voxel to the next.

## Why

Geometry-aware view selection places the camera along a detail's normal. Today the
normal is very local:
- in the viewer, the normal of the one marching-cubes triangle the click ray hits;
- in the renders, the TSDF gradient at one voxel.

Both describe a patch only 1–2 voxels across. V has bumps and double layers where
views disagree (2–14 mm per-view offset), so these normals are noisy (Anton, using
the e04 viewer).

## Input

- V of `2209-compute-ray-demo5` and `2209-box-ray-demo1` from e04 (scene scope, 1–5 mm).
- Test points: e03's node centroids (ports, labels, stickers on known faces) and
  clicked points.
- Reference: the plane of the face each point lies on, fitted over the whole face.

## What we do

From the research pass
(`docs/research/iwm/passes/normals/robust-normals-on-noisy-tsdf.md`).

**Why the current normals are noisy.** Every standard system takes the normal over about
one voxel: Open3D's `ray_cast` and `extract_point_cloud`, KinectFusion, Voxblox and
nvblox all use the TSDF gradient. That is the same scale as our noise. On a flat face,
the error of a plane fit keeps falling as the radius grows (Mitra & Nguyen 2003). So the
fix is to fit over a radius much larger than the noise, while staying on the face.

Compared, from the same surface points of V:
- **N0, baseline:** the triangle normal and the TSDF gradient (today's).
- **N1, weighted PCA in a radius:** Gaussian weights, r = 4, 8, 12, 16, 24 mm. This is
  the default when there is no mask.
- **N2, RANSAC plane, then PCA on the inliers:** in the same radii, threshold 3–5 mm.
  Robust near edges.
- **N3, smoothed TSDF gradient:** weighted Gaussian blur of the TSDF, then the
  gradient. Limited by the ~4-voxel truncation band.
- **N4, part-aware** (needs the masks; here e03's projected nodes stand in for e05's
  detections): RANSAC plane on the parent face's points with the detail removed, vs a
  fit on the detail's own points.

Every normal is flipped towards the cameras that saw the point.

## What we look at

- The normal field drawn on a face, per method and radius: arrows on the ray-cast
  image.
- Per test point:
  - the angle to its face's fitted plane normal;
  - the spread of normals within a few mm of the point;
  - how much the normal changes as the radius grows (r, 1.5r, 2r);
  - patch flatness λ0/(λ0+λ1+λ2) (Pauly et al. 2002);
  - the RANSAC inlier ratio.

  The sources give no pass thresholds, so they are calibrated on our runs.
- In the viewer: the normal at a clicked point, per method, with the radius as a
  slider.

## What would count as a no

- On a flat face, no method and radius keeps the normal within ~10° of the face normal
  at detail-sized radii (5–15 mm).
- Or: the radius needed to calm the normal is larger than the detail, so the normal
  of a small part is really its face's normal.

## Done when

Anton has seen the comparison and the viewer click with the chosen method, and the
method for "normal of a point / of a part" is chosen.
