# Robust region normals on a noisy TSDF (Open3D VoxelBlockGrid, D405, 2-14 mm view offsets)

Date: 2026-10-12. Scope: getting a stable normal for a *region* (face / part) at a chosen point, to aim a camera at it.

## Why today's normals are noisy (from the code)
- Open3D `VoxelBlockGrid.ray_cast(..., render_attributes=['normal'])` interpolates trilinearly a TSDF gradient over the 8 voxel corners, so it is a 1-voxel finite difference. `extract_point_cloud` uses a ±1-voxel central difference (`DeviceGetNormal`). [VoxelBlockGridImpl.h](https://www.open3d.org/html/cpp_api/_voxel_block_grid_impl_8h_source.html)
- KinectFusion does the same: raycast normal = ν[∇F] (numerical SDF derivative), and the per-frame normal is a cross product of neighbouring vertex-map pixels. [Newcombe et al. 2011](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf). Voxblox/nvblox also use the normalized TSDF gradient by central differences ([voxblox](https://arxiv.org/pdf/1611.03631), [nvblox](https://arxiv.org/pdf/2311.00626)).
- So none of the standard systems give a "region" normal. Their scale is 1 voxel (1-5 mm), the same as our noise and the per-view offsets. A single marching-cubes triangle is even worse.

## Theory that sets the radius
Mitra & Nguyen, *Estimating Surface Normals in Noisy Point Cloud Data* (SoCG 2003, [pdf](https://www.cs.jhu.edu/~misha/Fall13b/Papers/Mitra03.pdf)): the error of a least-squares plane fit is bounded by a curvature term (∝ κr) plus a noise term that shrinks with r. **With no curvature (a flat face), make r as large as possible**: the bound drops as r^-3/2. With both noise and curvature, r_opt ≈ Θ(1)(σn²/κ)^(1/3) at high density. For us this means: on a flat face, use the largest radius that stays on the face. PCL's tutorial gives the same rule in practice: too large a radius rotates normals at edges ([PCL normal estimation](https://pointclouds.org/documentation/tutorials/normal_estimation.html)).

*My inference, not from a source:* if the per-view offset is mostly along the face normal, the fused data looks like parallel layers. A plane fit over r ≫ layer separation keeps the right orientation. An offset in the tangent direction does not affect a flat face at all. The harm comes from steps and bumps near edges, and from small radii.

## Ranked shortlist

### 1. Segment-aware plane fit: RANSAC, then least-squares refine (best for "face behind a detail")
- Gather surface points near the target (`vbg.extract_point_cloud(weight_threshold=3.0)`, or ray-cast vertex maps). If a 2D mask exists, keep only the points that project into the parent-face mask, with the detail itself removed.
- `plane, inl = pcd.segment_plane(distance_threshold, ransac_n=3, num_iterations=1000)` (tensor API defaults: 0.01, 3, 100, 0.999 ([t.PointCloud](https://www.open3d.org/docs/release/python_api/open3d.t.geometry.PointCloud.html))). Then run PCA on the inliers only.
- distance_threshold: about the noise/offset band, i.e. a few mm (my suggestion, roughly 3-5 mm for our 2-5 mm noise; tune it).
- Pros: ignores the detail, edges and the other layers' outliers, and returns the *face* normal, which is what you aim along. Cons: needs a segmentation or crop, and fails on curved parts.
- Detail normal vs face normal: for ports and stickers (5-15 mm, about the size of the noise radius), the detail's own points are too few to fit reliably. Use the face normal. Fit the detail's own points only when it is clearly non-coplanar and large.

### 2. Weighted PCA over a radius neighbourhood (default when no mask)
- Classic Hoppe/PCA: normal = smallest eigenvector of the neighbourhood covariance. Open3D: `pcd.estimate_normals(max_nn=..., radius=r)` (hybrid search is recommended in the docs). For a single query point it is simpler to do this in numpy: KD-tree radius query, Gaussian weights w=exp(-d²/(2(r/2)²)), then eigh of the weighted covariance.
- Radius: well above the noise and layer separation, well below the face size and the distance to edges. *My suggestion:* sweep r ∈ {8, 12, 16, 24} mm and keep the largest r that still passes the quality checks below.
- Pros: easy and fast, and it gives a quality score for free. Cons: mixes faces near edges and corners. Plain PCA is not robust to outliers, which is why method 1 comes first.

### 3. Smoothed TSDF gradient (stay volumetric)
- Copy a dense sub-block around the point (the tsdf and weight arrays), smooth it, then take the gradient. Use **weight-aware normalized convolution**: G*(w·tsdf) / G*(w) (Knutsson & Westin 1993), with `scipy.ndimage.gaussian_filter(sigma=k voxels)` followed by `np.gradient`, or `gaussian_filter(order=1)` per axis ([scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html)).
- σ in voxels × voxel size = physical scale, so choose it with the same logic as r above. Truncation is only about 4 voxels and gradients are only valid inside the band, so a large σ needs a wider truncation or a coarser grid. Open3D has no built-in smoothing or multi-level option for this (not found in the API).
- Pros: no extraction step, and it is consistent with the ray-cast hit. Cons: limited by the truncation band, and more code.

### 4. Mesh side: area-weighted normals over a patch, after Taubin smoothing
- Open3D `compute_vertex_normals()` sums *unnormalized* triangle normals, so it is already area-weighted, but only over the 1-ring ([TriangleMesh.cpp](https://raw.githubusercontent.com/isl-org/Open3D/main/cpp/open3d/geometry/TriangleMesh.cpp)). For a region, sum `cross(e1, e2)` over all triangles within radius r of the hit. Optionally call `mesh.filter_smooth_taubin(number_of_iterations=10, lambda_filter=0.5, mu=-0.53)` first, which smooths without shrinkage ([TriangleMesh](https://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html)).
- Edge-preserving option: *Bilateral Normal Filtering for Mesh Denoising* (Zheng et al., TVCG 2011, [pubmed](https://pubmed.ncbi.nlm.nih.gov/21173457/)). There is no Open3D implementation. It is equivalent to method 2 on a mesh, so it is only worth it if you already work on the mesh.

### (5) Robust/sharp-feature alternatives, for reference only
Hough-transform normals ([Boulch & Marlet 2012](https://imagine.enpc.fr/~marletr/publi/SGP-2012-Boulch-Marlet.pdf)) and MLS (PCL `MovingLeastSquares`: `setSearchRadius`, `setPolynomialOrder(2)`, `setComputeNormals(true)` ([PCL resampling](https://pointclouds.org/documentation/tutorials/resampling.html))). They target sharp edges or curved surfaces, and our faces are flat. Learned normals (PCPNet, etc.) are overkill here.

## Sign/orientation
PCA/RANSAC normals have no sign. Flip the normal so it points toward the cameras that observed the point (the mean camera centre of the views that see it, or the best-weighted view). Open3D: `orient_normals_towards_camera_location(cam)`; PCL: `flipNormalTowardsViewpoint` / `setViewPoint` ([PCL](https://pointclouds.org/documentation/tutorials/normal_estimation.html)). Because the object sits on a table, a "dot with +z_world > 0" check is a cheap sanity check for top faces only.

## Measuring quality
- **Surface variation** σ = λ0/(λ0+λ1+λ2) of the neighbourhood covariance (Pauly, Gross, Kobbelt 2002, *Efficient simplification of point-sampled surfaces*). Low means planar. PCL reports this as "curvature".
- **Radius stability**: the angle between normals at r, 1.5r and 2r. A large jump means the neighbourhood reached an edge or another face.
- **Neighbour spread**: the angular std of normals estimated at points within a few mm of the target.
- **Plane residual / inlier ratio** from RANSAC, plus the angle between the PCA normal and the RANSAC face normal.
- Pass thresholds are not given in the sources. Calibrate them on our own runs.

## Sources
Mitra & Nguyen 2003; Newcombe et al. 2011 (KinectFusion); Oleynikova et al. 2017 (voxblox); Millane et al. 2023 (nvblox); Open3D 0.19 Python API and C++ source; PCL tutorials; Zheng et al. 2011; Boulch & Marlet 2012; Pauly et al. 2002; Klasing et al. ICRA 2009, *Comparison of surface normal estimation methods for range sensing applications* ([ACM](https://dl.acm.org/doi/10.5555/1703435.1703753)), a practitioner comparison I did not read in full.
