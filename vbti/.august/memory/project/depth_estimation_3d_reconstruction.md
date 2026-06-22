# Depth Estimation + 3D Reconstruction for Gaussian Splatting (2025-2026)

## The Landscape: COLMAP Alternatives

Traditional GS pipeline: images -> COLMAP (SfM) -> sparse point cloud + camera poses -> 3DGS training.
New paradigm: feed-forward neural networks replace COLMAP entirely, giving camera poses + depth + point clouds in seconds.

## Key Tools (Ranked by Current SOTA)

### 1. Depth Anything 3 (DA3) - ByteDance, Nov 2025
- **Best overall**: Beats VGGT by 35.7% in pose accuracy, 23.6% in geometric accuracy
- Single transformer: monocular depth, multi-view depth, camera poses, 3D Gaussians
- Outputs: depth maps, ray maps, camera extrinsics/intrinsics (OpenCV/COLMAP format)
- DA3Metric-Large: metric depth with real-world scale
- DA3Nested-Giant-Large: combines any-view + metric for best results
- Repo: https://github.com/ByteDance-Seed/Depth-Anything-3
- HF: depth-anything/DA3-BASE, DA3-LARGE, DA3Metric-Large, DA3Nested-Giant-Large

### 2. VGGT - Meta/Oxford, CVPR 2025 Best Paper
- Feed-forward: camera intrinsics/extrinsics, depth, point maps, tracks in <1s
- Has `demo_colmap.py` for direct COLMAP format export -> gsplat training
- 1-200+ images, scales to 1000+ with VGGT-X
- VRAM: 5.6GB (20 views) to 40.6GB (200 views)
- Commercial license available (VGGT-1B-Commercial)
- Repo: https://github.com/facebookresearch/vggt

### 3. MapAnything - Meta/CMU, 3DV 2026
- Universal metric 3D reconstruction, 12+ tasks in one model
- Handles metric scale via `is_metric_scale` flag + `metric_scaling_factor`
- COLMAP export supported
- Repo: https://github.com/facebookresearch/map-anything

### 4. MASt3R / MASt3R-SfM - Naver Labs
- Drop-in COLMAP replacement, exports COLMAP-style poses
- Dense feature matching, good for sparse views
- Limited to ~30 images on RTX 4090 (VRAM bottleneck)
- Repo: https://github.com/naver/mast3r

## Depth Anything Evolution
- V1 (2023): Monocular relative depth. Teacher-student with unlabeled data.
- V2 (2024): Better metric depth, ViT architecture, synthetic data training.
- V3 (Nov 2025): Unified model - single/stereo/multi-view depth + camera poses + 3D Gaussians. SOTA.

## Metric Scale (the "accurate scene scale" your mentor mentioned)
- Monocular depth = relative (no real-world units). Useless alone for reconstruction.
- Metric depth = absolute (meters). Required for accurate 3D geometry.
- DA3 Metric Series, MapAnything, and COLMAP (via triangulation) all provide metric scale.
- DA3Nested model combines relative quality with metric scale.

## Practical Pipeline for MILo/3DGS (our use case)

### Option A: DA3 (recommended, best accuracy)
```bash
pip install xformers torch>=2 torchvision
pip install -e .  # from DA3 repo
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git
```
```python
from depth_anything_3.api import DepthAnything3
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
prediction = model.inference(image_paths)
# Returns: depth [N,H,W], extrinsics [N,3,4], intrinsics [N,3,3]
```

### Option B: VGGT (fastest, easiest COLMAP export)
```bash
python demo_colmap.py --scene_dir=/path/to/scene/ --use_ba
# Outputs COLMAP format directly -> feed to gsplat or original 3DGS
```

## For Our Pipeline
- MILo already uses COLMAP. DA3/VGGT can replace COLMAP step.
- For real-world robot scenes: capture images -> DA3/VGGT -> COLMAP format -> 3DGS/MILo training
- Faster than COLMAP (seconds vs hours), better on textureless/complex scenes.
