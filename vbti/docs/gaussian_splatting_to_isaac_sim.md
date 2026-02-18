# Gaussian Splatting Pipeline: Video to Isaac Sim

Complete guide for reconstructing real-world scenes as 3D Gaussian Splats and importing them into NVIDIA Isaac Sim for robotics simulation.

**Tested with:** Fedora 42, RTX 4070 Ti SUPER (sm_89), CUDA 12.9, PyTorch 2.5.1+cu124, nerfstudio 1.1.5, conda env `gsplat-pt25`

## Prerequisites

### Conda Environment

```bash
conda activate gsplat-pt25
```

### Required Packages

```bash
# Nerfstudio (installed as editable from libs/nerfstudio)
pip install -e libs/nerfstudio

# Viewer (upgrade to latest for better UI)
pip install viser==1.0.21

# 3DGRUT for PLY → USDZ conversion
cd libs
git clone https://github.com/nv-tlabs/3dgrut.git
cd 3dgrut
git submodule update --init --recursive
pip install -e .
pip install hydra-core omegaconf plyfile usd-core addict slangtorch==1.3.4
pip install fire scikit-learn polyscope piexif kornia einops dataclasses_json pygltflib libigl
```

### GCC 14 (Fedora 42 ships gcc 15, but CUDA 12.9 requires <=14)

```bash
sudo dnf install -y gcc14-c++ gcc14
```

### COLMAP Vocab Tree Fix

COLMAP switched from flann to faiss index format in May 2025. Nerfstudio downloads the old flann version, which crashes on newer COLMAP. Fix:

```bash
rm ~/.local/share/nerfstudio/vocab_tree.fbow
wget -O ~/.local/share/nerfstudio/vocab_tree.fbow \
  "https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin"
```

---

## Step 1: Capture Video

Record a video of the scene with a smartphone or camera. Guidelines:
- Move slowly and smoothly around the scene
- Cover all angles you want reconstructed
- Avoid motion blur (good lighting helps)
- Overlap between frames is important — don't skip large areas

## Step 2: Process Data with COLMAP

### From video:

```bash
ns-process-data video \
  --data data/your_scene/video.mp4 \
  --output-dir data/output/your_scene
```

### From images:

```bash
ns-process-data images \
  --data data/your_scene/images/ \
  --output-dir data/output/your_scene
```

### Troubleshooting: Low Pose Recovery

If COLMAP reports low pose percentage (e.g., "only found poses for 0.63% of images"), check for **multiple reconstruction models**:

```python
import struct

def count_images(path):
    with open(path, 'rb') as f:
        return struct.unpack('<Q', f.read(8))[0]

# Check all models
import os
sparse_dir = "data/output/your_scene/colmap/sparse"
for model in sorted(os.listdir(sparse_dir)):
    images_bin = os.path.join(sparse_dir, model, "images.bin")
    if os.path.exists(images_bin):
        print(f"Model {model}: {count_images(images_bin)} images")
```

If a larger model exists (e.g., model `1` has 316 images while model `0` has 2), swap them:

```bash
cd data/output/your_scene/colmap/sparse
mv 0 0_old
mv 1 0
```

Then regenerate `transforms.json`:

```python
# Run with the nerfstudio python
from nerfstudio.process_data.colmap_utils import colmap_to_json
from pathlib import Path

colmap_to_json(
    recon_dir=Path("data/output/your_scene/colmap/sparse/0"),
    output_dir=Path("data/output/your_scene"),
)
```

## Step 3: Train Splatfacto

### High-Quality Parameters

```bash
ns-train splatfacto \
  --output-dir data/scenes/your_scene \
  --max-num-iterations 50000 \
  --pipeline.model.cull-alpha-thresh 0.005 \
  --pipeline.model.continue-cull-post-densification False \
  --pipeline.model.use-scale-regularization True \
  --pipeline.model.stop-split-at 25000 \
  --pipeline.model.rasterize-mode antialiased \
  --pipeline.model.num-downscales 0 \
  --pipeline.model.resolution-schedule 250 \
  nerfstudio-data --data data/output/your_scene
```

> **CLI argument order matters:** trainer/pipeline args come first, then `nerfstudio-data`, then `--data`.

### Parameter Reference

| Parameter | Default | High Quality | Purpose |
|-----------|---------|-------------|---------|
| `cull-alpha-thresh` | 0.1 | 0.005 | Keep more translucent gaussians |
| `continue-cull-post-densification` | True | False | Stop culling after densification |
| `use-scale-regularization` | False | True | Reduce spikey artifacts (PhysGaussian) |
| `stop-split-at` | 15000 | 25000 | Extend densification for 50k runs |
| `rasterize-mode` | classic | antialiased | Better small gaussian rendering |
| `num-downscales` | 2 | 0 | Train at full resolution immediately |
| `resolution-schedule` | 3000 | 250 | Faster resolution ramp-up |

### Viewer During Training

The viser viewer opens at `http://localhost:7007` during training.

- **Click a camera frustum** to jump into the scene (default view is from far away)
- **Lower Max Res** to 256-512 for more frequent live updates
- **Lower Train Util** slider to 0.5 for smoother viewer at cost of slower training

The viewer update frequency is: `render_freq = train_util * vis_time / ((1 - train_util) * train_time)`. Fast models like splatfacto benefit from lower max resolution.

### View Trained Model

```bash
ns-viewer --load-config data/scenes/your_scene/<run>/config.yml
```

## Step 4: Export Gaussian Splats

### Export PLY (with optional crop)

Get crop parameters from the viewer's Crop Viewport tool, then:

```bash
ns-export gaussian-splat \
  --load-config data/scenes/your_scene/<run>/config.yml \
  --output-dir data/scenes/your_scene/export/ \
  --obb_center <x> <y> <z> \
  --obb_rotation <rx> <ry> <rz> \
  --obb_scale <sx> <sy> <sz>
```

Without crop:

```bash
ns-export gaussian-splat \
  --load-config data/scenes/your_scene/<run>/config.yml \
  --output-dir data/scenes/your_scene/export/
```

## Step 5: Clean Splats (Optional)

### SuperSplat (Browser-based)

1. Open [superspl.at/editor](https://superspl.at/editor)
2. Drag in the exported `splat.ply`
3. Enable **Splat View Mode** to visualize gaussians as ellipses
4. Use **Brush/Lasso/Polygon** to select floaters
5. Delete selected, then export cleaned PLY

### Training-time Prevention

For next training, add tighter culling:

```bash
--pipeline.model.cull-screen-size 0.10
--pipeline.model.cull-scale-thresh 0.3
```

## Step 6: Convert PLY to USDZ for Isaac Sim

### Run Converter

```bash
CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 TORCH_CUDA_ARCH_LIST="8.9" \
  python -m threedgrut.export.scripts.ply_to_usd \
    data/scenes/your_scene/export/splat.ply \
    --output_file data/scenes/your_scene/export/splat.usdz
```

> **First run** JIT-compiles the CUDA tracer (~60s). Subsequent runs use the cached build.

> **gcc-14 is required** on Fedora 42 (gcc 15 is unsupported by nvcc). Set `TORCH_CUDA_ARCH_LIST` to your GPU's compute capability (8.9 for RTX 4070 Ti SUPER).

### Import in Isaac Sim

Requires **Isaac Sim 5.0+** (Omniverse Kit 107.3+):

1. `File > Import` and select the `.usdz` file
2. Or drag-and-drop from the content browser

The USDZ uses NVIDIA's custom `UsdVolVolume` schema — gaussians are rendered natively via the NuRec neural rendering pipeline, preserving full visual quality.

### Extract Collision Mesh (for physics)

Gaussian splats are visual-only. For robot collision geometry, extract a mesh from the cleaned PLY:

```python
import open3d as o3d
import numpy as np
from plyfile import PlyData

# Load cleaned gaussian splat PLY
ply = PlyData.read("data/scenes/your_scene/export/splat.ply")
v = ply["vertex"]

# Extract positions, filter by opacity (keep solid gaussians)
positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)
opacities = 1.0 / (1.0 + np.exp(-v["opacity"]))  # sigmoid
mask = opacities > 0.5
positions = positions[mask]

# Create point cloud and estimate normals
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(positions)
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
)
pcd.orient_normals_consistent_tangent_plane(k=15)

# Poisson surface reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# Remove low-density vertices (cleanup)
densities = np.asarray(densities)
mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.05))

o3d.io.write_triangle_mesh("data/scenes/your_scene/export/collision_mesh.ply", mesh)
```

> This uses the **cleaned** PLY (after SuperSplat editing), not the raw checkpoint. The opacity filter at 0.5 keeps only solid gaussians for the collision surface.

Import this mesh alongside the USDZ as a collision body in Isaac Sim.

---

## Quick Reference

```
Video → ns-process-data → COLMAP poses → ns-train splatfacto → ns-export → ply_to_usd → Isaac Sim
```

| Step | Command | Output |
|------|---------|--------|
| Process video | `ns-process-data video` | `transforms.json` + images |
| Train splats | `ns-train splatfacto` | checkpoint + config.yml |
| View result | `ns-viewer --load-config` | localhost:7007 |
| Export PLY | `ns-export gaussian-splat` | `splat.ply` |
| Clean (optional) | [superspl.at/editor](https://superspl.at/editor) | cleaned `splat.ply` |
| Convert to USDZ | `python -m threedgrut.export.scripts.ply_to_usd` | `splat.usdz` |
| Isaac Sim | `File > Import` | Scene in simulation |

---

## Troubleshooting

### COLMAP vocab tree crash: "Failed to read faiss index"

COLMAP switched from flann to faiss. Replace the cached vocab tree:

```bash
rm ~/.local/share/nerfstudio/vocab_tree.fbow
wget -O ~/.local/share/nerfstudio/vocab_tree.fbow \
  "https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin"
```

### COLMAP low pose recovery

Check `colmap/sparse/` for multiple models. Nerfstudio picks model `0` which may not be the best. Swap the larger model into `0`.

### ns-train argument order error

Trainer args → `nerfstudio-data` → data args. `--data` goes **after** `nerfstudio-data`.

### CUDA JIT: "unsupported GNU version"

Install gcc-14 and set `CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14`.

### CUDA JIT: arch list warning

Set `TORCH_CUDA_ARCH_LIST` to your GPU compute capability:
- RTX 4070/4080/4090: `"8.9"`
- RTX 3070/3080/3090: `"8.6"`
- A100: `"8.0"`
