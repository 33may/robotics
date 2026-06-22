# Gaussian Splat Dataset Formats — Comprehensive Reference

## Standard 3DGS PLY Format (59-Dimensional Vector)

The original graphdeco-inria implementation defines the canonical format. Each Gaussian is stored as a 59-dim float32 vector:

| Attribute | Dims | Properties | Storage Notes |
|-----------|------|------------|---------------|
| Position | 3 | x, y, z | World coordinates |
| Normals | 3 | nx, ny, nz | **Always zeros** (placeholder) |
| SH DC | 3 | f_dc_0, f_dc_1, f_dc_2 | 0th-order SH (base RGB) |
| SH Rest | 45 | f_rest_0..f_rest_44 | Orders 1-3, 3 channels each |
| Opacity | 1 | opacity | **Logit** space: alpha = sigmoid(opacity) |
| Scale | 3 | scale_0, scale_1, scale_2 | **Log** space: sigma = exp(scale) |
| Rotation | 4 | rot_0, rot_1, rot_2, rot_3 | Quaternion (not required normalized) |

Total: 3+3+3+45+1+3+4 = 62 properties in PLY, but 59 meaningful dims (normals are zeros).

### SH Coefficient Breakdown (3 bands = degree 3)
- Degree 0: 1 coeff x 3 channels = 3 (f_dc)
- Degree 1: 3 coeffs x 3 channels = 9
- Degree 2: 5 coeffs x 3 channels = 15
- Degree 3: 7 coeffs x 3 channels = 21
- Total SH rest: 9+15+21 = 45

### Key Code (graphdeco-inria/gaussian-splatting/scene/gaussian_model.py)
```python
def construct_list_of_attributes(self):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(self._scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(self._rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l

def save_ply(self, path):
    xyz = self._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)  # ALWAYS ZEROS
    f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = self._opacity.detach().cpu().numpy()
    scale = self._scaling.detach().cpu().numpy()
    rotation = self._rotation.detach().cpu().numpy()
    dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def load_ply(self, path):
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    # f_rest count = 3*(max_sh_degree + 1)^2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1)**2 - 1))
    # ... scales, rotations loaded similarly
```

---

## 1. SceneSplat-49K / GaussianWorld

### Overview
- **Size**: ~49K raw scenes, ~46K curated, 29.24B Gaussians total
- **Avg Gaussians/scene**: 1.26M
- **Storage**: ~8.36 TB total
- **Sources**: SceneSplat-7K (ScanNet, ScanNet++, Replica, Hypersim, 3RScan, ARKitScenes, Matterport3D), DL3DV-10K, HoliCity, Aria Synthetic Environments, crowdsourced
- **Quality**: Mean PSNR 27.83 dB, SSIM 0.898, LPIPS 0.209, depth L1 0.061m
- **License**: CC-BY-SA-4.0 (processing scripts), individual dataset licenses (non-commercial)

### File Format: Preprocessed .npy files (NOT raw PLY)
Each scene directory contains:
```
scene_id/
  color.npy      # (N, 3) RGB values — NOT SH, just RGB
  coord.npy      # (N, 3) xyz positions
  opacity.npy    # (N, 1) opacity values
  quat.npy       # (N, 4) quaternion rotation
  scale.npy      # (N, 3) scale parameters
  lang_feat.npy  # (N, 768) SigLIP2 language features (SceneSplat++ only)
  valid_feat_mask.npy  # validity mask for language features
```

**IMPORTANT**: Color is stored as RGB, NOT as SH coefficients. The preprocessed format strips SH to just DC component (converted to RGB). This means higher-order SH view-dependent effects are LOST in this format.

### Chunking for Training
- Grid voxelization: 1.0cm resolution
- Chunk dimensions: 6x6 spatial units (60cm x 60cm)
- Stride: 3x3 (30cm overlap)
- Max chunks per scene: 6

### Loading
```python
import numpy as np
color = np.load("scene_id/color.npy")    # (N, 3)
coord = np.load("scene_id/coord.npy")    # (N, 3)
opacity = np.load("scene_id/opacity.npy") # (N, 1)
quat = np.load("scene_id/quat.npy")      # (N, 4)
scale = np.load("scene_id/scale.npy")    # (N, 3)
```

### gaussian_world_49k vs scene_splat_49k
- `gaussian_world_49k` = raw PLY files (standard 3DGS format)
- `scene_splat_49k` = preprocessed .npy format (split by attribute, RGB only)

### SceneSplat++ Additions
- 12,061 scenes with per-Gaussian vision-language embeddings (768-dim SigLIP2)
- Features aligned in image-embedding space, not text-embedding space
- Dynamic weighting: global context + local features + masked features
- Processing cost: 361 GPU-days for embeddings alone

### Curation Criteria
- Depth supervision during optimization for geometry quality
- Blurry frame filtering using variance of Laplacian as sharpness metric
- Per-scene quality metrics (PSNR, SSIM, LPIPS, depth L1) tracked in CSV

---

## 2. ShapeSplat / ShapeSplatsV1

### Overview
- **Objects**: 65K total (87 categories); ShapeSplatsV1 = 52K (55 categories from ShapeNetCore)
- **Avg Gaussians/object**: ~24K (ShapeNet-Core: 24,267; ModelNet: 22,456)
- **Quality**: PSNR 44-45 dB
- **Compute**: 3.8 GPU-years (TITAN XP)
- **License**: Requires access request (research only)

### File Format: Standard PLY (59-dim)
```
shapesplat_ply/
  {category_id}-{object_id}.ply
```

Each PLY contains the standard 3DGS attributes: xyz, normals (zeros), f_dc (3), f_rest (45), opacity, scale (3), rotation (4).

### Training Parameters Used to Generate
- SH degree: 3 (full 48 SH coefficients per Gaussian)
- Init: 5K uniformly sampled points
- Pruning: 60% at iterations 16K and 24K
- Views: 72 uniformly spaced per object
- Resolution: 400x400

### Gaussian-MAE Loading Code
```python
from plyfile import PlyData
import numpy as np

gs_vertex = PlyData.read('path.ply')['vertex']
x = gs_vertex['x'].astype(np.float32)
y = gs_vertex['y'].astype(np.float32)
z = gs_vertex['z'].astype(np.float32)
centroids = np.stack((x, y, z), axis=-1)  # (N, 3)

opacity = gs_vertex['opacity'].astype(np.float32).reshape(-1, 1)  # (N, 1)

# Scale
scale_names = [p.name for p in gs_vertex.properties if p.name.startswith("sca")]
scales = np.zeros((len(x), len(scale_names)))
for idx, name in enumerate(sorted(scale_names)):
    scales[:, idx] = gs_vertex[name].astype(np.float32)

# Rotation (quaternion)
rot_names = [p.name for p in gs_vertex.properties if p.name.startswith("rot")]
rots = np.zeros((len(x), len(rot_names)))
for idx, name in enumerate(sorted(rot_names)):
    rots[:, idx] = gs_vertex[name].astype(np.float32)
rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)  # normalize

# SH base (DC only for grouping)
sh_base = np.stack([gs_vertex['f_dc_0'], gs_vertex['f_dc_1'], gs_vertex['f_dc_2']], axis=-1)
```

### Gaussian-MAE Normalization
- All attributes except quaternions: recentered to zero mean, mapped to unit sphere
- SH: only base DC (3D) used for grouping, not full 48D
- Downsampled to 1,024 Gaussians per object for MAE pretraining
- Grouping: KNN on centroids, then pool within groups

### Dataset Statistics
| Dataset | Objects | Avg GS/obj | PSNR |
|---------|---------|-----------|------|
| ShapeNet-Core | 52,121 | 24,267 | 44.19 |
| ShapeNet-Part | 16,823 | 23,689 | — |
| ModelNet | 12,308 | 22,456 | 45.10 |

---

## 3. InteriorGS (SpatialVerse)

### Overview
- **Scenes**: 1,000 indoor (752 residential, 248 public)
- **Objects**: 554K instances, 755 categories
- **Source**: Artist-created mesh scenes, rendered ~3K views/scene, reconstructed with gsplat
- **Coordinate system**: XYZ = (Right, Back, Up), meters
- **License**: Custom InteriorGS terms of use

### File Format: SuperSplat Compressed PLY
```
scene_id/
  3dgs_compressed.ply   # ~35.5 MB per scene
  labels.json           # ~264 KB (semantic annotations + 3D bounding boxes)
  occupancy.json        # ~290 bytes (metadata for occupancy map)
  occupancy.png         # ~3.4 KB (1024x1024, 255=free/0=occupied/127=unknown)
  structure.json        # ~46 KB (rooms, walls, holes, instances)
```

### SuperSplat Compressed PLY Format
NOT standard 3DGS PLY. Uses chunked quantization:
- Splats grouped into 256-splat chunks
- Per-chunk: 18 floats for min/max quantization bounds
- Per-vertex: 4 packed uint32s
  - Position: 11+10+11 bits (XYZ)
  - Rotation: 2+10+10+10 bits (quaternion IJKL)
  - Scale: 11+10+11 bits (XYZ)
  - Color+Opacity: 8+8+8+8 bits (RGBA)
- SH coefficients: quantized to uint8 (optional)
- Morton ordering for spatial coherence
- ~60-70% size reduction vs uncompressed

**CRITICAL**: Compressed PLY needs decompression before use as training data. Standard plyfile won't work — need SuperSplat's decompression or conversion tool.

### Semantic Annotations (labels.json)
- Object-level annotations (NOT per-Gaussian labels)
- Each object: category label, instance ID, 3D oriented bounding box (8 corner vertices)
- Bounding boxes are axis-aligned, refined via surface point sampling

### Object Decomposition Strategy
To get per-Gaussian labels from bounding box annotations:
1. Decompress PLY to get Gaussian positions
2. For each Gaussian, test containment in object bounding boxes
3. Assign label of enclosing bounding box
4. Handle overlaps/ambiguity with priority rules

### Estimated Total Size
- ~35.5 MB/scene x 1000 scenes = ~35 GB for PLY files
- Plus annotations: ~310 KB/scene x 1000 = ~310 MB
- **Total: ~36 GB**

---

## 4. uCO3D (Meta/Facebook Research)

### Overview
- **Objects**: 170,000 across 1,000+ categories (LVIS taxonomy, 50 super-categories)
- **Total size**: 19.27 TB (full dataset); gaussian_splats modality: 1.18 TB
- **Source**: Real-world turntable-like videos with 360-degree coverage
- **License**: Research use

### File Format: gsplat Compressed (PngCompression)
```
super_category/category/sequence_name/gaussian_splats/
  means.png       # compressed positions
  scales.png      # compressed scales
  quats.png       # compressed quaternions
  opacities.png   # compressed opacities
  sh0.png         # compressed DC SH
  shN.png         # compressed higher-order SH
  meta.npz        # quantization metadata
```

### gsplat PngCompression Format
Required keys in splats dictionary:
- `means`: (N, 3) — positions
- `scales`: (N, 3) — scale factors
- `quats`: (N, 4) — quaternions
- `opacities`: (N,) — opacity values
- `sh0`: (N, 1, 3) — base SH
- `shN`: (N, 24, 3) — higher-order SH (24 coefficients)

Compression uses: quantization + spatial sorting (SOGS) + PNG encoding + K-means for SH.
Dependencies: `imageio`, `plas`, `torchpq`

### Loading Code
```python
# Install: pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
from uco3d import UCO3DFrameDataBuilder, render_splats

# Build dataset
frame_data = dataset[100]
gaussian_splats = frame_data.sequence_gaussian_splats

# Render
render_colors, render_alphas, render_info = render_splats(
    cameras=frame_data.camera,
    splats=gaussian_splats,
    render_size=[512, 512]
)

# Background truncation option:
# gaussian_splats_truncate_background=True in UCO3DFrameDataBuilder
```

### Download
```bash
python dataset_download/download_dataset.py \
  --download_folder <DEST> \
  --download_modalities "gaussian_splats" \
  --use_huggingface
```

---

## 5. DL3DV-GS-960P

### Overview
- **Scenes**: 6,939
- **Resolution**: 960P undistorted images
- **Source**: DL3DV dataset, post-processed by FCGS
- **Total size**: >1 TB
- **License**: Requires access approval

### File Format: Standard 3DGS PLY (via FCGS)
Stores pre-trained 3DGS as standard PLY files (or compressed bitstreams).

### FCGS Compression
FCGS = Fast Feedforward 3D Gaussian Splatting Compression
- Input: standard .ply → compressed bitstream → .ply roundtrip
- Feed-forward (no optimization), compresses in seconds
- Lambda parameter controls quality/size tradeoff

### Commands
```bash
# Encode PLY → bitstream
python encode_single_scene.py --lmd 4e-4 --ply_path_from point_cloud.ply --bit_path_to BITS/ --determ 1

# Decode bitstream → PLY
python decode_single_scene.py --lmd 4e-4 --bit_path_from BITS/ --ply_path_to restored.ply
```

### Download
```bash
python download_DL3DV-GS-960P.py --odir ./DL3DV-GS-960P --subset 1K --file_type 3DGS --split train
```

---

## Cross-Dataset Compatibility Analysis

### Attribute Comparison

| Dataset | Format | SH Bands | Dims/Gaussian | Avg GS Count | Activation Space |
|---------|--------|----------|---------------|-------------|------------------|
| SceneSplat-49K | .npy (RGB) | 0 (DC only→RGB) | 14 | 1.26M | Pre-activated RGB |
| ShapeSplat | .ply (standard) | 3 | 59 | ~24K | Raw (logit/log) |
| InteriorGS | compressed .ply | 0-3 (uint8) | varies | unknown | Quantized |
| uCO3D | gsplat PNG | 3 (24 SH) | ~62 | unknown | Pre-activation |
| DL3DV-GS | .ply (standard) | 3 | 59 | unknown | Raw (logit/log) |

### Key Compatibility Issues

1. **SH representation mismatch**: SceneSplat stores RGB only (no view-dependent color), while ShapeSplat/DL3DV store full SH. Need to decide: train with full SH or strip to DC/RGB.

2. **Activation space**: Standard PLY stores opacity as logit, scale as log-scale. SceneSplat's npy format may store pre-activated values. uCO3D gsplat explicitly stores pre-activation values. Must verify and standardize.

3. **Scale differences**: Scene-level datasets (SceneSplat, InteriorGS, DL3DV) have millions of Gaussians; object-level (ShapeSplat, uCO3D) have thousands. Need subsampling or different handling.

4. **Coordinate conventions**: InteriorGS uses XYZ=(Right,Back,Up). Others may differ. Need alignment.

5. **Compression formats**: InteriorGS (SuperSplat) and uCO3D (gsplat PNG) need decompression before training. DL3DV-GS can optionally use FCGS bitstreams.

### Recommended Standardization for Training

For a GS completion model, standardize to:
```python
# Per-Gaussian feature vector (14 dims, matching Gaussian-MAE):
features = {
    'xyz': (N, 3),      # world coordinates, zero-centered per scene
    'opacity': (N, 1),   # sigmoid-activated [0, 1]
    'scale': (N, 3),     # exp-activated (actual scale, not log)
    'rotation': (N, 4),  # unit quaternion (normalized)
    'color': (N, 3),     # RGB from DC SH, or SH2RGB conversion
}
# Total: 14 dims per Gaussian
```

Normalization (following Gaussian-MAE):
- All except quaternions: zero-mean, unit-sphere per attribute dim
- Quaternions: normalize to unit length

### Disk Space Summary

| Dataset | Approx Size | Access |
|---------|------------|--------|
| SceneSplat-49K (.npy) | ~8.36 TB | Gated, non-commercial |
| ShapeSplat (PLY) | ~100-200 GB (est.) | Gated, research |
| InteriorGS | ~36 GB | Gated, custom license |
| uCO3D (GS only) | 1.18 TB | Research |
| DL3DV-GS-960P | >1 TB | Gated |
| **Combined** | **~11 TB** | |
