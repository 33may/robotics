# 3D Gaussian Splatting Segmentation Methods

## Summary Table

| Method | Retraining? | Post-hoc? | PLY Export? | Interaction | Speed | GPU | Code Available |
|--------|------------|-----------|-------------|-------------|-------|-----|----------------|
| SAGA | Feature training only (frozen GS) | Yes | Binary mask (scriptable) | Click/text/auto | 4ms segment | 24GB+ | Yes |
| Gaussian Grouping | Train from scratch | No | Decoupled groups (scriptable) | Auto (SAM) | ~170 FPS render | A100 | Yes |
| GaussianCut | No retraining | Yes | Not built-in | Click/scribble/text | Seconds | Unknown | Yes |
| Click-Gaussian | Feature training (frozen GS) | Partial | Not documented | Single click | 10ms | Unknown | No code |
| OpenGaussian | 4-stage pipeline | No | PLY with features | Click/text | Minutes | 24GB+ | Yes |
| LangSplat | Train from scratch | No | Not built-in | Text query | 199x faster than LERF | 24GB | Yes |
| LBG (Lifting by Gaussians) | No retraining | Yes | 3D asset extraction | Auto | ~450s total | RTX 3090 | Promised |
| SAGOnline | No retraining | Yes | Labels per Gaussian | Click prompt | 27ms/frame | RTX 4090 | No code yet |
| ArtisanGS | No retraining | Yes | Not documented | Click/paint/bbox/SAM | 1-5s | Unknown | Not yet |
| Feature Splatting (nerfstudio) | Train from scratch | No | Not documented | Text query | Real-time render | Unknown | Yes (nerfstudio) |
| Segment-then-Splat | Train from scratch (different) | No | Per-object by design | Text query | Unknown | Unknown | Project page only |
| DIY (gsply + clustering) | No | Yes | Yes (trivial) | Manual/scripted | Seconds | CPU/GPU | gsply on PyPI |

---

## Detailed Method Breakdown

### 1. SAGA (Segment Any 3D Gaussians) [AAAI 2025]
- **Repo**: https://github.com/Jumpat/SegAnyGAussians
- **Paper**: https://arxiv.org/abs/2312.00860
- **How it works**: Attaches a learnable affinity feature vector to each Gaussian (frozen geometry). Scale-gated mechanism: gate function S: [0,1] -> [0,1]^D maps scale to gating vector. Gated feature = S(s) * f_g (Hadamard product). Features rendered via alpha-blending, then scale gate applied to 2D rendered features.
- **Training**: Post-hoc on frozen Gaussians. 10K iterations on single GPU. Uses scale-aware contrastive loss distilling SAM segmentation capability.
- **Scale mechanism**: 2D mask projects to 3D via depth, scale = 2*sqrt(std(X)^2 + std(Y)^2 + std(Z)^2). Larger scales close more gate channels (36% positive, 64% negative).
- **Interaction**: GUI (saga_gui.py) with right-click point prompts, Jupyter notebook for text queries, automatic SAM-based "segment everything"
- **Visualization**: Real-time 3D viewpoint control, PCA decomposition of features, similarity heatmaps, 3D clustering views
- **Export**: Saves segmentation as PyTorch tensor (.pt) with binary masks. No built-in .ply export, but trivial to script: apply binary mask to Gaussian attributes and save subset.
- **Pipeline**: 1) Train 3DGS normally, 2) Extract SAM masks, 3) Generate scales, 4) Train contrastive features
- **GPU**: Needs downsampling for limited VRAM. HDBSCAN clustering on CPU (slow).
- **Compatibility**: Tied to specific 3DGS architecture. Would need adaptation for MILo.

### 2. Gaussian Grouping [ECCV 2024]
- **Repo**: https://github.com/lkeab/gaussian-grouping
- **Paper**: https://arxiv.org/abs/2312.00732
- **How it works**: Adds 16-dim Identity Encoding to each Gaussian. Uses zero-degree SH (view-independent). Rendered to 16*H*W feature maps. Supervised by cross-entropy loss against SAM mask labels + 3D KL-divergence regularization (k=5 nearest neighbors, m=1000 sampled points).
- **Training**: Requires training from scratch (joint RGB + identity). Minimal PSNR degradation (28.43 vs 28.69). 30K iterations on A100, ~1 hour.
- **Cannot work with pre-trained GS** - needs joint training from the start.
- **Export**: Groups are "fully decoupled" - each object is an independent set of Gaussians. Extraction scriptable but not built-in.
- **Editing**: Supports removal, inpainting, colorization, style transfer, recomposition via Gaussian Operation List.
- **Rendering**: ~170 FPS (vs ~200 FPS baseline).

### 3. GaussianCut [NeurIPS 2024]
- **Repo**: https://github.com/umangi-jain/gaussiancut
- **Paper**: https://arxiv.org/abs/2411.07555
- **How it works**: Models Gaussians as graph nodes, edges connect k-nearest neighbors. Edge weights combine spatial distance + color similarity (zero-degree SH). Minimizes energy E = unary + lambda*pairwise via graph cut. Unary terms from SAM-Track mask propagation.
- **Post-hoc**: Works on pre-trained 3DGS without any modification or retraining.
- **Interaction**: Point clicks, scribbles, text (via Grounding-DINO). Single view input, SAM-Track propagates across views.
- **Pipeline**: 1) User selects object in Segment-and-Track-Anything UI, 2) Masks propagated across views, 3) Graph cut refinement
- **Export**: Saves to point_cloud/ directory. No explicit .ply per-object documentation.
- **Install**: `conda env create -f environment.yml`

### 4. Click-Gaussian [ECCV 2024]
- **Project**: https://seokhunchoi.github.io/Click-Gaussian/
- **Paper**: https://arxiv.org/abs/2407.11793
- **How it works**: Learns two-level granularity feature fields (coarse + fine) via Global Feature-guided Learning (GFL). Clusters global feature candidates from noisy 2D SAM segments, smoothing noise when learning 3D Gaussian features.
- **Speed**: 10ms per click, 15-130x faster than previous methods.
- **Code**: Not publicly available as of Feb 2026.

### 5. OpenGaussian [NeurIPS 2024]
- **Repo**: https://github.com/yanmin-wu/OpenGaussian
- **Paper**: https://arxiv.org/abs/2406.02058
- **How it works**: 4-stage pipeline: 1) 3DGS pre-training (0-30K steps), 2) Instance feature learning with SAM masks, 3) Two-stage codebook discretization (coarse: position-based clustering, fine: feature refinement), 4) Instance-level 3D-2D feature association linking 3D points to CLIP features.
- **Requires full pipeline** - not post-hoc on arbitrary GS.
- **Interaction**: Script-based click (specify pixel coords) + text queries via pre-computed CLIP embeddings.
- **Output**: PLY with 6D instance features encoded as RGB. Text-guided segmentation renders to cluster directories.
- **Compatibility**: Tightly coupled to ashawkey-diff-gaussian-rasterization (DreamGaussian fork).

### 6. LangSplat [CVPR 2024 Highlight]
- **Repo**: https://github.com/minghanqin/LangSplat
- **Paper**: https://langsplat.github.io/
- **How it works**: Trains scene-wise autoencoder to compress 512D CLIP features to 3D latent space. Each Gaussian stores 3D language feature. Decoder expands back to 512D at render time.
- **Pipeline**: 1) SAM feature extraction, 2) Autoencoder training, 3) LangSplat training (needs pre-trained RGB GS checkpoint), 4) Rendering
- **GPU**: 24GB VRAM, CUDA 11.8, Compute Capability 7.0+
- **199x faster than LERF** at 1440x1080
- **V2**: 450+ FPS rendering (NeurIPS 2025)
- **Requires training from scratch** with language features.

### 7. Lifting By Gaussians (LBG) [WACV 2025]
- **Paper**: https://arxiv.org/abs/2502.00173
- **BEST FOR POST-HOC**: Zero training required. Works on ANY existing 3DGS reconstruction.
- **How it works**: Per-pixel max-contributor Gaussian assignment from 2D SAM masks. Fragments merged across frames using geometric overlap (intersection ratio) + semantic similarity (CLIP/DINOv2 cosine similarity). Running average for feature updates.
- **3D asset extraction**: Yes - includes statistical outlier removal + split-merge via 3D connected component analysis.
- **Speed**: ~450s total (423s preprocessing + 27s segmentation) vs 5207s for SAGA. 10x faster.
- **GPU**: RTX 3090 benchmarked.
- **Compatibility**: Works with 3DGS, 2DGS, any Gaussian parameterization.
- **Code**: Promised upon acceptance, may not be public yet.

### 8. SAGOnline [2025]
- **Paper**: https://arxiv.org/abs/2508.08219
- **Zero-shot, no training**: Uses SAM2 for view-consistent 2D mask propagation.
- **Inverse Projection Voting**: Projects Gaussian centroids to all views, collects instance IDs, assigns most frequent label via histogram voting. GPU-accelerated.
- **Performance**: 92.7% mIoU on NVOS, 95.2% on Spin-NeRF. 15-1500x faster than Feature3DGS/OmniSeg3D/SA3D.
- **GPU**: RTX 4090 benchmarked.
- **Code**: Not available yet.

### 9. ArtisanGS [2026]
- **Paper**: https://arxiv.org/abs/2602.10173
- **Post-hoc**: Works on any in-the-wild capture without original training views.
- **Interactive tools**: SAM click, paint tool, bounding box, frustum projection, depth projection. Selection modes: New/Add/Subtract/Intersect.
- **AI propagation**: 1) Generate ~50 dense views, 2) Cutie video segmentation tracks masks, 3) Optimize per-Gaussian feature channel via differentiable renderer, threshold at 0.5.
- **Speed**: 1-5 seconds for full segmentation.
- **Code**: Not available yet.

### 10. SAGS (Segment Anything in 3D Gaussians)
- **Repo**: https://github.com/XuHu0529/SAGS
- **3-stage pipeline**: 1) User clicks -> SAM masks across views, 2) Gaussian Decomposition for boundary Gaussians + label propagation, 3) 3D voting across views.
- **Key innovation**: Addresses ambiguous Gaussian structure at boundaries by decomposing boundary Gaussians.
- **Supports**: Object removal, translation, rotation, collision mesh extraction.

### 11. GARField (Group Anything with Radiance Fields)
- **Paper**: https://arxiv.org/abs/2401.09419
- **How it works**: Scale-conditioned 3D affinity feature field. A point belongs to different groups at different scales. Hierarchical grouping: scenes -> objects -> sub-parts.
- **Integration with GS**: Queries affinity field at Gaussian centers for 3D segmentation.

### 12. Feature Splatting (nerfstudio)
- **Docs**: https://docs.nerf.studio/nerfology/methods/feature_splatting.html
- **How it works**: Distills CLIP features via view-independent rasterization. SAM-enhanced CLIP + DINOv2 supervision.
- **Requires training from scratch** within nerfstudio.
- **Object-level masked average pooling** for boundary refinement.

### 13. Segment-then-Splat [NeurIPS 2025]
- **Repo**: https://github.com/vulab-AI/Segment-then-Splat
- **Novel approach**: Segments BEFORE reconstruction. Divides Gaussians into object sets before training.
- **Eliminates Gaussian-object misalignment**. No separate language field needed.
- **Requires training from scratch** with pre-segmented data.

---

## DIY Approach: gsply + Spatial Clustering

### gsply Library
- **Install**: `pip install gsply`
- **Load**: `data = GSData.load("model.ply")` or `data = plyread("model.ply")`
- **Attributes**: data.means (N,3), data.scales (N,3), data.quats (N,4), data.opacities (N,), data.sh0 (N,3), data.shN
- **Filter**: `filtered = data[boolean_mask]`, `subset = data[100:200]`
- **Save**: `filtered.save("object.ply")`
- **Speed**: 93M Gaussians/sec read, 57M write. 400K Gaussians in 6-7ms.
- **No CUDA compilation required** - pure Python (NumPy + Numba).

### Simple Position-Based Script
```python
from gsply import plyread
import numpy as np
from sklearn.cluster import DBSCAN

data = plyread("scene.ply")
positions = data.means  # (N, 3)

# Cluster by spatial proximity
clustering = DBSCAN(eps=0.05, min_samples=10).fit(positions)
labels = clustering.labels_

# Export each cluster
for label in set(labels) - {-1}:
    mask = labels == label
    data[mask].save(f"object_{label}.ply")
```

### SAM Mask Projection (DIY Post-hoc)
```python
# Pseudocode for projecting SAM2 masks onto Gaussians
# 1. Render depth from trained GS for each training view
# 2. Run SAM2 on each training image
# 3. For each Gaussian, project its center to each view
# 4. Collect mask labels from all views where it's visible
# 5. Majority vote for final label
# 6. Export labeled subsets
```

---

## Recommendations for Our Use Case (MILo-trained GS of objects on table)

### Best Options (ranked):

1. **DIY gsply + SAM2 projection** - Simplest, works immediately, full control over .ply export
2. **LBG (Lifting By Gaussians)** - Best academic post-hoc method, no retraining, works on any GS
3. **GaussianCut** - Interactive, post-hoc, graph cut refinement, code available
4. **SAGA** - Requires feature training but geometry frozen, good visualization
5. **SAGOnline** - Best performance but no code yet

### For MILo Compatibility:
- Post-hoc methods (LBG, GaussianCut, SAGOnline, ArtisanGS, DIY) work with ANY GS including MILo
- Feature-training methods (SAGA, Click-Gaussian) need adaptation to MILo's rasterizer
- Joint-training methods (Gaussian Grouping, OpenGaussian, LangSplat) require modifying MILo's training loop
