# Feed-Forward 3D Gaussian Splat Completion/Repair: Architecture Research

**Date**: 2026-02-23
**Goal**: Input a GS scene with a hole (region of removed Gaussians), output the missing Gaussians to fill the hole.

---

## 1. Gaussian MAE (ShapeSplat) -- 3DV 2025 Oral

**Paper**: [ShapeSplat](https://arxiv.org/abs/2408.10906) | [GitHub](https://github.com/qimaqi/ShapeSplat-Gaussian_MAE)

### Architecture
- Standard ViT encoder `f_theta` and decoder `g_phi`
- Learnable tokens `T_l` concatenated before decoding
- ~50M parameters (estimated from ViT-base)

### Gaussian Tokenization
- Each Gaussian has 59 dimensions: `X = [C(3), O(1), S(3), R(4), SH(48)] in R^{N x 59}`
- Grouping: FPS on grouping features -> k-NN to find k neighbors per group -> Tokenizer produces `T in R^{n x D}`
- **Grouping features** G(·): typically centroids only, optionally + SH base (3 dims)
- **Embedding features** E(·): configurable subset -- E(C), E(C,S,R), E(All)
- Splats Pooling Layer: temperature-scaled learned aggregation with per-dimension temperature

### Masking Strategy
- Default mask ratio: **60%** (ablated 20%-80%, 40% best for some tasks)
- Random masking (standard MAE approach)

### Loss Functions
- **Centroids**: Chamfer Distance
- **Other params**: L1 loss
- Multiple Conv1d projectors for different attribute groups

### Completability Assessment
- **Not designed for generation** -- decoder recovers masked embedding features for reconstruction loss
- But the MAE pretext task IS completion: reconstruct masked Gaussians from visible ones
- Could be repurposed by: (1) training with intentional hole patterns, (2) using decoder to predict full Gaussian params for masked regions, (3) adding a generative head
- **Key limitation**: operates on OBJECTS (ShapeNet/Objaverse), not scenes
- Normalization: recenters all params (except quaternions) to zero mean, maps to unit sphere

---

## 2. Point Cloud Completion Networks

### PoinTr (ICCV 2021 Oral)
**Paper**: [arxiv](https://arxiv.org/abs/2108.08839) | [GitHub](https://github.com/yuxumin/PoinTr)

- Reformulates completion as set-to-set translation
- Pipeline: downsample partial -> DGCNN local features -> position embedding -> transformer encoder-decoder -> point proxies for missing parts -> MLP + FoldingNet coarse-to-fine
- Geometry-aware block models local geometric relationships
- **For GS adaptation**: decoder outputs point proxies that could be extended with MLP heads to predict scale/rotation/opacity/SH per proxy

### AdaPoinTr (TPAMI 2023)
**Paper**: [arxiv](https://arxiv.org/abs/2301.04545)

- Adaptive query generation mechanism
- Denoising task during completion
- 15x faster training, 20%+ better completion
- Extended to scene-level SSC

### SeedFormer (ECCV 2022)
- "Patch Seeds" representation: captures global structure + preserves local patterns
- Upsample Transformer extends transformer into point generators
- Spatial + semantic relationships between neighboring points

### SnowflakeNet
- Models generation as snowflake-like growth from parent points
- Gradually expands from base to complete shape
- Simulates complex geometric structures

### AnchorFormer (CVPR 2023)
- Predicts discriminative "anchor points" as intermediate representation
- Position offsets from anchors to missing points
- Morphing scheme: deforms canonical 2D grid at each sparse point into 3D structure

### ProxyFormer
- Missing part-sensitive transformer
- Alignment strategy to perceive position of missing parts

### SVDFormer (ICCV 2023)
- Self-view augmentation: projects partial cloud to multiple views
- Self-structure dual-generator (SDG): Structure Analysis + Similarity Alignment paths
- Improved version: PointSea (IJCV 2025)

### PointCFormer (AAAI 2025)
- Relation-based local feature extraction
- Fine-grained relationship metric between target points and k-NN
- Cross-resolution attention
- SOTA on multiple benchmarks

### SPAC-Net (IEEE TG 2025)
- "Interface" structural prior: intersection between known and missing parts
- Marginal Detector (MAD) localizes the interface
- Structure Supplement (SSP) modules augment details before upsampling

### PCDreamer (CVPR 2025)
- Uses multi-view diffusion priors for completion
- Three modules: multi-view image generation, fusion with attention, consistency reduction
- Average CD of 6.52 (10^-3) on PCN dataset -- SOTA

### PriorGroundNet / Completion-by-Correction (2025)
- Starts with topologically complete shape prior from pretrained image-to-3D model
- Performs feature-space correction to align with partial observation
- Shifts from unconstrained synthesis to guided refinement

### DANCE (2025)
- Density-agnostic completion via opacity prediction per candidate point
- Ray-based candidate sampling over observed surface
- Transformer decoder with variable output density

### Key Insight for GS Adaptation
All point cloud completion networks output xyz only. To predict full Gaussian params:
- Add parallel MLP heads for: scale (3), rotation (4), opacity (1), SH coefficients (48 or 3 for DC-only)
- Train with Gaussian-aware losses (rendering loss + per-attribute L1/L2)
- PoinTr/AdaPoinTr most promising base: proven transformer architecture, proxy-based generation naturally extends to multi-attribute prediction

---

## 3. 3D Generative Models Operating on Gaussians Directly

### GaussianCube (NeurIPS 2024)
**Paper**: [arxiv](https://arxiv.org/abs/2403.19655) | [GitHub](https://github.com/GaussianCube/GaussianCube)

- **Representation**: Fixed 32^3 voxel grid, 14 channels per voxel (offset(3), scale(3), rotation(4), opacity(1), RGB(3))
- **Total**: 32,768 Gaussians per object
- **Structuring**: Optimal Transport maps fitted Gaussians to grid centers (Jonker-Volgenant, O(N^3))
- **Generation**: Standard 3D U-Net diffusion (ADM architecture with 3D convolutions)
- **Training**: 16 V100s, ShapeNet + OmniObject3D
- **For completion**: Could mask partial voxels and condition diffusion on known occupancy
- **Limitation**: Object-only, fixed 32^3 resolution, not scalable to scenes

### DiffGS (NeurIPS 2024)
**Paper**: [arxiv](https://arxiv.org/abs/2410.19657) | [GitHub](https://github.com/weiqi-zhang/DiffGS)

- **Key innovation**: Disentangles GS into 3 functions on a triplane:
  - GauPF (Gaussian Probability Function): P(location is Gaussian center)
  - GauCF (Gaussian Color Function): predicts color from position
  - GauTF (Gaussian Transform Function): predicts rotation, scale, opacity from position
- **Gaussian VAE**: encoder -> latent z -> triplane decoder -> function predictors
- **Latent Diffusion**: DALLE-2 backbone, cross-attention for conditioning
- **Discretization**: Octree-guided sampling, then gradient descent on GauPF to refine centers
- **Conditional generation from partial 3DGS**: Modified PointNet encoder for partial GS -> global embedding -> cross-attention in diffusion
  - Partial input: 1/8 of Gaussians (7/8 occluded)
- **Scale**: Tested 50K-350K Gaussians per shape
- **DIRECTLY RELEVANT**: Has partial-3DGS-to-complete-3DGS pipeline already!

### GVGEN (ECCV 2024)
**Paper**: [arxiv](https://arxiv.org/abs/2403.12957) | [GitHub](https://github.com/SOTAMak1r/GVGEN)

- GaussianVolume: structured form with fixed number of Gaussians
- Candidate Pool Strategy for pruning/densifying
- Coarse-to-fine: basic geometry -> full Gaussian attributes
- Text-to-3D in ~7 seconds

### GaussianAnything (ICLR 2025)
**Paper**: [arxiv](https://arxiv.org/abs/2411.08033) | [GitHub](https://github.com/NIRVANALAN/GaussianAnything)

- 3D VAE with point-cloud structured latent space
- Cross-attention with sparse point cloud -> shape-texture disentanglement
- DiT-based decoder: upsamples latent point cloud to dense surfel Gaussians
- Cascaded diffusion with AdaLN-single and QK-Norm
- Multi-modal input: point cloud, caption, single/multi-view images
- **Good candidate for adaptation**: already accepts point cloud input

### Atlas Gaussians (ICLR 2025 Spotlight)
- Patches with UV-based sampling for infinite Gaussians
- Transformer decoder maps latents to Atlas Gaussians
- Disentangles geometry and appearance
- VAE + latent diffusion pipeline

### DiffSplat (ICLR 2025)
**Paper**: [arxiv](https://arxiv.org/abs/2501.16764) | [GitHub](https://github.com/chenguolin/DiffSplat)

- Repurposes 2D image diffusion for 3D GS generation
- Treats GS 2D grids as "images in a special style"
- Image VAE fine-tuned to encode Gaussian properties
- Rendering loss for 3D consistency
- 1-2 second generation

### G3PT (IJCAI 2025)
- Autoregressive coarse-to-fine generation
- Cross-scale vector quantization
- Cross-scale Querying Transformer (CQT): cross-attention across scale levels
- Power-law scaling behaviors observed

### LGM (ECCV 2024 Oral)
- Multi-view Gaussian features from asymmetric U-Net backbone
- Attention blocks for cross-view information sharing
- 5-second generation, 512 resolution training

### GRM (ECCV 2024)
**Paper**: [arxiv](https://arxiv.org/abs/2403.14621) | [GitHub](https://github.com/justimyhxu/GRM)

- Pure transformer: 24 self-attention layers, 768 width, 12 heads
- Pixel-aligned Gaussians: 12 attributes per pixel (depth, quaternion(4), scale(3), opacity(1), DC SH(3))
- 4-block upsampler with windowed self-attention + PixelShuffle
- **~1M Gaussians** per scene (4 views x 512x512)
- 0.1s inference, trained on 100K Objaverse objects, 32 A100s, 4 days
- **Limitation**: "lacks capability for hallucination" -- pure reconstruction, no generative priors

### LPM (Large Point-to-Gaussian Model)
- U-Net encoder-decoder with APP blocks
- Input: 4K points upsampled to 16K -> 16K Gaussians
- Multi-head Gaussian Decoder: separate linear heads per attribute
- Attributes: position offset, scale, rotation(quat), opacity, DC SH
- Loss: MSE + LPIPS + Chamfer Distance + EMD
- 16 A100s, 3 days, Objaverse-LVIS

### SplatFormer (ICLR 2025)
**Paper**: [arxiv](https://arxiv.org/abs/2411.06390) | [GitHub](https://github.com/ChenYutongTHU/SplatFormer)

- **First point transformer operating directly on Gaussian splats**
- Based on Point Transformer V3 (PTv3)
- Encoder: 5 levels, blocks (2,2,2,6,2), dims (64,96,128,256,512)
- Decoder: 4 levels, blocks (2,2,2,2)
- Grid resolution 384, pooling strides (1,2,2,2)
- **50M parameters**
- Processes ALL Gaussian attributes: position, scale(3), rotation(4), opacity, SH
- **Outputs residuals**: G'_k = G_k + delta_G_k
- Loss: L1 + LPIPS (photometric)
- 108ms inference, trained on 33K-48K ShapeNet/Objaverse scenes
- **Limitation for completion**: fixed K Gaussians in = K out, no new Gaussians generated
- **Adaptation needed**: variable-output decoder, new loss for geometric fill

---

## 4. Conditional 3D Generation / Inpainting

### ComPC (ICLR 2025)
**Paper**: [arxiv](https://arxiv.org/abs/2404.06814) | [GitHub](https://github.com/Tianxinhuang/ComPC)

- **Optimization-based** (not feed-forward)
- Converts partial PC to Gaussians, optimizes with 2D diffusion guidance
- Preservation Constraint maintains input geometry
- Object-level only

### RI3D (ICCV 2025)
- Two personalized diffusion models: "repair" + "inpainting"
- Repair: rendered image -> high-quality pseudo GT
- Inpainting: hallucinate unobserved areas
- Two-stage optimization strategy

### GSFix3D (2025)
- Diffusion-guided novel view repair for 3DGS
- Scene-adapted latent diffusion fine-tuned per scene
- Random mask augmentation for inpainting capability
- NOT feed-forward

### SplatFill (2025)
- Depth-guided 3D scene inpainting
- Joint depth + object supervision
- Consistency-aware refinement
- 24.5% faster than alternatives

### Inpaint360GS (2025)
- 360 degree scene inpainting via GS
- Multi-object removal support
- Directly manipulates Gaussian field

### SurfFill (2025)
- LiDAR point cloud completion via Gaussian surfel splatting
- Ambiguity heuristic identifies missing areas
- Grows Gaussians into ambiguous regions
- Divide-and-conquer for **building-scale scans with tens of millions of points**
- Key insight: use GS optimization to fill detected gaps

### Complete GS from Single Image (2025)
- VAR (Variational AutoReconstructor) learns latent space from 2D images only
- Latent diffusion conditioned on input image
- 15 params per Gaussian: opacity(1), depth(1), offset(3), scale(3), rotation(4), color(3)
- 2 Gaussians per pixel, 32K total
- Diverse samples via classifier-free guidance
- 50 diffusion steps, 2.7s inference

### PVD (ICCV 2021)
- Point-voxel diffusion for shape generation + conditional completion
- Denoising from noise to point cloud
- Can condition on partial observations

### LION (NeurIPS 2022)
- Hierarchical latent point diffusion
- VAE with global + point-structured latent space
- Text/image-driven generation

---

## 5. Tokenization of Gaussians for Transformers

### Strategy 1: Per-Gaussian Tokenization (SplatFormer)
- Each Gaussian = one token
- Attributes concatenated as feature vector
- PTv3 serialization with space-filling curves (Z-order, Hilbert)
- Voxelization for spatial organization (grid resolution 384)
- Pooling strides for hierarchy
- **Scales to ~50K Gaussians** in practice

### Strategy 2: FPS + kNN Grouping (Gaussian MAE)
- FPS to select n centers
- kNN to group k neighbors per center
- Per-group tokenizer (mini-PointNet)
- Group size and number configurable
- Feature normalization: unit sphere mapping per attribute dimension

### Strategy 3: Voxel Grid (GaussianCube)
- Fixed N^3 grid (32^3 = 32K Gaussians)
- Optimal Transport assignment (Gaussians -> grid cells)
- 14-dim feature per voxel
- Enables standard 3D U-Net/CNN architectures
- **Not scalable** to large scenes

### Strategy 4: Triplane + Functions (DiffGS)
- No explicit tokenization of individual Gaussians
- Triplane feature representation (H x W x C x 3)
- Three function networks predict attributes from 3D position queries
- Octree-guided sampling for discretization
- **Scales to 350K Gaussians**

### Strategy 5: Pixel-Aligned (GRM, LGM, DiffSplat)
- Gaussians organized as 2D grids aligned with camera views
- Each pixel predicts Gaussian params
- Standard 2D transformer/U-Net architectures work
- **Scales to ~1M Gaussians**

### Strategy 6: Patch-based UV mapping (Atlas Gaussians)
- Shape = union of local patches
- Each patch decodes Gaussians via UV sampling
- Transformer decoder over patch features
- Disentangled geometry/appearance

### Strategy 7: Cross-scale VQ (G3PT)
- Multi-resolution vector quantization
- Coarse-to-fine token hierarchy
- Cross-scale attention between levels

### Strategy 8: Point Cloud as SSM Sequence (Gamba, MVGamba)
- Mamba/SSM-based processing
- Serialized point cloud as 1D sequence
- Linear complexity vs quadratic for attention
- But less flexible than attention for sparse 3D data

### Strategy 9: Structured Surfel Latents (GaussianAnything)
- Cross-attention between unordered latent tokens and sparse point cloud
- Concretizes tokens into point-cloud-structured latent space
- DiT-based upsampling

---

## 6. Feasibility Analysis for Scene-Level GS Completion

### Scale Challenge
- Objects: 30K-350K Gaussians (handled by existing methods)
- Scenes: 100K-1M+ Gaussians (requires new approaches)
- PTv3 handles millions of points with linear complexity
- SurfFill handles building-scale with tens of millions

### Most Promising Architectures for Our Task

#### Option A: DiffGS-style Functional Approach
- **Already has partial-3DGS conditioning**
- Triplane representation scales well
- Octree discretization = arbitrary output count
- Would need: scene-level training data, larger triplane, multi-scale functions
- **Limitation**: diffusion = iterative, not truly feed-forward

#### Option B: PoinTr/AdaPoinTr + Gaussian Head
- Proven completion architecture
- Add parallel attribute prediction heads
- Geometry-aware transformer handles spatial relationships
- **Limitation**: designed for 2K-16K point outputs, needs scaling work

#### Option C: SplatFormer (PTv3) Modified for Variable Output
- Already operates on raw Gaussian splats
- PTv3 backbone scales to large point clouds
- Would need: (1) variable-length output, (2) generative decoder, (3) hole-aware conditioning
- **Most architecturally clean** for scene-level

#### Option D: GaussianCube Conditional Diffusion
- Condition 3D U-Net on partial occupancy mask
- Well-suited for fixed-size completion
- **Limitation**: 32^3 resolution too coarse for scenes

#### Option E: Hybrid -- SplatFormer encoder + DiffGS-style functional decoder
- PTv3 encodes partial Gaussian scene (any size)
- Cross-attention to triplane features
- Functional decoder generates missing Gaussians via octree sampling
- Combines best of both: scalable encoding + flexible generation

### Training Data Requirements
- Paired data: complete GS scenes + artificially holes
- Sources: ShapeSplat (206K objects), Objaverse, ScanNet++, Replica
- Augmentation: random region removal, object removal, view-dependent masking
- Rendering-based supervision possible (no need for GT Gaussian params)

### Recommended Architecture (our use case)
**SplatFormer encoder + generative decoder** because:
1. PTv3 proven on Gaussian splats, 50M params, 108ms inference
2. Scene-level scalability via serialization + space-filling curves
3. Residual refinement of existing Gaussians + generation of new ones
4. Can train with rendering loss (no GT Gaussian labels needed)
5. Could start from SplatFormer pretrained weights

Key modifications needed:
- Add "hole detection" module (identify regions needing completion)
- Variable-output decoder that generates NEW Gaussians (not just refines existing)
- Condition decoder on boundary Gaussians around the hole
- Multi-scale generation: coarse position first, then attributes
