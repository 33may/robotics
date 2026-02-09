# Gaussian Splatting for Industrial Robotics Simulation
## Research Report - January 2026

---

## Executive Summary

Gaussian Splatting (GS) has reached an inflection point for industrial robotics simulation in 2025-2026. NVIDIA's ecosystem dominates the space, with their **NuRec** library now integrated into Omniverse/Isaac Sim providing a complete smartphone-to-simulation pipeline. The recommended stack for your setup (RTX 4070 Ti SUPER, 16GB VRAM) is:

**gsplat + 3DGUT + COLMAP/GLOMAP → Isaac Sim 5.0**

Key findings:
- **3DGUT** is essential for real robot cameras - handles lens distortion that standard GS cannot
- **Physics simulation** is the current gap - mesh extraction required for complex collision
- **glTF standardization** (Aug 2025) enables interoperability; **SPZ format** provides 90% compression
- Major robotics companies (Covariant, Figure AI, Agility) all use NVIDIA Isaac Sim

---

## 1. Industry Landscape

### 1.1 Company Tech Stacks

| Company | Simulation Platform | Key Technologies |
|---------|---------------------|------------------|
| **NVIDIA** | Isaac Sim + Omniverse | PhysX, Newton, Cosmos WFMs, NuRec (GS) |
| **Covariant** (Amazon) | Custom + NVIDIA GPUs | RFM-1 foundation model, video-based world model |
| **Google Intrinsic** | Custom + NVIDIA | IVM, Genie 3, RoboBallet, sub-mm precision |
| **Boston Dynamics/RAI** | Proprietary + RL | Zero-shot sim-to-real transfer |
| **Agility Robotics** | NVIDIA Isaac Sim | LSTM whole-body control, Jetson AGX Thor |
| **Figure AI** | NVIDIA infrastructure | Helix VLA (7-9Hz planning, 200Hz control) |

### 1.2 GS Adoption in Industry

- **1,692 papers** published on arxiv in 2025 alone
- **NVIDIA NuRec** announced at SIGGRAPH 2025 for Omniverse/Isaac Sim
- **Applied Intuition NeuralSim** powered by Gaussian Splatting
- **Tesla** using GS for autonomous systems
- **Third Dimension SuperSim** - simulator built from reality using GS

### 1.3 Key Research Projects

| Project | Description | Use Case |
|---------|-------------|----------|
| **SplatSim** (CMU) | Zero-shot Sim2Real using GS-rendered RGB | Diffusion policies |
| **RL-GSBridge** | RL training with 3DGS environments | Manipulation policies |
| **GWM** | Diffusion Transformer + 3D VAE | Model-based RL |
| **NVIDIA Physically Embodied Gaussians** | Real-time digital twin sync | Live robotics |

### 1.4 Market Verticals

**Automotive (BMW, Mercedes-Benz, Tesla):**
- BMW Virtual Factory: collision checks reduced 4 weeks → 3 days
- Market: $2.1B (2024) → $28.7B (2034) at 30.1% CAGR

**Warehouse/Logistics:**
- Amazon Robotics: development years → months (BlueJay: concept to production ~1 year)
- Locus Robotics: 350+ sites globally, AGV/AMR market ~30% CAGR

**Manufacturing:**
- TSMC: Omniverse for fab design, Isaac for robotics at Phoenix facility

---

## 2. Reconstruction Pipelines

### 2.1 Recommended End-to-End Pipeline

```
Capture → Pose Estimation → Training → Cleanup → Export → Isaac Sim
   ↓           ↓               ↓          ↓         ↓          ↓
Gimbal    GLOMAP/COLMAP     gsplat    Clean-GS    USDZ    NuRec import
60+ FPS   (10-100x faster)  (4x VRAM)  (2-5min)   (SPZ)   + physics proxy
```

### 2.2 Camera Pose Estimation

| Tool | Approach | Speed | Best Use |
|------|----------|-------|----------|
| **COLMAP** | Incremental SfM | Baseline | High accuracy |
| **GLOMAP** | Global SfM | **10-100x faster** | Large-scale |
| **hloc** | Hierarchical localization | Variable | Learning-based features |
| **FASTMAP** | Dense SfM | 1-2 orders faster | Thousands of images |

GLOMAP achieves 8% higher recall with 9-point AUC improvement at 0.1m threshold.

### 2.3 Training Tools Comparison

| Tool | VRAM Efficiency | Speed | Key Features |
|------|-----------------|-------|--------------|
| **gsplat** | **4x less** | **15% faster** | CUDA backend, 3DGUT integrated, Apache 2.0 |
| **Nerfstudio Splatfacto** | Standard | Standard | Full pipeline, web viewer, CLI |
| **3DGRUT** (NVIDIA) | - | 347-846 FPS | Reflections, refractions, ray tracing |

**Commercial Tools:**
- **Postshot** (Jawset): Best for local, live preview, Blender/Unreal export
- **Polycam**: Easy mobile capture, cloud processing
- **Luma AI**: Dream Machine AI, PLY export only

### 2.4 Data Capture Best Practices

**Equipment:**
- Wide-angle lens (e.g., Sigma 14mm f/1.4)
- Gimbal stabilization (e.g., DJI RS 4)
- Smartphone works for basic scenes (lock focus/exposure)

**Guidelines:**
- 20-30% overlap between views (60% for NuRec workflow)
- 60+ FPS if using video
- 100-250 images for small objects
- Even, diffuse lighting
- No scene movement during capture
- Minimize motion blur and shallow DoF

**Challenging Lighting:**
- **Luminance-GS**: Handles low-light and overexposure with view-adaptive curve adjustment

### 2.5 Quality Metrics

| Metric | Good Value | Measures |
|--------|------------|----------|
| **PSNR** | >30 dB | Pixel-level accuracy |
| **SSIM** | >0.9 | Structural similarity |
| **LPIPS** | <0.1 | Perceptual similarity |

**Benchmark Datasets:** Mip-NeRF 360, Tanks & Temples, ScanNet++

---

## 3. Post-Processing & Cleanup

### 3.1 Floater Removal

| Method | Speed | Quality |
|--------|-------|---------|
| **EFA-GS** | - | +3.57 dB PSNR over Mip-splatting |
| **Clean-GS** | **2-5 min CPU** | 60-80% model compression |
| **CloudCompare** | Manual | Point cloud view only |

### 3.2 Background Removal / Segmentation

| Method | Speed | Feature |
|--------|-------|---------|
| **FlashSplat** (ECCV 2024) | **~30 sec** | 50x faster, robust to noise |
| **Gaussian Grouping** | - | SAM integration |
| **Object-Centric 2DGS** | 71% faster training | 96% smaller models |
| **Clean-GS** | 2-5 min | Only needs 3 masks (1% of views) |

### 3.3 Mesh Extraction

| Method | Technique | Best For |
|--------|-----------|----------|
| **SuGaR** | Poisson reconstruction | Blender/Unity export |
| **OMeGa** | Joint mesh-splat optimization | +37-80% F-score |
| **2DGS** | Depth map extraction | Simple scenes |
| **MILo** (SIGGRAPH Asia 2025) | Mesh-In-the-Loop | Physics simulation |

---

## 4. Scene Editing

### 4.1 Text-Guided Editing

| Tool | Speed | Key Feature |
|------|-------|-------------|
| **GaussianEditor** | ~20 min | Gaussian semantic tracing |
| **Instruct-GS2GS** | Variable | InstructPix2Pix, nerfstudio integrated |
| **GaussCtrl** (ECCV 2024) | Faster | Multi-view consistent via ControlNet |
| **3DSceneEditor** | Fastest | Fully 3D-based, lowest GPU memory |

### 4.2 Interactive Editors

**SuperSplat** (PlayCanvas) - The leading free tool:
- Browser-based, MIT licensed
- v2.14: video export (WebM/MOV/MKV), VR/AR (Quest 2/3, Vision Pro)
- Timeline animation
- **Production-ready**

### 4.3 Relighting & Materials

- **Ref-Gaussian** (ICLR 2025): Real-time reflective objects with PBR
- **Large Material Gaussian Model**: Generates albedo, roughness, metallic
- **Relightable 3D Gaussian**: Per-Gaussian BRDF with BVH ray-tracing

---

## 5. Export & Integration

### 5.1 Export Formats

| Format | Status | Size | Notes |
|--------|--------|------|-------|
| **PLY** | Standard | 250MB+ | Universal but large |
| **SPZ** (Niantic) | Production | **90% smaller** | "JPG for splats", MIT licensed |
| **glTF** | **Standardized Aug 2025** | Variable | KHR_gaussian_splatting extension |
| **USD/USDZ** | NVIDIA custom | Variable | OmniNuRecVolumeAPI schema |

### 5.2 NVIDIA Isaac Sim Integration (NuRec)

**Complete Workflow:**

1. **Capture** - Smartphone, lock focus/exposure, 60% overlap, JPEG
2. **Sparse Reconstruction** - COLMAP with pinhole camera model
3. **Dense Reconstruction** - 3DGRUT with MCMC densification
4. **Export** - USDZ with `apply_normalizing_transform=true`
5. **Simulate** - Isaac Sim 5.0+, add ground plane physics proxy, insert robot

**Requirements:**
- Isaac Sim 5.0+
- Linux with CUDA 11.8+, GCC <= 11
- Pre-made scenes on Hugging Face (NVIDIA Physical AI)

### 5.3 Physics Proxy Generation

| Method | Approach | Best For |
|--------|----------|----------|
| **PhysGaussian** | MPM on Gaussians, no meshing | Soft-body, deformable |
| **MILo** | Mesh-In-the-Loop | Complex collision |
| **MaGS** (ICCV 2025) | Mesh-adsorbed Gaussians | Soft-body + quality |
| **DecoupledGaussian** | Proxy points from depth | Newtonian physics |

**Current Gap:** Isaac Sim NuRec uses simple ground plane proxy. Complex collision needs mesh extraction.

---

## 6. Emerging Methods (2025-2026)

### 6.1 3DGUT (CVPR 2025 Oral) - CRITICAL FOR ROBOTICS

- Handles **distorted cameras** (fisheye, rolling shutter) using Unscented Transform
- Supports reflections/refractions via hybrid rendering
- Real robot cameras have lens distortion standard GS cannot handle
- **Free in gsplat library**

### 6.2 4D Gaussian Splatting (Dynamic Scenes)

- **4DGS-1K** (NeurIPS 2025): 1000+ FPS for dynamic scenes
- **MEGA** (ICCV 2025): 7% of baseline Gaussians
- Native 4D primitives with Spherindrical Harmonics

### 6.3 Language-Guided Editing

- **LangSplat**: 199x faster than LERF, SAM for hierarchical semantics
- **LangSplat V2** (NeurIPS 2025): **450+ FPS**
- **4D LangSplat** (CVPR 2025): Dynamic scenes with MLLM

### 6.4 Mesh Hybrids

- **MeGA** (CVPR 2025): Mesh for face, Gaussians for hair
- **HybridSplat**: Reflection-baked Gaussian tracing
- **SDFoam**: SDF-biased hybrid for cleaner surfaces

---

## 7. Hardware Requirements

| GPU | VRAM | Use Case |
|-----|------|----------|
| RTX 3060 | 12GB | Entry-level |
| **RTX 4070 Ti SUPER** | **16GB** | **Your setup - good with gsplat** |
| RTX 3090/4090 | 24GB | Paper-quality |
| A6000 | 48GB | Professional |
| A100 | 80GB | Large-scale |

**Tips for 16GB VRAM:**
- Use gsplat (4x more efficient)
- Increase `--densify_grad_threshold`
- Reduce `--densify_until_iter`
- Use FLoD for flexible Level-of-Detail

---

## 8. Recommendations

### 8.1 Immediate Actions

1. **Install gsplat** in your `isaac` conda environment
2. **Test NuRec pipeline** with a smartphone capture of your workspace
3. **Verify 3DGUT** works with distorted camera footage

### 8.2 Recommended Stack

```
gsplat (training) + 3DGUT (distortion) + GLOMAP (poses)
            ↓
    Clean-GS (cleanup) + FlashSplat (segmentation)
            ↓
    SuGaR/MILo (mesh for physics)
            ↓
    Isaac Sim 5.0 via NuRec (USDZ export)
```

### 8.3 Key Decisions to Make

1. **Camera**: Smartphone vs dedicated camera with gimbal?
2. **Physics**: Simple proxy (ground plane) vs full mesh collision?
3. **Dynamic scenes**: Static GS sufficient or need 4DGS?
4. **Semantics**: Need LangSplat for language queries?

---

## Sources

### Industry Landscape
- [Covariant AI Foundation Model](https://covariant.ai/insights/the-future-of-robotics-robotics-foundation-models-and-the-role-of-data/)
- [Google Intrinsic AI for Industry](https://www.intrinsic.ai/events/ai-for-industry-challenge)
- [NVIDIA Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/)
- [Agility Robotics Whole-Body Control](https://www.agilityrobotics.com/content/training-a-whole-body-control-foundation-model)
- [Figure AI Wikipedia](https://en.wikipedia.org/wiki/Figure_AI)
- [BMW Virtual Factory](https://www.assemblymag.com/articles/99322-bmw-scales-virtual-factory-with-accelerated-computing-digital-twins-and-ai)

### GS Pipelines & Tools
- [gsplat GitHub](https://github.com/nerfstudio-project/gsplat)
- [Nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html)
- [GLOMAP GitHub](https://github.com/colmap/glomap)
- [Clean-GS arXiv](https://arxiv.org/abs/2601.00913)
- [FlashSplat ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03300.pdf)
- [SuGaR GitHub](https://github.com/Anttwo/SuGaR)

### Editing & Integration
- [SuperSplat Editor](https://superspl.at/editor)
- [GaussianEditor GitHub](https://github.com/buaacyw/GaussianEditor)
- [3DGUT NVIDIA](https://research.nvidia.com/labs/toronto-ai/3DGUT/)
- [NuRec NVIDIA Blog](https://developer.nvidia.com/blog/reconstruct-a-scene-in-nvidia-isaac-sim-using-only-a-smartphone/)
- [glTF GS Standard](https://www.cgchannel.com/2025/08/3d-gaussian-splats-are-being-added-to-the-gltf-standard/)
- [SPZ Format](https://scaniverse.com/news/spz-gaussian-splat-open-source-file-format)
- [LangSplat](https://langsplat.github.io/)

---

*Research conducted: January 27, 2026*
*Knowledge Graph Project: robotics (4537ffeb-e483-4d91-987f-8bab60900c49)*
*Task: Tools Research (290994be-9b37-471c-b220-81ae8aa38f83)*
