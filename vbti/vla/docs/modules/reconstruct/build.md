# Building the simulation environments

`envs/gsplat-pt25.yml` and `envs/isaac.yml` resolve with conda, but several pip packages in them are built from source and are on no index — install those by hand as below.

## `gsplat-pt25` — reconstruction (COLMAP, MILo, mesh processing)

Vendored libraries live in `libs/` (cloned repos, not submodules):

```bash
bash vbti/vla/libs/setup_libs.sh      # clones MILo, nerfstudio, sharp-frame-extractor
pip install -e vbti/vla/libs/nerfstudio
pip install -e vbti/vla/libs/sharp-frame-extractor
pip install viser==1.0.21
```

| Library | Repo | Used by |
|---|---|---|
| MILo | `Anttwo/MILo` | `gs_milo_utils.py` — mesh extraction from Gaussian splats |
| nerfstudio | `nerfstudio-project/nerfstudio` | `colmap_utils.py` — COLMAP SfM wrapper |
| sharp-frame-extractor | `cansik/sharp-frame-extractor` | `video_utils.py` — quality-based frame selection |

MILo and its CUDA extensions (diff-gaussian-rasterization, simple-knn, fused-ssim, nvdiffrast) must be compiled locally. Notes from building on Fedora 42:

```bash
sudo dnf install -y gcc14-c++ gcc14        # system GCC 15 is unsupported by CUDA
export CUDA_HOME=/usr/local/cuda-12.9
export CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14
export TORCH_CUDA_ARCH_LIST="8.9"          # set to your GPU
export MAX_JOBS=8
```

- glibc 2.41 has `cospi`/`sinpi` conflicts — patch CUDA `math_functions.h` if the MILo build fails there
- 9 MILo headers miss `#include <cstdint>` — add it if compilation errors on fixed-width ints

## `isaac` — Isaac Sim 5.0 + Isaac Lab 0.47.1

Install Isaac Sim / Isaac Lab per NVIDIA docs, then LeIsaac from source. numpy must stay < 2 (`omni.syntheticdata` breaks on numpy 2; opencv ≥ 4.11 pulls numpy 2 — keep the pinned opencv).
