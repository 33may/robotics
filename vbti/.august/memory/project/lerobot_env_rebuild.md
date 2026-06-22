---
name: lerobot env rebuild (2026-04-13)
description: LeRobot conda env was rebuilt — torchcodec ABI fix required PyTorch 2.7 + torchcodec 0.5
type: project
originSessionId: 1a4916d7-6577-4f75-b1aa-dc5c75803a7c
---
## LeRobot env rebuild (2026-04-13)

**Problem**: torchcodec 0.10.0 pip wheels ship with CXX11_ABI=False .so files. PyTorch pip wheels (cu124) also use ABI=False, but the torchcodec .so for FFmpeg 7 had an ABI mismatch (`undefined symbol: _ZN3c1013MessageLogger6streamB5cxx11Ev`). Even after upgrading PyTorch to 2.7 (ABI=True), torchcodec 0.10 still failed.

**Fix**: torchcodec 0.5 works with both ABI variants. It's within LeRobot 0.4.4's pin (`>=0.2.1,<0.11.0`).

**Why:** The isaac env (PyTorch 2.7+cu128, torchcodec 0.5) worked fine — matched the working config.

**How to apply:** If torchcodec breaks again, use version 0.5, not 0.10.x. The 0.10 pip wheels have known ABI issues.

### Final working config
```
conda create -n lerobot python=3.12
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
cd lerobot/ && pip install -e '.[dev]'
pip install torchcodec==0.5
```

### Versions
- PyTorch 2.7.0+cu128 (CXX11_ABI=True)
- torchvision 0.22.0+cu128
- torchcodec 0.5
- LeRobot 0.4.4 (editable from lerobot/)
- Python 3.12, CUDA 12.8
