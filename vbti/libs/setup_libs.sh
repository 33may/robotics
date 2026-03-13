#!/bin/bash
# Setup vendored libraries for VBTI pipeline
# Run from repo root: bash vbti/libs/setup_libs.sh

set -e

LIBS_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Installing libraries to: $LIBS_DIR"

# MILo — Mesh extraction from Gaussian Splats (SIGGRAPH Asia 2025)
# Used by: gs_milo_utils.py
if [ ! -d "$LIBS_DIR/MILo" ]; then
    echo "Cloning MILo..."
    git clone https://github.com/Anttwo/MILo.git "$LIBS_DIR/MILo"
    cd "$LIBS_DIR/MILo"
    git submodule update --init --recursive
    cd -
else
    echo "MILo already exists, skipping"
fi

# Nerfstudio — GS training + COLMAP wrapper
# Used by: colmap_utils.py (ns-process-data)
if [ ! -d "$LIBS_DIR/nerfstudio" ]; then
    echo "Cloning nerfstudio..."
    git clone https://github.com/nerfstudio-project/nerfstudio.git "$LIBS_DIR/nerfstudio"
else
    echo "nerfstudio already exists, skipping"
fi

# sharp-frame-extractor — Quality-based video frame extraction
# Used by: video_utils.py
if [ ! -d "$LIBS_DIR/sharp-frame-extractor" ]; then
    echo "Cloning sharp-frame-extractor..."
    git clone https://github.com/cansik/sharp-frame-extractor.git "$LIBS_DIR/sharp-frame-extractor"
else
    echo "sharp-frame-extractor already exists, skipping"
fi

echo ""
echo "Libraries cloned. Next steps:"
echo ""
echo "1. Activate gsplat-pt25 conda env"
echo "2. Install nerfstudio:     pip install -e $LIBS_DIR/nerfstudio"
echo "3. Install sharp-frame-extractor: pip install -e $LIBS_DIR/sharp-frame-extractor"
echo "4. Install MILo submodules (requires gcc-14, CUDA 12.9):"
echo "   See docs/project_knowledge_base.md for MILo build instructions"
