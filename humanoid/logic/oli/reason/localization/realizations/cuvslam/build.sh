#!/usr/bin/env bash
# Compile vendor/pycuvslam (cuVSLAM C++ core -> libcuvslam.so, then the py3.11 nanobind
# bindings) inside `bench-cuvslam` via `locbench env create`. Idempotent — reruns reuse the
# cmake build dir; the pip reinstall is required after any core rebuild (scikit-build-core
# limitation, see vendor README).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/vendor/pycuvslam"
BUILD="$SRC/build"

# System toolchain, on purpose (see environment.yml header):
#   - gcc-14: nvcc rejects Fedora's default gcc-15
#   - /usr/local/cuda-12.9: the glibc-2.41-patched toolkit (cospi/sinpi conflict)
#   - SM 89 only: RTX 4070 Ti SUPER — full-arch builds waste ~10x compile time
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
TOOLCHAIN=(
  -DCMAKE_C_COMPILER=/usr/bin/gcc-14
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-14
)

cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 \
  "${TOOLCHAIN[@]}"
cmake --build "$BUILD" -j"$(nproc)"

# Bindings: scikit-build-core honors CMAKE_ARGS; CUVSLAM_BUILD_DIR locates libcuvslam.so.
export CMAKE_ARGS="${TOOLCHAIN[*]}"
CUVSLAM_BUILD_DIR="$BUILD" pip install "$SRC/python/"

python -c "import cuvslam; print('cuvslam import OK:', cuvslam.__file__)"
