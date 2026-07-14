# vendor/pycuvslam — upstream provenance

| field | value |
|---|---|
| url | https://github.com/NVlabs/PyCuVSLAM (redirect target of `nvidia-isaac/PyCuVSLAM`) |
| commit | `05588424d0440ac80af21080fd1650408fd9282a` (shallow clone 2026-07-14, `.git` stripped) |
| license | NVIDIA Community License — commercially usable, NVIDIA-hardware-only (Oli qualifies; Anton cleared 2026-07-14) |
| why this repo | cuVSLAM standalone Python API (ROS-free, invariance-safe) — the L1 GPU visual-odometry candidate from the 2026-07-10 localization research. Repo now ships the full cuVSLAM C++ source: prebuilt wheels are py3.10/3.12 only, but from-source supports py3.9+ → buildable for our py3.11 `bench-cuvslam` brain env. |
| build path | `cmake -B build && cmake --build build` → `libcuvslam.so`, then `CUVSLAM_BUILD_DIR=<build> pip install python/` (see `../build.sh`) |

## Not committed (gitignored fat assets — see `realizations/.gitignore`)

- `examples/assets/` (191 MB demo media), `examples/pycuvslam.gif` (7.5 MB)
- `doc/images/` (42 MB)
- `test_data/` (2.5 MB LFS samples)
- `build/` (our compile output)

## Our patches

- `cmake/ext/libjpeg.cmake`: add `-DCMAKE_INSTALL_LIBDIR=lib` to the libjpeg-turbo
  ExternalProject — Fedora installs static libs to `lib64/`, upstream's imported target
  hardcodes `lib/libjpeg.a` ("No rule to make target … libjpeg.a" at build). Upstreamable.
- `python/cuvslam2.cpp`: add `#include <algorithm>` — `std::copy_n` is not transitively
  included by GCC 14's libstdc++ ("'copy_n' is not a member of 'std'"). Upstreamable.
