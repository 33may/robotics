# Vendored: EMS superquadric recovery

**Upstream:** https://github.com/bmlklwx/EMS-superquadric_fitting
**Commit:** `5a213d46e8b013b5a153f5aa914b0925ee517af7` (2024-10-03)
**License:** MIT — full text in `LICENSE`, © 2022 Weixiao Liu.

**Paper:** Weixiao Liu, Yuwei Wu, Sipu Ruan, Gregory S. Chirikjian,
*Robust and Accurate Superquadric Recovery: a Probabilistic Approach*,
CVPR 2022 (Oral). arXiv:2111.14517.

## Why vendored rather than installed

Upstream is not published to PyPI (the `EMS` name on PyPI is an unrelated
0.0.0 placeholder), and its `setup.py` declares `mayavi` and `plyfile` as
hard install requirements although the algorithm needs neither — `mayavi`
is only reached from `utilities.showSuperquadrics`. Vendoring the two
algorithm modules keeps a heavy VTK/Qt stack out of a shared conda env.

## Files taken

| file | upstream path |
|---|---|
| `EMS_recovery.py` | `Python/src/EMS/EMS_recovery.py` |
| `superquadrics.py` | `Python/src/EMS/superquadrics.py` |

`Python/src/EMS/utilities.py` was **not** taken (mayavi/plyfile only).

## Local patches — all marked `# PATCH:` in the source

1. `EMS_recovery.py`: import rewritten to the vendored path.
2. `EMS_recovery.py`: `np.int(0)` → `int(0)`. `np.int` was removed in
   numpy 1.24; the env runs numpy 2.3.5, so upstream raises `AttributeError`
   on the first call without this.
3. `superquadrics.py`: dropped `import EMS.utilities` and made
   `showSuperquadric()` raise `NotImplementedError`. That is the only
   mayavi entry point; rendering lives in `inspection/fit/render.py`.

The algorithm itself — GUM likelihood, the E/M/S steps, the similarity
switch, the radial distance — is unmodified.

## Runtime dependency

`numba` (JIT for the inner loops). Installed in the `robo` env
(numba 0.67.0 + llvmlite 0.49.0, no numpy downgrade).
Measured: first call ~7.5 s (JIT), warm ~0.25-0.36 s per 6-7k-point cloud.
