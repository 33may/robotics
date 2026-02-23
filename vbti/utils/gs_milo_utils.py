"""
MILo Gaussian Splatting reconstruction utilities.

Wraps MILo train.py (GS training) and mesh_extract_sdf.py (mesh extraction).
Takes undistorted COLMAP output and produces a mesh PLY.

Usage:
    python vbti/utils/gs_milo_utils.py reconstruct --source_dir data/undistorted --model_dir data/milo_output
    python vbti/utils/gs_milo_utils.py train_gs --source_dir data/undistorted --model_dir data/milo_output
    python vbti/utils/gs_milo_utils.py extract_mesh --source_dir data/undistorted --model_dir data/milo_output
"""

import os
import subprocess
import sys
from pathlib import Path

# MILo lives in vbti/libs/MILo/milo/
MILO_DIR = Path(__file__).resolve().parent.parent / "libs" / "MILo" / "milo"

# Fedora 42 build environment — GCC 15 unsupported by CUDA, use 14
# NOTE: Do NOT set CPLUS_INCLUDE_PATH=/usr/include here — it breaks
# #include_next <stdlib.h> in nvdiffrast JIT compilation by deduplicating
# /usr/include from GCC's default search path (see GCC include_next ordering).
# LIBRARY_PATH for nvdiffrast GL linking.
MILO_ENV = {
    **os.environ,
    "CUDA_HOME": "/usr/local/cuda-12.9",
    "CC": "/usr/bin/gcc-14",
    "CXX": "/usr/bin/g++-14",
    "LIBRARY_PATH": "/usr/lib64",
    "TORCH_CUDA_ARCH_LIST": "8.9",
}


def train_gs(
    source_dir: str,
    model_dir: str,
    mesh_config: str = "default",
    imp_metric: str = "indoor",
    rasterizer: str = "radegs",
    iterations: int = 18000,
    data_device: str = "cpu",
    config_path: str = "./configs/fast",
):
    """Train MILo Gaussian Splatting model.

    Args:
        source_dir: path to undistorted COLMAP output (PINHOLE cameras)
        model_dir: where to save the trained model
        mesh_config: mesh quality preset (verylowres/lowres/default/highres/veryhighres)
        imp_metric: importance metric (indoor for tabletop, outdoor for large scenes)
        rasterizer: GS rasterizer (radegs supports depth/normal rendering)
        iterations: training iterations (18000=fast, 30000=default)
        data_device: where to load images (cpu saves VRAM)
        config_path: MILo base config (./configs/fast or ./configs/quality)
    """
    source_dir = str(Path(source_dir).resolve())
    model_dir = str(Path(model_dir).resolve())

    cmd = [
        sys.executable, "train.py",
        "-s", source_dir,
        "-m", model_dir,
        "--imp_metric", imp_metric,
        "--rasterizer", rasterizer,
        "--mesh_config", mesh_config,
        "--data_device", data_device,
        "--iterations", str(iterations),
        "--config_path", config_path,
    ]

    print(f"Training MILo GS ({mesh_config}, {iterations} iters)")
    print(f"Source: {source_dir}")
    print(f"Output: {model_dir}")
    subprocess.run(cmd, cwd=str(MILO_DIR), env=MILO_ENV, check=True)

    # Verify output
    pc_path = Path(model_dir) / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    if not pc_path.exists():
        raise RuntimeError(f"Training failed — no point cloud at {pc_path}")

    print(f"Training complete. Point cloud at {pc_path}")


def extract_mesh(
    source_dir: str,
    model_dir: str,
    mesh_config: str = "default",
    imp_metric: str = "indoor",
    rasterizer: str = "radegs",
    refine_iter: int = 1000,
    remove_oof: bool = True,
):
    """Extract mesh from trained MILo model via learnable SDF + Marching Tetrahedra.

    Args:
        source_dir: path to undistorted COLMAP output (same as training)
        model_dir: path to trained model output
        mesh_config: must match the config used during training
        imp_metric: must match the metric used during training
        rasterizer: must match the rasterizer used during training
        refine_iter: SDF refinement iterations (default 1000, try 2000-3000 for better quality)
        remove_oof: remove vertices not visible from any training camera
    """
    source_dir = str(Path(source_dir).resolve())
    model_dir = str(Path(model_dir).resolve())

    cmd = [
        sys.executable, "mesh_extract_sdf.py",
        "-s", source_dir,
        "-m", model_dir,
        "--rasterizer", rasterizer,
        "--imp_metric", imp_metric,
        "--config", mesh_config,
        "--refine_iter", str(refine_iter),
    ]

    if remove_oof:
        cmd.append("--remove_oof_vertices")

    print(f"Extracting mesh (refine_iter={refine_iter})")
    subprocess.run(cmd, cwd=str(MILO_DIR), env=MILO_ENV, check=True)

    mesh_path = Path(model_dir) / "mesh_learnable_sdf.ply"
    if not mesh_path.exists():
        raise RuntimeError(f"Mesh extraction failed — no mesh at {mesh_path}")

    print(f"Mesh extracted at {mesh_path}")


def reconstruct_mesh(
    source_dir: str,
    model_dir: str,
    mesh_config: str = "default",
    imp_metric: str = "indoor",
    rasterizer: str = "radegs",
    iterations: int = 18000,
    refine_iter: int = 1000,
    data_device: str = "cpu",
    remove_oof: bool = True,
    config_path: str = "./configs/fast",
):
    """Full MILo pipeline: GS training → mesh extraction.

    This is the master entry point. Takes undistorted COLMAP output and
    produces a mesh PLY ready for USD conversion.
    """
    print("=" * 50)
    print("Stage 1: GS Training")
    print("=" * 50)
    train_gs(source_dir, model_dir, mesh_config, imp_metric, rasterizer, iterations, data_device, config_path)

    print()
    print("=" * 50)
    print("Stage 2: Mesh Extraction")
    print("=" * 50)
    extract_mesh(source_dir, model_dir, mesh_config, imp_metric, rasterizer, refine_iter, remove_oof)

    mesh_path = Path(model_dir) / "mesh_learnable_sdf.ply"
    print()
    print("=" * 50)
    print(f"Done. Mesh at: {mesh_path}")
    print("=" * 50)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "train_gs": train_gs,
        "extract_mesh": extract_mesh,
        "reconstruct": reconstruct_mesh,
    })
