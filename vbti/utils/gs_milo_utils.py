"""
MILo Gaussian Splatting reconstruction utilities.

Wraps MILo train.py (GS training) and mesh_extract_sdf.py (mesh extraction).
Takes undistorted COLMAP output and produces a mesh PLY.

Usage:
    python vbti/utils/gs_milo_utils.py reconstruct --source_dir data/undistorted --model_dir data/milo_output
    python vbti/utils/gs_milo_utils.py train_gs --source_dir data/undistorted --model_dir data/milo_output
    python vbti/utils/gs_milo_utils.py extract_mesh --source_dir data/undistorted --model_dir data/milo_output
    python vbti/utils/gs_milo_utils.py create_config --output_dir data/my_scene/gs
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from loguru import logger

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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Reconstruction Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Each field: (default_value, description)
CONFIG_FIELDS = {
    # GS Training
    "mesh_config":  ("default", "mesh quality: verylowres/lowres/default/highres/veryhighres"),
    "imp_metric":   ("indoor",  "importance metric: indoor (tabletop) or outdoor (large scenes)"),
    "rasterizer":   ("radegs",  "GS rasterizer: radegs supports depth/normal rendering"),
    "iterations":   (18000,     "training iterations: 18000=fast, 30000=quality"),
    "data_device":  ("cpu",     "image loading device: cpu saves VRAM, cuda faster"),
    # Mesh Extraction
    "refine_iter":  (1000,      "SDF refinement iterations: 1000=fast, 2000-3000=better quality"),
    "remove_oof":   (True,      "remove vertices not visible from any training camera"),
}

DEFAULT_CONFIG = {k: v[0] for k, v in CONFIG_FIELDS.items()}


def save_config(config: dict, output_path: str):
    """Save reconstruction config to YAML."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.success(f"Config saved: {output_path}")


def load_config(config_path: str) -> dict:
    """Load reconstruction config from YAML, filling missing fields with defaults."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    merged = {**DEFAULT_CONFIG, **(config or {})}
    return merged


def _parse_value(raw: str, default):
    """Cast user input string to match the type of the default value."""
    if isinstance(default, bool):
        return raw.lower() in ("true", "1", "yes", "y")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def create_config(output_dir: str):
    """Interactive CLI to create a reconstruction config.

    Walks through each parameter, shows default and description.
    Press Enter to keep default, or type a new value.
    Saves recon_config.yaml to the specified directory.
    """
    logger.info("GS Reconstruction Config Generator")
    print()

    config = {}
    for field, (default, desc) in CONFIG_FIELDS.items():
        print(f"  {field} ({default})")
        print(f"    {desc}")
        raw = input(f"    > ").strip()
        if raw == "":
            config[field] = default
        else:
            config[field] = _parse_value(raw, default)
        print()

    output_path = Path(output_dir).resolve() / "recon_config.yaml"
    save_config(config, str(output_path))
    return str(output_path)


def _make_milo_config(iterations: int) -> str:
    """Write a temporary MILo config file with the given iterations.

    MILo's read_config() overrides CLI args with config file values,
    so we generate a config that matches what the user asked for.
    """
    config = (
        f"Namespace(iterations={iterations}, densify_until_iter=3_000, "
        f"aggressive_clone_from_iter=500, aggressive_clone_interval=250, "
        f"warn_until_iter=3_000, depth_reinit_iter=2_000, "
        f"simp_iteration1=3_000, simp_iteration2=8_000)\n"
    )
    f = tempfile.NamedTemporaryFile(mode="w", suffix="_milo_cfg", delete=False)
    f.write(config)
    f.close()
    return f.name


def train_gs(
    source_dir: str,
    model_dir: str,
    mesh_config: str = "default",
    imp_metric: str = "indoor",
    rasterizer: str = "radegs",
    iterations: int = 18000,
    data_device: str = "cpu",
):
    """Train MILo Gaussian Splatting model.

    Args:
        source_dir: path to undistorted COLMAP output (PINHOLE cameras)
        model_dir: where to save the trained model
        mesh_config: mesh quality preset (verylowres/lowres/default/highres/veryhighres)
        imp_metric: importance metric (indoor for tabletop, outdoor for large scenes)
        rasterizer: GS rasterizer (radegs supports depth/normal rendering)
        iterations: training iterations (18000=fast, 30000=quality)
        data_device: where to load images (cpu saves VRAM)
    """
    source_dir = str(Path(source_dir).resolve())
    model_dir = str(Path(model_dir).resolve())
    config_path = _make_milo_config(iterations)

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

    logger.info(f"Training MILo GS ({mesh_config}, {iterations} iters)")
    logger.debug(f"Source: {source_dir}")
    logger.debug(f"Output: {model_dir}")
    subprocess.run(cmd, cwd=str(MILO_DIR), env=MILO_ENV, check=True)

    # Verify output — find the last saved iteration (config may override iterations arg)
    pc_dir = Path(model_dir) / "point_cloud"
    if not pc_dir.exists():
        raise RuntimeError(f"Training failed — no point_cloud directory at {pc_dir}")
    last_iter = sorted(pc_dir.iterdir())[-1]
    pc_path = last_iter / "point_cloud.ply"
    if not pc_path.exists():
        raise RuntimeError(f"Training failed — no point cloud at {pc_path}")

    logger.success(f"Training complete. Point cloud at {pc_path}")


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

    logger.info(f"Extracting mesh (refine_iter={refine_iter})")
    subprocess.run(cmd, cwd=str(MILO_DIR), env=MILO_ENV, check=True)

    mesh_path = Path(model_dir) / "mesh_learnable_sdf.ply"
    if not mesh_path.exists():
        raise RuntimeError(f"Mesh extraction failed — no mesh at {mesh_path}")

    logger.success(f"Mesh extracted at {mesh_path}")


def reconstruct_mesh(
    source_dir: str,
    model_dir: str,
    config_path: str = None,
    mesh_config: str = "default",
    imp_metric: str = "indoor",
    rasterizer: str = "radegs",
    iterations: int = 18000,
    refine_iter: int = 1000,
    data_device: str = "cpu",
    remove_oof: bool = True,
):
    """Full MILo pipeline: GS training → mesh extraction.

    This is the master entry point. Takes undistorted COLMAP output and
    produces a mesh PLY ready for USD conversion.

    Args:
        config_path: path to recon_config.yaml — overrides individual params.
    """
    if config_path:
        cfg = load_config(config_path)
        mesh_config = cfg["mesh_config"]
        imp_metric = cfg["imp_metric"]
        rasterizer = cfg["rasterizer"]
        iterations = cfg["iterations"]
        refine_iter = cfg["refine_iter"]
        data_device = cfg["data_device"]
        remove_oof = cfg["remove_oof"]
        logger.info(f"Loaded config: {config_path}")

    logger.info("Stage 1: GS Training")
    train_gs(source_dir, model_dir, mesh_config, imp_metric, rasterizer, iterations, data_device)

    logger.info("Stage 2: Mesh Extraction")
    extract_mesh(source_dir, model_dir, mesh_config, imp_metric, rasterizer, refine_iter, remove_oof)

    mesh_path = Path(model_dir) / "mesh_learnable_sdf.ply"
    logger.success(f"Done. Mesh at: {mesh_path}")


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "train_gs": train_gs,
        "extract_mesh": extract_mesh,
        "reconstruct": reconstruct_mesh,
        "create_config": create_config,
    })
