"""
VBTI Pipeline Orchestrator

Constant structure — module internals change, this file stays the same.
Each phase calls one entry function from its module. Phases run sequentially
with manual checkpoints between them for human verification.

Usage:
    python vbti/utils/master.py video_processing  --video_path scene.mp4 --output_dir data/frames
    python vbti/utils/master.py gs_reconstruction --frames_dir data/frames --output_dir data/recon
    python vbti/utils/master.py gs_reconstruction --frames_dir data/frames --output_dir data/recon --config_path data/recon/gs/recon_config.yaml
    python vbti/utils/master.py ply_to_usda       --mesh_path data/recon/milo/mesh.ply --output_path data/recon/scene.usda
    python vbti/utils/master.py scene_composition  --scene_usda_path data/scene/scene.usda --task_name vbti_so_v1
    python vbti/utils/master.py scene_composition  --scene_usda_path data/scene/scene.usda --task_name vbti_so_v1 --robot_usd_path /path/to/custom_robot.usd
    python vbti/utils/master.py data_collection    (TODO)
"""

import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.video_utils import extract_frames
from utils.colmap_utils import process_colmap
from utils.gs_milo_utils import reconstruct_mesh
from utils.format_utils import mesh_to_usd
from vbti.utils.isaac_cfg_utils import pipeline as isaaclab_pipeline
from vbti.utils.isaac_cfg_utils import generate_isaaclab_env


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: Video → Stable Frames
# Module: video_utils.py
# Input:  video.mp4
# Output: frames/*.png
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def video_processing(
    video_path: str,
    output_dir: str,
    mode: str = "count",
    value: float = 200,
):
    extract_frames(video_path, output_dir, mode, value)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2a: Frames → COLMAP → MILo GS → Mesh
# Modules: colmap_utils.py, gs_milo_utils.py
# Input:  frames/*.png (or config_path for GS params)
# Output: mesh PLY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def gs_reconstruction(
    frames_dir: str,
    output_dir: str,
    config_path: str = None,
    matching_method: str = "exhaustive",
):
    output_path = Path(output_dir).resolve()
    undistorted_dir = str(output_path / "undistorted")
    milo_dir = str(output_path / "milo")

    logger.info("━━━ COLMAP reconstruction ━━━")
    process_colmap(frames_dir, str(output_path), matching_method)

    logger.warning(f"CHECKPOINT: verify COLMAP output at {undistorted_dir}/images/")

    logger.info("━━━ MILo GS -> mesh ━━━")
    reconstruct_mesh(
        source_dir=undistorted_dir,
        model_dir=milo_dir,
        config_path=config_path,
    )

    mesh_ply = Path(milo_dir) / "mesh_learnable_sdf.ply"
    logger.success(f"Done. Mesh at: {mesh_ply}")
    logger.info(f"Next: python vbti/utils/master.py ply_to_usda --mesh_path {mesh_ply} --output_path {output_path / 'scene.usda'}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2b: Mesh PLY → Scene USDA
# Module: format_utils.py
# Input:  mesh PLY
# Output: scene.usda (mesh + collision + physics)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ply_to_usda(
    mesh_path: str,
    output_path: str,
    static_friction: float = 0.7,
    dynamic_friction: float = 0.5,
    restitution: float = 0.1,
    apply_colmap_transform: bool = True,
    apply_srgb_conversion: bool = True,
):
    logger.info("━━━ Mesh -> USDA ━━━")
    mesh_to_usd(
        mesh_path=mesh_path,
        output_path=output_path,
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
        apply_colmap_transform=apply_colmap_transform,
        apply_srgb_conversion=apply_srgb_conversion,
    )
    logger.success(f"Scene at: {output_path}")
    logger.info("Open in Isaac Sim to verify mesh quality + physics.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: Scene USDA → LeIsaac Task Config
# Module: isaaclab_cfg_utils.py
# Input:  composed scene.usda (with robot, cameras, lights)
# Output: leisaac task (scene asset, env cfg, gym registration)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def scene_composition(
    scene_usda_path: str,
    task_name: str,
    robot_prim_path: str = "/World/so101_simready_follower_leisaac",
    robot_usd_path: str = None,
    cosmos_sensors: bool = False,
):
    """Generate leisaac task from a composed Isaac Sim scene.

    Compose your scene in Isaac Sim GUI first (place robot, cameras, lights,
    objects), save as USDA, then run this to generate the leisaac config.

    Args:
        scene_usda_path: Path to the composed scene USDA (with robot).
        task_name: Snake_case task name, e.g. 'vbti_so_v1'.
        robot_prim_path: Prim path of the robot in the scene.
        robot_usd_path: Path to custom robot USD. If None, uses default leisaac asset.
        cosmos_sensors: Enable depth + segmentation capture for Cosmos Transfer.
    """
    logger.info("━━━ Scene → LeIsaac Task ━━━")
    isaaclab_pipeline(
        scene_usda_path=scene_usda_path,
        task_name=task_name,
        robot_prim_path=robot_prim_path,
        robot_usd_path=robot_usd_path,
        cosmos_sensors=cosmos_sensors,
    )
    pascal = "".join(w.capitalize() for w in task_name.split("_"))
    logger.success(f"Task ready: LeIsaac-SO101-{pascal}-v0")
    logger.info(f"Next: teleop with --task LeIsaac-SO101-{pascal}-v0")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3b: Scene Config → Standalone IsaacLab Task
# Module: isaaclab_cfg_utils.py
# Input:  scene_config.json (generated by scene_composition)
# Output: standalone task folder (no leisaac deps)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def export_isaaclab_task(
    scene_config_path: str,
    task_name: str,
    output_dir: str = None,
):
    """Export a standalone IsaacLab task from scene config.

    Reads the scene_config.json (source of truth) and generates
    a self-contained task folder with no leisaac dependencies.

    Args:
        scene_config_path: Path to scene_config.json.
        task_name: Snake_case task name.
        output_dir: Output directory. Defaults to data/{task_name}_isaaclab/.
    """
    from pathlib import Path

    if output_dir is None:
        output_dir = str(Path(scene_config_path).parent / f"{task_name}_isaaclab")

    logger.info("━━━ Scene Config → IsaacLab Task ━━━")
    out = generate_isaaclab_env(scene_config_path, task_name, output_dir=output_dir)
    pascal = "".join(w.capitalize() for w in task_name.split("_"))
    logger.success(f"Task ready: Custom-SO101-{pascal}-v0")
    logger.info(f"Output: {out}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 4: Data Collection & Training
# Module: data_utils.py (TODO)
# Input:  leisaac task config
# Output: HuggingFace dataset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def data_collection():
    pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "video_processing":  video_processing,
        "gs_reconstruction": gs_reconstruction,
        "ply_to_usda":       ply_to_usda,
        "scene_composition":   scene_composition,
        "export_isaaclab":     export_isaaclab_task,
        "data_collection":     data_collection,
    })
