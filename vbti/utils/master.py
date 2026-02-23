"""
VBTI Pipeline Orchestrator

Constant structure — module internals change, this file stays the same.
Each phase calls one entry function from its module. Phases run sequentially
with manual checkpoints between them for human verification.

Usage:
    python vbti/utils/master.py video_processing --video_path scene.mp4 --output_dir data/frames
    python vbti/utils/master.py gs_reconstruction --frames_dir data/frames --output_dir data/recon
    python vbti/utils/master.py scene_composition  (TODO — Isaac Sim GUI + isaaclab_cfg_utils)
    python vbti/utils/master.py data_collection    (TODO — teleop + cosmos + format conversion)
"""

import sys
from pathlib import Path

# Allow running from any directory (e.g., python utils/master.py from vbti/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.video_utils import extract_frames
from utils.colmap_utils import process_colmap
from utils.gs_milo_utils import reconstruct_mesh
from utils.format_utils import mesh_to_usd


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: Video → Stable Frames
# Module: video_utils.py
# Input:  video.mp4
# Output: frames/*.png
# Verify: visually check extracted frames
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def video_processing(
    video_path: str,
    output_dir: str,
    mode: str = "count",
    value: float = 200,
):
    extract_frames(video_path, output_dir, mode, value)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: Frames → COLMAP → MILo GS → Mesh → USDA
# Modules: colmap_utils.py, gs_milo_utils.py, format_utils.py
# Step 2a: COLMAP reconstruction + undistort → PINHOLE cameras
# Step 2b: MILo GS training → mesh extraction
# Step 2c: Convert mesh PLY → scene USDA with physics
# Input:  frames/*.png
# Output: scene.usda (mesh with collision + physics material)
# Verify: check reconstruction quality, mesh holes, USDA in Isaac
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def gs_reconstruction(
    frames_dir: str,
    output_dir: str,
    matching_method: str = "exhaustive",
    mesh_config: str = "default",
    iterations: int = 18000,
    refine_iter: int = 1000,
):
    from pathlib import Path

    output_path = Path(output_dir).resolve()
    # process_colmap creates: output_dir/colmap/ and output_dir/undistorted/
    undistorted_dir = str(output_path / "undistorted")
    milo_dir = str(output_path / "milo")

    # Step 2a: COLMAP → validate → undistort
    print("=" * 50)
    print("Phase 2a: COLMAP reconstruction")
    print("=" * 50)
    process_colmap(frames_dir, str(output_path), matching_method)

    # CHECKPOINT: verify COLMAP reconstruction before MILo
    print("\n>>> CHECKPOINT: verify COLMAP output, then re-run to continue <<<")
    print(f"    Check: {undistorted_dir}/images/")

    # Step 2b: MILo GS training + mesh extraction
    print()
    print("=" * 50)
    print("Phase 2b: MILo GS → mesh")
    print("=" * 50)
    reconstruct_mesh(
        source_dir=undistorted_dir,
        model_dir=milo_dir,
        mesh_config=mesh_config,
        iterations=iterations,
        refine_iter=refine_iter,
    )

    # Step 2c: Mesh PLY → USDA
    mesh_ply = str(Path(milo_dir) / "mesh_learnable_sdf.ply")
    scene_usda = str(output_path / "scene.usda")

    print()
    print("=" * 50)
    print("Phase 2c: Mesh → USDA")
    print("=" * 50)
    mesh_to_usd(mesh_path=mesh_ply, output_path=scene_usda)

    print()
    print("=" * 50)
    print(f"Done. Scene at: {scene_usda}")
    print("Open in Isaac Sim to verify mesh quality + physics.")
    print("=" * 50)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: Scene Composition in Isaac Sim
# Module: isaaclab_cfg_utils.py (exists), robot_utils.py (exists)
# Step 3a: Load scene USDA + object assets in Isaac Sim GUI
# Step 3b: Place robot, cameras, verify scales & physics
# Step 3c: isaaclab_cfg_utils.pipeline() → leisaac task config
# Input:  scene.usda + object assets (from web interface)
# Output: leisaac task config (code)
# Verify: run in Isaac, check physics & rendering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def scene_composition():
    # Done in Isaac Sim GUI, then:
    # python isaaclab_cfg_utils.py pipeline scene.usda task_name
    print("Scene composition is done in Isaac Sim GUI.")
    print("After composing, run:")
    print("  python vbti/utils/isaaclab_cfg_utils.py pipeline <scene.usda> <task_name>")
    pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 4: Data Collection & Training
# Module: data_utils.py (TODO)
# Step 4a: Collect teleop demonstrations in sim
# Step 4b: (Optional) Augment with Cosmos Transfer
# Step 4c: Convert Isaac HDF5 → LeRobot format
# Input:  leisaac task config
# Output: HuggingFace dataset
# Verify: replay demos, check dataset stats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def data_collection():
    # TODO: data collection script
    # TODO: cosmos_transfer.py — already exists
    # TODO: format conversion — already exists
    pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "video_processing": video_processing,
        "gs_reconstruction": gs_reconstruction,
        "scene_composition": scene_composition,
        "data_collection": data_collection,
    })
