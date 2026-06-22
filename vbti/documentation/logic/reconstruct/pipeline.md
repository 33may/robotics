# Reconstruction Pipeline

## Full Flow

```text
video/photos
-> frame extraction
-> COLMAP/Nerfstudio SfM
-> undistortion
-> MILo Gaussian Splat training
-> MILo learnable SDF mesh extraction
-> mesh cleaning
-> USD conversion with physics
-> Isaac scene composition
-> LeIsaac/IsaacLab export
-> simulation data collection
-> HDF5 to LeRobot conversion
```

## Master Commands

```bash
python -m vbti.logic.reconstruct.master video_processing \
  --video_path data/scene.mp4 \
  --output_dir data/frames \
  --mode count \
  --value 200
```

```bash
python -m vbti.logic.reconstruct.master gs_reconstruction \
  --frames_dir data/frames \
  --output_dir data/recon \
  --matching_method exhaustive
```

```bash
python -m vbti.logic.reconstruct.master ply_to_usda \
  --mesh_path data/recon/milo/mesh_learnable_sdf.ply \
  --output_path data/scene.usda
```

```bash
python -m vbti.logic.reconstruct.master scene_composition \
  --scene_usda_path data/scene_composed.usda \
  --task_name vbti_so_v1
```

## Manual Checkpoints

The pipeline intentionally has manual checkpoints:

- choose/inspect extracted frames;
- validate COLMAP reconstruction;
- inspect MILo mesh;
- clean mesh if needed;
- compose scene manually in Isaac;
- verify camera rendering path, not only viewport;
- generate task code.

## Important Status

NuRec/GS USDZ can render in the viewport but fail/hang in robot-camera synthetic data. Use mesh-compatible USD assets for policy data.
