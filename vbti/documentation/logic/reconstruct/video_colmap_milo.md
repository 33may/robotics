# Video, COLMAP, And MILo

## Video Utilities

```bash
python -m vbti.logic.reconstruct.video_utils stats --video_path scene.mp4
python -m vbti.logic.reconstruct.video_utils count --video_path scene.mp4 --percentage 0.7
python -m vbti.logic.reconstruct.video_utils fix_rotation --video_path scene.MOV
python -m vbti.logic.reconstruct.video_utils extract --video_path scene.mp4 --output_dir data/frames --mode count --value 200
python -m vbti.logic.reconstruct.video_utils extract --video_path scene.mp4 --output_dir data/frames --mode every --value 2
python -m vbti.logic.reconstruct.video_utils extract --video_path scene.mp4 --output_dir data/frames --mode percentage --value 0.7
```

Outputs frames under:

```text
output_dir/<video_stem>/*.png
```

Dependencies:

- OpenCV;
- `ffmpeg`;
- `sharp_frame_extractor`.

## COLMAP

```bash
python -m vbti.logic.reconstruct.colmap_utils run_colmap --frames_dir data/frames --output_dir data/colmap --matching_method exhaustive
python -m vbti.logic.reconstruct.colmap_utils validate_models --colmap_dir data/colmap
python -m vbti.logic.reconstruct.colmap_utils undistort --colmap_dir data/colmap --output_dir data/undistorted
python -m vbti.logic.reconstruct.colmap_utils reconstruct --frames_dir data/frames --output_dir data/recon --matching_method exhaustive
```

Outputs:

```text
output_dir/colmap/
output_dir/undistorted/images/
output_dir/undistorted/sparse/0/
```

For small frame sets, `exhaustive` matching is usually appropriate.

## MILo

```bash
python -m vbti.logic.reconstruct.gs_milo_utils create_config --output_dir data/recon/gs
python -m vbti.logic.reconstruct.gs_milo_utils train_gs --source_dir data/recon/undistorted --model_dir data/recon/milo
python -m vbti.logic.reconstruct.gs_milo_utils extract_mesh --source_dir data/recon/undistorted --model_dir data/recon/milo
python -m vbti.logic.reconstruct.gs_milo_utils reconstruct --source_dir data/recon/undistorted --model_dir data/recon/milo
```

Default config:

```yaml
mesh_config: default
imp_metric: indoor
rasterizer: radegs
iterations: 18000
data_device: cpu
resolution: 4
refine_iter: 1000
remove_oof: true
```

Outputs:

```text
model_dir/point_cloud/iteration_*/point_cloud.ply
model_dir/mesh_learnable_sdf.ply
```

Environment assumptions:

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export CC=/usr/bin/gcc-14
export CXX=/usr/bin/g++-14
export TORCH_CUDA_ARCH_LIST=8.9
```

Do not set `CPLUS_INCLUDE_PATH=/usr/include`; MILo/nvdiffrast JIT include ordering can break.
