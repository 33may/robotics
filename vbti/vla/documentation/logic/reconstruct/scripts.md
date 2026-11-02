# Reconstruction Scripts

## `scripts/3d/reconstruct_mesh.py`

Gaussian Splat/object PLY to Poisson GLB:

```bash
python /home/may33/projects/ml_portfolio/robotics/vbti/scripts/3d/reconstruct_mesh.py \
  --input vbti/data/so_v1/assets/duck/object_0.ply \
  --output vbti/data/so_v1/assets/duck/object_0_reconstructed.glb \
  --depth 9 \
  --estimate-normals
```

## `scripts/3d/fix_manifold.py`

Repair GLB for PhysX/deformables:

```bash
python /home/may33/projects/ml_portfolio/robotics/vbti/scripts/3d/fix_manifold.py \
  --input vbti/data/so_v1/assets/duck/object_0.glb \
  --output vbti/data/so_v1/assets/duck/object_0_fixed.glb
```

## Isaac Script Editor Utilities

Run these inside Isaac Sim Script Editor, not normal Python:

- `scripts/3d/create_deformable_cube.py`
- `scripts/3d/convert_duck_deformable.py`
- `scripts/3d/convert_duck_surface_deformable.py`
- `scripts/3d/dump_soft_cube_schema.py`

Surface deformable is preferred for the hollow duck because volumetric deformable fills the interior and can cause ghost collision.

## `scripts/3d/tune_deformable.py`

Shows/tunes deformable USDA parameters. Current `__main__` only calls `show()`. Edit/import `tune()` to modify.

## `scripts/3d/manifold_check.py`

Hardcoded quick check, not a general CLI. Read/edit paths before using.

## `scripts/sim/teleop_playground.py`

Run with IsaacLab:

```bash
isaaclab -p vbti/scripts/sim/teleop_playground.py
isaaclab -p vbti/scripts/sim/teleop_playground.py --port /dev/ttyACM1
isaaclab -p vbti/scripts/sim/teleop_playground.py --recalibrate
```

Purpose:

- minimal IsaacLab playground;
- physical leader arm controls simulated SO-101;
- hardcoded scene/asset paths;
- calibration cache under `scripts/sim/.cache/`.

Not a general reusable environment without editing.
