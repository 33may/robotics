# Robotics ML Portfolio

Robotics projects for SO-ARM100/101 robot arm: imitation learning, vision-language-action models, and Isaac Sim integration.

## Projects

### smolvla_in_isaac/
SmolVLA vision-language-action model running in NVIDIA Isaac Sim.
- Main: `world.py`
- Dual cameras (front + wrist)
- Natural language task control

### so101_lerobot/
SmolVLA model fine-tuning and dataset tools.
- `finetune_smolvla.ipynb` - Training
- `load_so101_dataset.ipynb` - Data prep

### diffusion/
Diffusion policy for imitation learning.
- `model_src/` - Models (diffusion, vision encoder)
- `imitation_learning.ipynb` - Training
- Supports RoboMimic, RoboSuite, Gym PushT

### nvda/
Isaac Sim projects.
- `collect_cubes/` - Cube collection task
- `isaac_so_arm101/` - SO-ARM101 setup

### robots/
Robot assets (USD, URDF, meshes).
- `SO-ARM100/` - SO-100 convex robot files
- `robo_car.usd`

### roarm-m3-assets/
RoArm-M3 meshes and conversion scripts.

### calibration/
SO-101 calibration tools.

## Structure

```
robotics/
├── smolvla_in_isaac/
├── so101_lerobot/
├── diffusion/
├── nvda/
├── robots/
├── roarm-m3-assets/
└── calibration/
```

## Hardware

SO-ARM100/101: 6-DOF arm + gripper
Joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper

## Dependencies

- Isaac Sim
- PyTorch
- lerobot (HuggingFace)

## Usage

SmolVLA in Isaac:
```bash
cd smolvla_in_isaac
python world.py
```
