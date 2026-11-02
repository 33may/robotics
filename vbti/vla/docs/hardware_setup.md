# Hardware Setup Reference

---

## Robot Arms

### SO-ARM101 Leader
- **Port:** `/dev/ttyACM2`
- **Purpose:** Teleoperation input (human demonstrates task)
- **Joints:** 6-DOF + gripper (7 joints total)

### SO-ARM101 Follower
- **Port:** `/dev/ttyACM1`
- **Purpose:** Executes actions (real-world deployment)

### Calibration
- Koen's calibration offsets are the known-good baseline
- Experimental recalibration overwrites must be restored from Koen's offsets
- Calibration data stored in `calibration/so101_leader/`

---

## Cameras

### Intel RealSense D405 (×4)
- **Resolution:** 640×480
- **Frame rate:** 15fps (USB 2.1 bandwidth limit for 4 cameras)
- **Alternative:** 2 cameras @30fps on USB 2.1
- **USB bandwidth:** Multiple USB 2.1 devices contend — needs `usbfs_memory_mb` tuning

### Camera Configuration in Simulation
| Camera | Mount | DR Jitter |
|--------|-------|-----------|
| Side camera | Fixed (scene) | ±25mm, ±3° |
| Table camera | Fixed (scene) | ±25mm, ±3° |
| Gripper camera | Robot-mounted | ±5mm, ±1° |

### Isaac Sim Camera Notes
- Scene cameras use `spawn=None` in LeIsaac (reference existing USD prims)
- Gripper camera must be spawned relative to gripper link (moves with robot)
- `--enable_cameras` flag required for camera rendering

---

## Compute

### Local Workstation
| Component | Specs |
|-----------|-------|
| GPU | NVIDIA RTX 4070 Ti SUPER (16GB VRAM, SM 89) |
| OS | Fedora 42 (Linux 6.18) |
| CUDA | 12.9 (patched for glibc 2.41) |
| GCC | 14 (system GCC 15 unsupported by CUDA) |
| Python | Miniconda, multiple envs |

### Key Conda Environments
| Env | Purpose | Key Deps |
|-----|---------|----------|
| `gsplat-pt25` | MILo, nerfstudio, 3D reconstruction | PyTorch 2.5.1+cu124, Python 3.11 |
| Isaac Sim env | Simulation, training, inference | Isaac Sim 5.0+, Isaac Lab |

### Cloud (RunPod)
| GPU | VRAM | $/hr | Use Case |
|-----|------|------|----------|
| A40 | 48GB | $0.76 | Cosmos Transfer 480p |
| A100 80GB | 80GB | $1.19-1.39 | Cosmos Transfer 720p |
| H100 80GB | 80GB | $1.99-2.69 | Fastest inference |

---

## Teleoperation Controls

```
B = Start recording
N = Mark success + reset
R = Discard + reset
```

### Run Teleop
```bash
python teleop_se3_agent.py \
  --task LeIsaac-SO101-VbtiMeshTable-v0 \
  --teleop_device so101leader \
  --enable_cameras
```

---

## USB Troubleshooting

- **4 cameras @30fps exceeds USB 2.1 bandwidth** — drop to 15fps or use USB 3.0 hub
- **usbfs_memory_mb**: May need to increase for multiple RealSense cameras
  ```bash
  echo 128 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
  ```
