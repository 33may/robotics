# Domain Randomization Configuration Reference

**Scene:** VBTI SO-ARM101 Mesh Table (`vbti_so_v1_env_cfg.py`)
**When:** Applied every environment reset

---

## Randomization Parameters

### Object Positions & Rotations

| Object | Parameter | Range | Notes |
|--------|-----------|-------|-------|
| Duck | Position X | ±0.21m | Relative to spawn |
| Duck | Position Y | ±0.32m | Relative to spawn |
| Duck | Rotation Yaw | ±180° | Full rotation |
| Cup | Position X | ±0.21m | Relative to spawn |
| Cup | Position Y | ±0.32m | Relative to spawn |
| Cup | Rotation | None | Symmetric object |

### Lighting

| Light | Parameter | Range | Notes |
|-------|-----------|-------|-------|
| DomeLight | Rotation Roll/Pitch | ±10° | HDRI orientation |
| DomeLight | Rotation Yaw | ±10° | HDRI orientation |
| DistantLight | Rotation Roll/Pitch | ±15° | Shadow direction |
| DistantLight | Rotation Yaw | ±30° | Shadow direction |
| DistantLight | Intensity | 200–4000 lux | Brightness variation |
| DistantLight | Angle (shadow softness) | 0–45° | Sharp to soft shadows |

### Camera Jitter

| Camera | Position | Rotation | Notes |
|--------|----------|----------|-------|
| Side camera | ±25mm | ±3° | External viewpoint |
| Table camera | ±25mm | ±3° | External viewpoint |
| Gripper camera | ±5mm | ±1° | Tighter bounds (mounted on arm) |

### Physics Properties

| Parameter | Range | Notes |
|-----------|-------|-------|
| Object mass | 0.8–1.2× scale | Multiplicative |
| Static friction | 0.3–1.0 | Per object |
| Dynamic friction | 0.2–0.8 | Per object |
| Gravity Z | ±0.3 m/s² | Around -9.81 |

---

## Implementation Notes

- All randomization happens at **reset time** (every episode start)
- Scale changes require **init-time** application — PhysX bakes geometry into collision pipeline at startup
- Object discovery is automatic via `parse_usd_and_create_subassets()` in the env config
- Camera jitter is applied additively to base positions extracted from scene USD
- Physics material randomization uses `PhysicsMaterialAPI` on collision meshes

## Key Files

- `vbti/data/ready_export_sov1/config/vbti_so_v1_env_cfg.py` — DR ranges defined here
- `vbti/data/so_v1/scene_config.json` — Base positions extracted from USDA
