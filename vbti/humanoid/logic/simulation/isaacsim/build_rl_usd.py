"""build_rl_usd.py — import the RL TRAINING urdf (HU_D04_01_rl.urdf) to a USD.

The policy was trained in IsaacLab/IsaacGym on this exact asset (serial PR ankle, no
achilles — confirmed by LimX's parallel_joint_mapping doc + the `_rl` URDF). This builds
`assets/oli/usd/HU_D04_01_rl.usd` so `Oli` loads the *training* model instead of the
shipped deploy USD. One-time:

    conda run -n isaac python humanoid/logic/simulation/isaacsim/build_rl_usd.py
"""

from __future__ import annotations

from pathlib import Path

_HUM = Path(__file__).resolve().parents[3]
_URDF = (_HUM / "vendor" / "oli-main-software-2.2.12" / "install" / "etc"
         / "HU_D04_description" / "urdf" / "HU_D04_01_rl.urdf")
_DEST = _HUM / "assets" / "oli" / "usd" / "HU_D04_01_rl.usd"

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import omni.kit.commands  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402

print(f"[build] URDF: {_URDF}\n[build] DEST: {_DEST}", flush=True)
assert _URDF.exists(), f"URDF not found: {_URDF}"

status, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
cfg.set_fix_base(False)               # free base (we pin at load time if needed)
cfg.set_merge_fixed_joints(True)      # fold fixed hand/contact joints → 31 revolute DOF
cfg.set_self_collision(False)
cfg.set_import_inertia_tensor(True)   # use the URDF inertias (the trained model's)
cfg.set_make_default_prim(True)
cfg.set_create_physics_scene(False)   # the World provides the physics scene
cfg.set_density(0.0)
cfg.set_default_drive_type(1)         # 1 = position drive (so set_gains/targets work)
cfg.set_default_drive_strength(0.0)   # gains overridden per-command at runtime
cfg.set_default_position_drive_damping(0.0)

status, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile", urdf_path=str(_URDF), import_config=cfg,
    dest_path=str(_DEST), get_articulation_root=False)
print(f"[build] import status={status} prim={prim_path}", flush=True)

# sanity: open the produced USD and report the articulation DOF + names
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 1000.0)
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
add_reference_to_stage(usd_path=str(_DEST), prim_path="/World/OliRL")
world.reset()
art = SingleArticulation(prim_path="/World/OliRL")
art.initialize()
print(f"[build] num_dof={art.num_dof}", flush=True)
print(f"[build] dof_names={list(art.dof_names)}", flush=True)
print(f"[build] DONE -> {_DEST} (exists={_DEST.exists()})", flush=True)
app.close()
