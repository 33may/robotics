"""
Minimal teleop playground for testing physics assets.

No leisaac env framework. No DR. No observations. No multi-env.
Just: scene + robot + leader arm + physics loop.

Edit the scene setup directly in this file, relaunch to test.

Usage:
    isaaclab -p vbti/scripts/sim/teleop_playground.py
    isaaclab -p vbti/scripts/sim/teleop_playground.py --port /dev/ttyACM1
    isaaclab -p vbti/scripts/sim/teleop_playground.py --recalibrate
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Minimal teleop playground")
parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Leader arm serial port")
parser.add_argument("--recalibrate", action="store_true", help="Recalibrate leader arm")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, DeformableObject, DeformableObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg


# ══════════════════════════════════════════════════════════════════════════════
# EDIT THIS SECTION — define your scene
# ══════════════════════════════════════════════════════════════════════════════

ROBOT_USD = "/home/may33/projects/ml_portfolio/robotics/robots/so101_leisaac_vbti/so101_simready_follower_leisaac.usd"
SCENE_USD = "/home/may33/projects/ml_portfolio/robotics/vbti/data/so_v1/scene/ready_scene_rigid_hdri_v1_no_robot_no_duck.usda"
# SOFT_DUCK_USD = "/home/may33/projects/ml_portfolio/robotics/vbti/data/so_v1/assets/duck/object_1_soft.usda"
SCENE_WITH_DUCK_USD = "/home/may33/projects/ml_portfolio/robotics/vbti/data/so_v1/scene/ready_scene_rigid_hdri_v1_no_robot.usda"

ROBOT_POS = (0.04386, 0.6008, 0.01133)
ROBOT_ROT = (0.7071, 0.0, 0.0, 0.7071)
DUCK_POS = (0.37, 0.76, 0.12)

# ══════════════════════════════════════════════════════════════════════════════


# Joint conversion: leader normalized [-100,100] → sim radians
# Same math as leisaac's convert_action_from_so101_leader
USD_JOINT_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]


def leader_to_radians(joint_state: dict, motor_limits: dict) -> list[float]:
    """Convert leader arm normalized positions to sim joint radians."""
    result = [0.0] * 6
    for i, name in enumerate(JOINT_NAMES):
        motor_lo, motor_hi = motor_limits[name]
        joint_lo, joint_hi = USD_JOINT_LIMITS_DEG[name]
        motor_range = motor_hi - motor_lo
        joint_range = joint_hi - joint_lo
        motor_deg = joint_state[name] - motor_lo
        joint_deg = motor_deg / motor_range * joint_range + joint_lo
        result[i] = joint_deg * math.pi / 180.0
    return result


def design_scene():
    """Build the scene: table + robot + soft duck."""

    # Ground plane
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())

    # Lights
    dome = sim_utils.DomeLightCfg(
        intensity=1000.0,
        texture_file="/home/may33/projects/ml_portfolio/robotics/vbti/data/so_v1/hdri/office_pano_4k.hdr",
        texture_format="latlong",
    )
    dome.func("/World/domeLight", dome)
    distant = sim_utils.DistantLightCfg(intensity=2000.0, angle=13.4)
    distant.func("/World/defaultLight", distant)

    # Scene WITH rigid duck (the original working scene)
    scene_cfg = sim_utils.UsdFileCfg(usd_path=SCENE_WITH_DUCK_USD)
    scene_cfg.func("/World/Scene", scene_cfg)

    # Robot
    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_POS,
            rot=ROBOT_ROT,
            joint_pos={
                "shoulder_pan": 0.0,
                "shoulder_lift": -1.745,
                "elbow_flex": 1.55,
                "wrist_flex": 0.873,
                "wrist_roll": 0.0,
                "gripper": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                effort_limit_sim=10, velocity_limit_sim=10,
                stiffness=17.8, damping=0.60,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper"],
                effort_limit_sim=10, velocity_limit_sim=10,
                stiffness=17.8, damping=0.60,
            ),
        },
    )
    robot = Articulation(cfg=robot_cfg)

    # No deformable objects — testing rigid duck from the scene USD
    return robot


def connect_leader(port: str, recalibrate: bool):
    """Connect to SO101 leader arm, return (bus, motor_limits)."""
    from leisaac.devices.lerobot.common.motors import (
        FeetechMotorsBus, Motor, MotorCalibration, MotorNormMode, OperatingMode,
    )
    import json, os

    CALIB_PATH = os.path.join(os.path.dirname(__file__), ".cache", "teleop_playground_leader.json")
    os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)

    motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }

    if recalibrate or not os.path.exists(CALIB_PATH):
        print("\n=== Leader Arm Calibration ===")
        bus = FeetechMotorsBus(port=port, motors=motors)
        bus.connect()
        bus.disable_torque()
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input("Move leader to MIDDLE of range, press ENTER...")
        homing_offset = bus.set_half_turn_homings()
        print("Move all joints through full range. Press ENTER when done...")
        range_mins, range_maxes = bus.record_ranges_of_motion()

        calibration = {}
        for motor_name, m in bus.motors.items():
            calibration[motor_name] = MotorCalibration(
                id=m.id, drive_mode=0,
                homing_offset=homing_offset[motor_name],
                range_min=range_mins[motor_name],
                range_max=range_maxes[motor_name],
            )
        bus.write_calibration(calibration)

        # Save
        save_data = {k: {"id": v.id, "drive_mode": v.drive_mode,
                         "homing_offset": v.homing_offset,
                         "range_min": v.range_min, "range_max": v.range_max}
                     for k, v in calibration.items()}
        with open(CALIB_PATH, "w") as f:
            json.dump(save_data, f, indent=4)
        print(f"Calibration saved: {CALIB_PATH}")
        bus.disconnect()

    # Load calibration and connect
    with open(CALIB_PATH) as f:
        calib_data = json.load(f)
    calibration = {
        k: MotorCalibration(id=int(v["id"]), drive_mode=int(v["drive_mode"]),
                            homing_offset=int(v["homing_offset"]),
                            range_min=int(v["range_min"]),
                            range_max=int(v["range_max"]))
        for k, v in calib_data.items()
    }

    bus = FeetechMotorsBus(port=port, motors=motors, calibration=calibration)
    bus.connect()
    bus.disable_torque()
    for motor in bus.motors:
        bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    # Motor limits from leisaac
    motor_limits = {
        "shoulder_pan": (-100.0, 100.0),
        "shoulder_lift": (-100.0, 100.0),
        "elbow_flex": (-100.0, 100.0),
        "wrist_flex": (-100.0, 100.0),
        "wrist_roll": (-100.0, 100.0),
        "gripper": (0.0, 100.0),
    }

    print("Leader arm connected.")
    return bus, motor_limits


def main():
    # Sim config
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args.device)
    sim_cfg.physx.solve_articulation_contact_last = True
    # Match NVIDIA FrankaDeformable config exactly
    sim_cfg.physx.solver_position_iteration_count = 8
    sim_cfg.physx.solver_velocity_iteration_count = 0
    sim_cfg.physx.contact_offset = 0.02
    sim_cfg.physx.rest_offset = 0.001
    sim_cfg.physx.bounce_threshold_velocity = 0.2
    sim_cfg.physx.friction_offset_threshold = 0.04
    sim_cfg.physx.friction_correlation_distance = 0.025
    sim_cfg.physx.enable_stabilization = True
    sim_cfg.physx.max_depenetration_velocity = 1000.0
    sim_cfg.physx.gpu_max_soft_body_contacts = 4 * 1024 * 1024
    sim_cfg.physx.gpu_max_particle_contacts = 1024 * 1024
    sim_cfg.physx.gpu_heap_capacity = 32 * 1024 * 1024
    sim_cfg.physx.gpu_temp_buffer_capacity = 16 * 1024 * 1024
    sim_cfg.physx.gpu_found_lost_pairs_capacity = 1024 * 1024
    sim_cfg.physx.gpu_max_rigid_contact_count = 524288
    sim_cfg.physx.gpu_max_rigid_patch_count = 32 * 1024 * 1024
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.1, 1.4, 0.6], target=[0.35, 0.65, 0.1])

    # Build scene
    robot = design_scene()

    # Reset sim
    sim.reset()

    # Get joint indices for the robot
    joint_ids = []
    for name in JOINT_NAMES:
        idx = robot.find_joints(name)[0][0]
        joint_ids.append(idx)
    print(f"Joint indices: {dict(zip(JOINT_NAMES, joint_ids))}")

    # Connect leader arm
    bus, motor_limits = connect_leader(args.port, args.recalibrate)

    print("\n" + "=" * 60)
    print("TELEOP PLAYGROUND")
    print("=" * 60)
    print("Move the leader arm to control the sim robot.")
    print("Ctrl+C to stop.")
    print("=" * 60 + "\n")

    sim_dt = sim.get_physics_dt()
    count = 0

    try:
        while simulation_app.is_running():
            # Read leader
            joint_state = bus.sync_read("Present_Position")

            # Convert to radians
            target_rad = leader_to_radians(joint_state, motor_limits)
            target_tensor = torch.tensor([target_rad], device=sim.device, dtype=torch.float32)

            # Apply to robot
            robot.set_joint_position_target(target_tensor, joint_ids=joint_ids)
            robot.write_data_to_sim()

            # Step
            sim.step()
            count += 1

            # Update assets
            robot.update(sim_dt)

            # Periodic status
            if count % 600 == 0:
                pos = [f"{joint_state[n]:6.1f}" for n in JOINT_NAMES]
                # print(f"  [{count:6d}] leader: {' '.join(pos)}")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        bus.disconnect()
        print("Leader disconnected.")


if __name__ == "__main__":
    main()
    simulation_app.close()
