import torch
import numpy as np

from isaaclab.app import AppLauncher

app_launcher = AppLauncher()
sim_app = app_launcher.app

from isaaclab.sim import SimulationContext, SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

sim_cfg = SimulationCfg(dt=0.01)
sim = SimulationContext(sim_cfg)

sim.set_camera_view((2.5, 2.5, 2.5), (0.0, 0.0, 0.0))

ground = sim_utils.GroundPlaneCfg()
ground.func("/World/GroundPlane", ground)

# joints: ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']

robot_cfg = ArticulationCfg(
    prim_path = "/World/Robot",
    spawn = sim_utils.UsdFileCfg(
        usd_path = "/home/may33/projects/ml_portfolio/robotics/robots/SO-ARM100/convex_fripper_robot.usd",
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity = False,
        ),
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions = True,
            fix_root_link = True,
        )
    ),
    init_state = ArticulationCfg.InitialStateCfg(
        pos = (0.0, 0.0, 0.0),
        joint_pos = {
            "shoulder_pan" : 0.0,
            "shoulder_lift" : 0.0,
            "elbow_flex" : 0.0,
            "wrist_flex" : 0.0,
            "wrist_roll" : 0.0,
            "gripper" : 0.0,
        }
    ),
    actuators = {
        "arm" : ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=100.0,
            damping=10.0,
        )
    }
)

robot = Articulation(robot_cfg)

sim.reset()

joint_pos = robot.data.default_joint_pos.clone()
joint_vel = robot.data.default_joint_vel.clone()
robot.write_joint_state_to_sim(joint_pos, joint_vel)

print("Joint Names:", robot.joint_names)
print("Number of joints:", robot.num_joints)

# Keyboard control
keyboard = Se2Keyboard(Se2KeyboardCfg())

current_positions = robot.data.default_joint_pos.clone()

print("Controls:")
print("  W/S - shoulder_lift up/down")
print("  A/D - shoulder_pan left/right")
print("  Q/E - elbow_flex")
print("  ESC - quit")

while sim_app.is_running():
    # Get keyboard input - returns (delta, command)
    delta = keyboard.advance()

    # print("delta:", delta)

    # delta is tuple (x, y, yaw)
    # WASD controls
    current_positions[0, 0] += delta[1] * 0.05  # shoulder_pan (W/S)
    current_positions[0, 1] += delta[0] * 0.05  # shoulder_lift (A/D)
    current_positions[0, 2] += delta[2] * 0.05  # elbow_flex (Q/E for yaw)

    # Apply positions to robot
    robot.set_joint_position_target(current_positions)
    robot.write_data_to_sim()

    sim.step()
    robot.update(sim_cfg.dt)

sim_app.close()