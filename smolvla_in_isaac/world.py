import torch
import numpy as np
import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()

AppLauncher.add_app_launcher_args(parser)

args_cli, unknown_args = parser.parse_known_args()


args_cli.headless = True      
args_cli.livestream = 2       # 2 = WebRTC
args_cli.enable_cameras = True


sys.argv.append("--/app/livestream/publicEndpointAddress=100.115.105.111")
sys.argv.append("--/app/livestream/port=49100")


app_launcher = AppLauncher(args_cli)
sim_app = app_launcher.app


# from smolvla_in_isaac.robots.so_101_cfg import so101_cfg


from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationContext, SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import TiledCameraCfg, TiledCamera

# from utils.helpers import xyz_to_quat_isaac

from isaaclab.utils.math import quat_from_euler_xyz

import omni.kit.viewport.utility

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg


from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

sim_cfg = SimulationCfg(dt=0.01)
sim = SimulationContext(sim_cfg)

sim.set_camera_view((2.5, 2.5, 2.5), (0.0, 0.0, 0.0))

ground = sim_utils.GroundPlaneCfg()
ground.func("/World/GroundPlane", ground)


front_cam_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Robot/base_link/front_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1, 0, 0.1),
            # rot=tuple(quat_from_euler_xyz(torch.Tensor([90.0]), torch.Tensor([90.0]), torch.Tensor([0.0])).squeeze().tolist()),
            rot=(0.5, 0.5, 0.5, 0.5), 
            convention="opengl"
        ),  # wxyz
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=40.6,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )


gripper_cam_cfg: TiledCameraCfg = TiledCameraCfg(
    prim_path = "/World/Robot/gripper_link/gripper_camera",
    offset=TiledCameraCfg.OffsetCfg(
        pos = (0.005, 0.06, 0.02),
        rot = (0.0, 0.0, -0.24192, -0.9703),
        convention = "opengl"
    ),
    data_types = ["rgb"],
    spawn = sim_utils.PinholeCameraCfg(
        focal_length=40.6,
        focus_distance=400,
        horizontal_aperture = 38.11,
        clipping_range=(0.01, 50.0),
        lock_camera = True,
    ),
    width=640,
    height=480,
    update_period = 1 /30.0
)

so101_cfg = ArticulationCfg(
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

robot_cfg = so101_cfg

robot = Articulation(robot_cfg)

sim.reset()

print("[INFO] Warming up render graph (15 steps)...")
for i in range(50):
    sim.step()

print("[INFO] Initializing TiledCamera...")
front_camera = TiledCamera(front_cam_cfg)
gripper_camera = TiledCamera(gripper_cam_cfg)



# viewport_api = omni.kit.viewport.utility.get_active_viewport()

# print(viewport_api)

# if viewport_api:
#     print("[INFO] Setting Camera Viewport...")
#     viewport_api.set_active_camera("/World/Robot/base_link/front_camera")


joint_pos = robot.data.default_joint_pos.clone()
joint_vel = robot.data.default_joint_vel.clone()
robot.write_joint_state_to_sim(joint_pos, joint_vel)

print("Joint Names:", robot.joint_names)
print("Number of joints:", robot.num_joints)

# Keyboard control
keyboard = Se2Keyboard(Se2KeyboardCfg())

current_positions = robot.data.default_joint_pos.clone()


leader_config = SO101LeaderConfig(
    port="/dev/ttyACM0",
    id="blue"
)

# leader = SO101Leader(leader_config)

# leader.connect()

while sim_app.is_running():
    delta = keyboard.advance()

    # action = leader.get_action()

    # print("Action:", action)

    # print("delta:", delta)

    # delta is tuple (x, y, yaw)
    current_positions[0, 0] -= delta[1] * 0.05
    current_positions[0, 1] += delta[0] * 0.05

    # Apply positions to robot
    robot.set_joint_position_target(current_positions)
    robot.write_data_to_sim()

    sim.step()
    robot.update(sim_cfg.dt)

sim_app.close()