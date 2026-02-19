import os
import warnings

# Suppress Isaac Sim warnings
os.environ["ISAAC_SUPPRESS_WARNINGS"] = "1"
os.environ["CARB_LOGGING_MAX_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")


import torch
import numpy as np
import argparse
import sys

from isaaclab.app import AppLauncher

# =======================================================
# create isaac app
# =============================================================

parser = argparse.ArgumentParser()

AppLauncher.add_app_launcher_args(parser)

args_cli, unknown_args = parser.parse_known_args()


# -----------------------------
# setup streaming and cameras

args_cli.headless = True
args_cli.livestream = 2       # 2 = WebRTC
args_cli.enable_cameras = True

sys.argv.append("--/app/livestream/publicEndpointAddress=100.115.105.111")
sys.argv.append("--/app/livestream/port=49100")


print(f"[DEBUG] enable_cameras = {args_cli.enable_cameras}")
print(f"[DEBUG] headless = {args_cli.headless}")

app_launcher = AppLauncher(args_cli)


sim_app = app_launcher.app


# from smolvla_in_isaac.robots.so_101_cfg import so101_cfg


# ------------------------------------------------
# imports after app is created

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

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors


sim_cfg = SimulationCfg(dt=0.01)
sim = SimulationContext(sim_cfg)



sim.set_camera_view((2.5, 2.5, 2.5), (0.0, 0.0, 0.0))

ground = sim_utils.GroundPlaneCfg()
ground.func("/World/GroundPlane", ground)


# =====================================================
# Add a cube to the scene
# ====================================================

cube_cfg = sim_utils.CuboidCfg(
    size=(0.02, 0.02, 0.02),  # 5cm cube
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
    ),
    mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),  # Red cube
)

cube_cfg.func("/World/Cube", cube_cfg, translation=(0.4, 0.0, 0.5))


# =====================================================
# setup cameras
# ====================================================


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


# =======================================================
# setup so101 robot
# ========================================================

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


# ====================================================
# CORRECT sequence (like in IsaacLab tests):
# 1. Create robot and cameras
# 2. sim.reset() - triggers _initialize_callback
# 3. Warmup steps to let render graph stabilize
# ======================================================

# Create robot
robot = Articulation(robot_cfg)




print("[INFO] Creating TiledCamera objects...")
front_camera = TiledCamera(front_cam_cfg)
gripper_camera = TiledCamera(gripper_cam_cfg)



# Reset triggers _initialize_callback for cameras
print("[INFO] Calling sim.reset() to trigger camera initialization...")
sim.reset()

# Warmup steps to stabilize render graph
print("[INFO] Warmup steps to stabilize render graph...")
for i in range(5):
    sim.step()


# print("[INFO] Creating TiledCamera objects...")
# front_camera = TiledCamera(front_cam_cfg)
# gripper_camera = TiledCamera(gripper_cam_cfg)

# sim.reset()



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


# ====================================================
# loading smolvla from_pretrained
# ====================================================

print("[INFO] Loading SmolVLA model...")
model_id = "lerobot/smolvla_base"
policy = SmolVLAPolicy.from_pretrained(model_id)

# Create preprocessor and postprocessor
print("[INFO] Creating preprocessor and postprocessor...")
preprocessor, postprocessor = make_smolvla_pre_post_processors(policy.config, dataset_stats=None)

policy.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = policy.to(device)
print(f"[INFO] SmolVLA model loaded on {device}")




# Text instruction for the robot
instruction = "pick up the red cube"
step_count = 0

import time

time.sleep(2)

while sim_app.is_running():
    # Update cameras
    front_camera.update(dt=sim_cfg.dt)
    gripper_camera.update(dt=sim_cfg.dt)

    # Get camera images
    front_rgb = front_camera.data.output["rgb"][0].cpu().numpy()  # (H, W, C)
    gripper_rgb = gripper_camera.data.output["rgb"][0].cpu().numpy()  # (H, W, C)

    # Ensure images are in 0-1 float32 range
    if front_rgb.max() > 1.0:
        front_rgb = front_rgb.astype(np.float32) / 255.0
        gripper_rgb = gripper_rgb.astype(np.float32) / 255.0
    else:
        front_rgb = front_rgb.astype(np.float32)
        gripper_rgb = gripper_rgb.astype(np.float32)

    # Convert from HWC to CHW format (channels first) for the preprocessor
    front_rgb_chw = np.transpose(front_rgb, (2, 0, 1))  # (C, H, W)
    gripper_rgb_chw = np.transpose(gripper_rgb, (2, 0, 1))  # (C, H, W)

    print("Front camera shape: ", front_rgb_chw.shape)

    # Add time dimension: (C, H, W) -> (1, C, H, W) for temporal observations
    front_rgb_temporal = np.expand_dims(front_rgb_chw, axis=0)  # (1, C, H, W)
    gripper_rgb_temporal = np.expand_dims(gripper_rgb_chw, axis=0)  # (1, C, H, W)

    print("Front camera shape: ", front_rgb_temporal.shape)

    state = current_positions[0].cpu().numpy()
    state_temporal = np.expand_dims(state, axis=0)  # (1, state_dim)

    # Prepare observation dictionary for SmolVLA
    observation = {
        "observation.images.camera1": front_rgb_temporal,
        "observation.images.camera2": gripper_rgb_temporal,
        "observation.state": state_temporal,
        "task": instruction,  # Add task instruction
    }

    # Preprocess observation
    preprocessed_obs = preprocessor(observation)

    # Debug: print shapes
    if step_count == 0:
        print("\n[DEBUG] Preprocessed observation shapes:")
        for k, v in preprocessed_obs.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
            else:
                print(f"  {k}: {type(v)}")

    # Move to device (don't add batch dimension - preprocessor already handles it)
    preprocessed_obs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in preprocessed_obs.items()}

    # Get action from SmolVLA
    with torch.no_grad():
        action_dict = policy.select_action(preprocessed_obs)

    
    print("predicted action: ", action_dict)
    print("predicted action: ", action_dict)

    # Extract action (should be joint positions or deltas)
    predicted_action = action_dict  # Shape: (chunk_size, action_dim) without batch

    # Get first action from chunk
    predicted_action = predicted_action[0].cpu()  # Shape: (action_dim,)

    if step_count % 30 == 0:  # Print every 30 steps (~1 second at 30Hz)
        print(f"[Step {step_count}] Predicted action: {predicted_action[:3]}...")  # Print first 3 values

    # Apply action to robot
    # Assuming action is absolute joint positions
    action_tensor = predicted_action.unsqueeze(0).to(current_positions.device)
    robot.set_joint_position_target(action_tensor)
    robot.write_data_to_sim()

    sim.step()
    robot.update(sim_cfg.dt)

    # Update current positions for next iteration
    current_positions = robot.data.joint_pos.clone()
    step_count += 1

sim_app.close()