"""
Isaac Lab environment setup utilities.

This module provides common functions for setting up Isaac Lab simulation,
including AppLauncher configuration and environment creation.
"""
import os
import sys
import warnings
from pathlib import Path
from typing import Any

# Isaac Lab imports (must be after AppLauncher)
# These will be imported conditionally in functions


def setup_isaac_environment():
    """
    Set up Isaac Lab environment variables and suppress warnings.

    Call this BEFORE importing Isaac Lab modules.
    """
    # Set leisaac assets root
    leisaac_assets = Path("/home/may33/projects/ml_portfolio/robotics/leisaac/assets")
    if leisaac_assets.exists():
        os.environ["LEISAIAC_ASSETS_ROOT"] = str(leisaac_assets)

    # Suppress Isaac Sim warnings
    os.environ["ISAAC_SUPPRESS_WARNINGS"] = "1"
    os.environ["CARB_LOGGING_MAX_LEVEL"] = "ERROR"
    warnings.filterwarnings("ignore")


def create_app_launcher(
    headless: bool = False,
    livestream: int = 0,
    enable_cameras: bool = True,
    livestream_address: str = "100.115.105.111",
    livestream_port: int = 49100,
) -> tuple:
    """
    Create and configure Isaac Lab AppLauncher.

    Args:
        headless: Run without GUI
        livestream: Livestream mode (0=off, 2=WebRTC)
        enable_cameras: Enable camera rendering
        livestream_address: WebRTC server address
        livestream_port: WebRTC server port

    Returns:
        Tuple of (app_launcher, simulation_app)
    """
    from isaaclab.app import AppLauncher
    import argparse

    # Create minimal parser for AppLauncher
    parser = argparse.ArgumentParser(description="Isaac Lab Environment Setup")
    args_cli, _ = parser.parse_known_args()
    args_cli.headless = headless
    args_cli.enable_cameras = enable_cameras

    # Setup streaming if enabled
    if livestream == 2:
        args_cli.headless = True
        sys.argv.append(f"--/app/livestream/publicEndpointAddress={livestream_address}")
        sys.argv.append(f"--/app/livestream/port={livestream_port}")

        print(f"[INFO] Livestream enabled (WebRTC mode)")
        print(f"[INFO] Stream address: {livestream_address}:{livestream_port}")

    # Launch Isaac Sim
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    return app_launcher, simulation_app


def create_lift_cube_environment(
    num_envs: int = 1,
    teleop_device: str = "so101leader",
) -> Any:
    """
    Create Isaac Lab lift_cube environment.

    Args:
        num_envs: Number of parallel environments
        teleop_device: Teleoperation device type ("so101leader")

    Returns:
        Configured ManagerBasedRLEnv environment
    """
    from isaaclab.envs import ManagerBasedRLEnv
    from leisaac.tasks.lift_cube.lift_cube_env_cfg import LiftCubeEnvCfg

    print("\nInitializing Isaac Lab environment...")
    env_cfg = LiftCubeEnvCfg()
    env_cfg.scene.num_envs = num_envs

    # Configure actions for joint position control
    env_cfg.use_teleop_device(teleop_device)

    env = ManagerBasedRLEnv(cfg=env_cfg)

    print(f"Environment created: {env.unwrapped.__class__.__name__}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space keys: {list(env.observation_space.keys())}")

    return env


def print_environment_info(env: Any):
    """
    Print detailed information about Isaac Lab environment.

    Args:
        env: Isaac Lab environment instance
    """
    print("\n" + "=" * 60)
    print("ISAAC LAB ENVIRONMENT INFO")
    print("=" * 60)

    # Command Manager
    print(f"\n[INFO] Command Manager: {env.command_manager}")

    # Event Manager
    print(f"[INFO] Event Manager: {env.event_manager}")

    # Recorder Manager
    print(f"[INFO] Recorder Manager: {env.recorder_manager}")

    # Action Manager
    print(f"[INFO] Action Manager: {env.action_manager}")

    # Observation Manager
    print(f"[INFO] Observation Manager: {env.observation_manager}")
    for group_name, group in env.observation_manager._groups.items():
        print(f"\n  Observation Group: '{group_name}'")
        for term_name, term in group._terms.items():
            print(f"    - {term_name}: {term}")

    # Termination Manager
    print(f"\n[INFO] Termination Manager: {env.termination_manager}")

    # Reward Manager
    print(f"[INFO] Reward Manager: {env.reward_manager}")

    # Curriculum Manager
    print(f"[INFO] Curriculum Manager: {env.curriculum_manager}")

    print("=" * 60 + "\n")


def safe_close_environment(env: Any, simulation_app: Any):
    """
    Safely close Isaac Lab environment and simulation app.

    Args:
        env: Isaac Lab environment instance
        simulation_app: Isaac Sim application instance
    """
    try:
        if env is not None:
            env.close()
            print("[INFO] Environment closed successfully")
    except Exception as e:
        print(f"[WARNING] Error closing environment: {e}")

    try:
        if simulation_app is not None:
            simulation_app.close()
            print("[INFO] Simulation app closed successfully")
    except Exception as e:
        print(f"[WARNING] Error closing simulation app: {e}")
