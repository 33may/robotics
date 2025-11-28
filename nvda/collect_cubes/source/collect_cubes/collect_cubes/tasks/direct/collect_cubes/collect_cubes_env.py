# # Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# from __future__ import annotations

# import math
# import torch
# from collections.abc import Sequence

# import isaaclab.sim as sim_utils
# from isaaclab.assets import Articulation
# from isaaclab.envs import DirectRLEnv
# from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
# from isaaclab.utils.math import sample_uniform

# from .collect_cubes_env_cfg import CollectCubesEnvCfg


# class CollectCubesEnv(DirectRLEnv):
#     cfg: CollectCubesEnvCfg

#     def __init__(self, cfg: CollectCubesEnvCfg, render_mode: str | None = None, **kwargs):
#         super().__init__(cfg, render_mode, **kwargs)

#         self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
#         self._pole_dof_idx, _ = self.robot.find_joints(self.cfg.pole_dof_name)

#         self.joint_pos = self.robot.data.joint_pos
#         self.joint_vel = self.robot.data.joint_vel

#     def _setup_scene(self):
#         self.robot = Articulation(self.cfg.robot_cfg)
#         # add ground plane
#         spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
#         # clone and replicate
#         self.scene.clone_environments(copy_from_source=False)
#         # we need to explicitly filter collisions for CPU simulation
#         if self.device == "cpu":
#             self.scene.filter_collisions(global_prim_paths=[])
#         # add articulation to scene
#         self.scene.articulations["robot"] = self.robot
#         # add lights
#         light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
#         light_cfg.func("/World/Light", light_cfg)

#     def _pre_physics_step(self, actions: torch.Tensor) -> None:
#         self.actions = actions.clone()

#     def _apply_action(self) -> None:
#         self.robot.set_joint_effort_target(self.actions * self.cfg.action_scale, joint_ids=self._cart_dof_idx)

#     def _get_observations(self) -> dict:
#         obs = torch.cat(
#             (
#                 self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
#                 self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
#                 self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
#                 self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
#             ),
#             dim=-1,
#         )
#         observations = {"policy": obs}
#         return observations

#     def _get_rewards(self) -> torch.Tensor:
#         total_reward = compute_rewards(
#             self.cfg.rew_scale_alive,
#             self.cfg.rew_scale_terminated,
#             self.cfg.rew_scale_pole_pos,
#             self.cfg.rew_scale_cart_vel,
#             self.cfg.rew_scale_pole_vel,
#             self.joint_pos[:, self._pole_dof_idx[0]],
#             self.joint_vel[:, self._pole_dof_idx[0]],
#             self.joint_pos[:, self._cart_dof_idx[0]],
#             self.joint_vel[:, self._cart_dof_idx[0]],
#             self.reset_terminated,
#         )
#         return total_reward

#     def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
#         self.joint_pos = self.robot.data.joint_pos
#         self.joint_vel = self.robot.data.joint_vel

#         time_out = self.episode_length_buf >= self.max_episode_length - 1
#         out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
#         out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
#         return out_of_bounds, time_out

#     def _reset_idx(self, env_ids: Sequence[int] | None):
#         if env_ids is None:
#             env_ids = self.robot._ALL_INDICES
#         super()._reset_idx(env_ids)

#         joint_pos = self.robot.data.default_joint_pos[env_ids]
#         joint_pos[:, self._pole_dof_idx] += sample_uniform(
#             self.cfg.initial_pole_angle_range[0] * math.pi,
#             self.cfg.initial_pole_angle_range[1] * math.pi,
#             joint_pos[:, self._pole_dof_idx].shape,
#             joint_pos.device,
#         )
#         joint_vel = self.robot.data.default_joint_vel[env_ids]

#         default_root_state = self.robot.data.default_root_state[env_ids]
#         default_root_state[:, :3] += self.scene.env_origins[env_ids]

#         self.joint_pos[env_ids] = joint_pos
#         self.joint_vel[env_ids] = joint_vel

#         self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
#         self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
#         self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


# @torch.jit.script
# def compute_rewards(
#     rew_scale_alive: float,
#     rew_scale_terminated: float,
#     rew_scale_pole_pos: float,
#     rew_scale_cart_vel: float,
#     rew_scale_pole_vel: float,
#     pole_pos: torch.Tensor,
#     pole_vel: torch.Tensor,
#     cart_pos: torch.Tensor,
#     cart_vel: torch.Tensor,
#     reset_terminated: torch.Tensor,
# ):
#     rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
#     rew_termination = rew_scale_terminated * reset_terminated.float()
#     rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
#     rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
#     rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
#     total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
#     return total_reward


# comments in English only
from __future__ import annotations
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, AssetBaseCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .collect_cubes_env_cfg import CollectCubesEnvCfg


class CollectCubesEnv(DirectRLEnv):
    """Minimal environment that only spawns a Franka arm and steps safely."""

    cfg: CollectCubesEnvCfg

    def __init__(self, cfg: CollectCubesEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Time step for action scaling
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # Get joint limits from robot
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        # Speed scales for different joints (gripper slower)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self.robot.find_joints("panda_finger_joint1")[0]] = 0.1
        self.robot_dof_speed_scales[self.robot.find_joints("panda_finger_joint2")[0]] = 0.1

        # Buffer for target positions
        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

        # Find body indices
        self.hand_link_idx = self.robot.find_bodies("panda_hand")[0][0]

    # ------------------------------------------------------------------ #
    # Scene setup
    # ------------------------------------------------------------------ #
    def _setup_scene(self):
        # spawn the robot
        # self.robot = Articulation(self.cfg.robot_cfg)

        # # replicate environments
        # self.scene.clone_environments(copy_from_source=False)
        # if self.device == "cpu":
        #     self.scene.filter_collisions(global_prim_paths=[])

        # # register robot
        # self.scene.articulations["robot"] = self.robot


        # 1. create articulation
        self.robot = Articulation(self.cfg.robot_cfg)

        # spawn cube
        self.cube = RigidObject(self.cfg.cube_cfg)


        # spawn bucket components (5 cuboids)
        self.bucket_floor = RigidObject(self.cfg.bucket_floor_cfg)
        self.bucket_wall_left = RigidObject(self.cfg.bucket_wall_left_cfg)
        self.bucket_wall_right = RigidObject(self.cfg.bucket_wall_right_cfg)
        self.bucket_wall_front = RigidObject(self.cfg.bucket_wall_front_cfg)
        self.bucket_wall_back = RigidObject(self.cfg.bucket_wall_back_cfg)


        # 2. Spawn ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # 4. Clone envs
        self.scene.clone_environments(copy_from_source=False)

        # 5. Register articulation robot and rigid objects
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["cube"] = self.cube
        self.scene.rigid_objects["bucket_floor"] = self.bucket_floor
        self.scene.rigid_objects["bucket_wall_left"] = self.bucket_wall_left
        self.scene.rigid_objects["bucket_wall_right"] = self.bucket_wall_right
        self.scene.rigid_objects["bucket_wall_front"] = self.bucket_wall_front
        self.scene.rigid_objects["bucket_wall_back"] = self.bucket_wall_back

        # 6. Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ #
    # RL interface stubs
    # ------------------------------------------------------------------ #
    def _pre_physics_step(self, actions: torch.Tensor):
        # Clamp actions to [-1, 1]
        self.actions = actions.clone().clamp(-1.0, 1.0)

        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale

        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits * self.robot_dof_upper_limits)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.robot_dof_targets)

    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0 * (self.robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )

        dof_vel_scaled = self.robot.data.joint_vel * 0.1

        hand_pos = self.robot.data.body_pos_w[:, self.hand_link_idx, :]

        cube_pos = self.cube.data.root_pos_w

        to_cube = cube_pos - hand_pos

        obs = torch.cat(
            (
                dof_pos_scaled,  # 9
                dof_vel_scaled,  # 9
                to_cube,         # 3
                cube_pos,        # 3
            ),
            dim=-1
        )                        # 24

        return {"policy" : torch.clamp(obs, -5.0, 5.0)}

    def _get_rewards(self) -> torch.Tensor:
        # Get positions
        hand_pos = self.robot.data.body_pos_w[:, self.hand_link_idx, :]
        cube_pos = self.cube.data.root_pos_w
        bucket_center = torch.tensor([0.5, 0.5, 0.05], device=self.device).repeat(self.num_envs, 1)

        # Distance from hand to cube (encourage reaching)
        dist_hand_to_cube = torch.norm(hand_pos - cube_pos, dim=-1)
        reward_reach = -dist_hand_to_cube * 0.5

        # Distance from cube to bucket center (encourage moving cube to bucket)
        dist_cube_to_bucket = torch.norm(cube_pos[:, :2] - bucket_center[:, :2], dim=-1)
        reward_cube_to_bucket = -dist_cube_to_bucket * 2.0

        # Check if cube is inside bucket
        bucket_x_min, bucket_x_max = 0.5 - 0.115, 0.5 + 0.115  # ~23cm inner width
        bucket_y_min, bucket_y_max = 0.5 - 0.115, 0.5 + 0.115
        bucket_z_min, bucket_z_max = 0.01, 0.13  # floor to wall height

        in_bucket_x = (cube_pos[:, 0] > bucket_x_min) & (cube_pos[:, 0] < bucket_x_max)
        in_bucket_y = (cube_pos[:, 1] > bucket_y_min) & (cube_pos[:, 1] < bucket_y_max)
        in_bucket_z = (cube_pos[:, 2] > bucket_z_min) & (cube_pos[:, 2] < bucket_z_max)
        cube_in_bucket = in_bucket_x & in_bucket_y & in_bucket_z

        reward_in_bucket = cube_in_bucket.float() * 100.0

        # Small time penalty to encourage faster completion
        reward_time = -0.01

        total_reward = reward_reach + reward_cube_to_bucket + reward_in_bucket + reward_time

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Get cube position
        cube_pos = self.cube.data.root_pos_w

        # Check if cube is inside bucket (success condition)
        bucket_x_min, bucket_x_max = 0.5 - 0.115, 0.5 + 0.115
        bucket_y_min, bucket_y_max = 0.5 - 0.115, 0.5 + 0.115
        bucket_z_min, bucket_z_max = 0.01, 0.13

        in_bucket_x = (cube_pos[:, 0] > bucket_x_min) & (cube_pos[:, 0] < bucket_x_max)
        in_bucket_y = (cube_pos[:, 1] > bucket_y_min) & (cube_pos[:, 1] < bucket_y_max)
        in_bucket_z = (cube_pos[:, 2] > bucket_z_min) & (cube_pos[:, 2] < bucket_z_max)
        cube_in_bucket = in_bucket_x & in_bucket_y & in_bucket_z

        # Terminate if cube successfully placed in bucket
        done = cube_in_bucket

        # Standard timeout
        timeout = self.episode_length_buf >= self.max_episode_length - 1

        return done, timeout

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # skip if robot not yet created (called during early reset)
        if self.robot is None or self.robot.data is None:
            return
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        q = self.robot.data.default_joint_pos[env_ids]
        dq = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(q, dq, None, env_ids)

