# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from Leatherback import LEATHERBACK_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform


@configclass
class LeatherbackEnvCfg(DirectRLEnvCfg):
    
    # # env
    decimation = 2
    episode_length_s = 5.0
    action_space = 0
    observation_space = 0
    state_space = 0

    # action_scale = 100.0  # [N]

    # # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # # robot
    # robot_cfg: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # wheel_1_dof_name = "Wheel__Upright__Rear_Right"
    # wheel_2_dof_name = "Wheel__Upright__Rear_Left"
    # wheel_3_dof_name = "Wheel__Upright__Front_Right"
    # wheel_4_dof_name = "Wheel__Upright__Front_Left"
    # steer_right_dof_name = "Knuckle__Upright__Front_Right"
    # steer_left_dof_name = "Knuckle__Upright__Front_Left"

    # # scene
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # # reset
    # # max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    # # initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]

    # # reward scales
    # rew_scale_cone_distance = 1.0

class LeatherbackEnv(DirectRLEnv):
    cfg: LeatherbackEnvCfg

    def __init__(self, cfg: LeatherbackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # self._wheel_1_dof_idx, _ = self.Leatherback.find_joints(self.cgf.wheel_1_dof_name)
        # self._wheel_2_dof_idx, _ = self.Leatherback.find_joints(self.cgf.wheel_2_dof_name)
        # self._wheel_3_dof_idx, _ = self.Leatherback.find_joints(self.cgf.wheel_3_dof_name)
        # self._wheel_4_dof_idx, _ = self.Leatherback.find_joints(self.cgf.wheel_4_dof_name)
        # self._steer_right_dof_idx, _ = self.Leatherback.find_joints(self.cgf.steer_right_dof_name)
        # self._steer_left_dof_idx, _ = self.Leatherback.find_joints(self.cgf.steer_left_dof_name)
        # self.action_scale = self.cfg.action_scale

        # self.joint_pos = self.Leatherback.data.joint_pos
        # self.joint_vel = self.Leatherback.data.joint_vel
        print("Init Complete")

    def _setup_scene(self):
        # self.Leatherback = Articulation(self.cfg.robot_cfg)
        # # add ground plane
        # spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # # clone, filter, and replicate
        # self.scene.clone_environments(copy_from_source=False)
        # self.scene.filter_collisions(global_prim_paths=[])
        # # add articulation to scene
        # self.scene.articulations["leatherback"] = self.Leatherback
        # # add lights
        # light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        # light_cfg.func("/World/Light", light_cfg)

        print("Setup Scene Complete")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # self.actions = self.action_scale * actions.clone()

        print("Pre Physics Step Complete")

    def _apply_action(self) -> None:
        # self.cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)
        print("Apply Action Complete")

    def _get_observations(self) -> dict:
        # obs = torch.cat(
        #     (
        #         self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
        #     ),
        #     dim=-1,
        # )
        # observations = {"policy": obs}
        # return observations
        print("Get Observations Complete")

    def _get_rewards(self) -> torch.Tensor:
        # total_reward = compute_rewards(
        #     self.cfg.rew_scale_alive,
        #     self.cfg.rew_scale_terminated,
        #     self.cfg.rew_scale_pole_pos,
        #     self.cfg.rew_scale_cart_vel,
        #     self.cfg.rew_scale_pole_vel,
        #     self.joint_pos[:, self._pole_dof_idx[0]],
        #     self.joint_vel[:, self._pole_dof_idx[0]],
        #     self.joint_pos[:, self._cart_dof_idx[0]],
        #     self.joint_vel[:, self._cart_dof_idx[0]],
        #     self.reset_terminated,
        # )
        # return total_reward
        print("Get Rewards Complete")
        

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # self.joint_pos = self.cartpole.data.joint_pos
        # self.joint_vel = self.cartpole.data.joint_vel

        # time_out = self.episode_length_buf >= self.max_episode_length - 1
        # out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        # out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        # return out_of_bounds, time_out
        print("get dones complete")

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # if env_ids is None:
        #     env_ids = self.leatherback._ALL_INDICES
        # super()._reset_idx(env_ids)

        # joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        # joint_pos[:, self._pole_dof_idx] += sample_uniform(
        #     self.cfg.initial_pole_angle_range[0] * math.pi,
        #     self.cfg.initial_pole_angle_range[1] * math.pi,
        #     joint_pos[:, self._pole_dof_idx].shape,
        #     joint_pos.device,
        # )
        # joint_vel = self.cartpole.data.default_joint_vel[env_ids]

        # default_root_state = self.cartpole.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # self.joint_pos[env_ids] = joint_pos
        # self.joint_vel[env_ids] = joint_vel

        # self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        print("reset idx complete")


@torch.jit.script
def compute_rewards(
    # rew_scale_alive: float,
    # rew_scale_terminated: float,
    # rew_scale_pole_pos: float,
    # rew_scale_cart_vel: float,
    # rew_scale_pole_vel: float,
    # pole_pos: torch.Tensor,
    # pole_vel: torch.Tensor,
    # cart_pos: torch.Tensor,
    # cart_vel: torch.Tensor,
    # reset_terminated: torch.Tensor,
):
    # rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    # rew_termination = rew_scale_terminated * reset_terminated.float()
    # rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    # rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    # rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    # total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    # return total_reward
    print("compute rewards complete")
    return None