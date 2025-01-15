# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Train to drive in a direction despite random starting orientation

from __future__ import annotations

import torch
from collections.abc import Sequence

from .Leatherback import LEATHERBACK_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass

@configclass
class LeatherbackEnvCfg(DirectRLEnvCfg):
    
    # env
    decimation = 2              # Decimation (number of time steps between actions)
    episode_length_s = 5.0      # Max each episode should last in seconds
    action_space = 2            # Number of actions the neural network should return    
    observation_space = 4       # Number of observations fed into neural network
    state_space = 0             # Observations to be used in Actor-Critic training

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    throttle_dof_name = [
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right",
        "Wheel__Upright__Rear_Right",
        "Wheel__Upright__Rear_Left",
    ]
    steering_dof_name = [
        "Knuckle__Upright__Front_Right",
        "Knuckle__Upright__Front_Left",
    ]

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    
class LeatherbackEnv(DirectRLEnv):
    cfg: LeatherbackEnvCfg

    def __init__(self, cfg: LeatherbackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._throttle_dof_idx, _ = self.Leatherback.find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.Leatherback.find_joints(self.cfg.steering_dof_name)

        self.throttle_state = torch.zeros((self.num_envs,4), dtype=torch.float32, device=self.device)
        self.steering_state = torch.zeros((self.num_envs,2), dtype=torch.float32, device=self.device)

    def _setup_scene(self):

        self.Leatherback = Articulation(self.cfg.robot_cfg)
        
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)       # Clones child environments from parent environment
        self.scene.filter_collisions(global_prim_paths=[])          # Prevents environments from colliding with each other
        
        # add articulation to scene
        self.scene.articulations["leatherback"] = self.Leatherback
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        
        throttle_scale = 0.05
        steering_scale = 0.01
        self.actions = actions.clone()
        
        self._throttle_action = actions[:, 0].repeat_interleave(4).reshape((-1, 4)) * throttle_scale
        self._throttle_action += self.throttle_state
        self.throttle_state = self._throttle_action

        self._steering_action = actions[:, 1].repeat_interleave(2).reshape((-1, 2)) * steering_scale
        self._steering_action += self.steering_state
        self.steering_state = self._steering_action

    def _apply_action(self) -> None:        
        self.Leatherback.set_joint_velocity_target(self._throttle_action, joint_ids=self._throttle_dof_idx)
        self.Leatherback.set_joint_position_target(self._steering_action, joint_ids=self._steering_dof_idx)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.Leatherback.data.root_quat_w[:,0].unsqueeze(dim=1),
                self.Leatherback.data.root_quat_w[:,3].unsqueeze(dim=1),
                self.throttle_state[:,0].unsqueeze(dim=1),
                self.steering_state[:,0].unsqueeze(dim=1)
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        
        rewards = -1.0 * (self.Leatherback.data.root_pos_w[:,0] - self.scene.env_origins[:,0])
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        
        clean_termination = self.episode_length_buf >= self.max_episode_length - 1
        
        failure_termination = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        return failure_termination, clean_termination

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.Leatherback._ALL_INDICES
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)
        
        # Reset from config
        default_state = self.Leatherback.data.default_root_state[env_ids]        # first three are pos, next 4 quats, next 3 vel, next 3 ang vel
        leatherback_pose = default_state[env_ids, :7]                                  # proper way of getting default pose from config file
        leatherback_velocities = default_state[env_ids, 7:]                            # proper way of getting default velocities from config file
        joint_positions = self.Leatherback.data.default_joint_pos[env_ids]       # proper way to get joint positions from config file
        joint_velocities = self.Leatherback.data.default_joint_vel[env_ids]      # proper way to get joint velocities form config file

        leatherback_pose[:, :3] += self.scene.env_origins[env_ids]               # Adds center of each env position to leatherback position

        self.throttle_state[env_ids] = 0.0
        self.steering_state[env_ids] = 0.0

        # Randomize Starting Position
        # leatherback_pose[:, :2] += 2.0 * torch.rand((num_reset, 2), dtype=torch.float32, device=self.device)
        
        # Randomize starting Heading
        angles = torch.pi * torch.rand((num_reset,), dtype=torch.float32, device=self.device)
        
        # Isaac Sim quaternions are W-first (w, x, y, z) To rotate about the Z axis, we'll modify the W and Z values
        leatherback_pose[:, 3] = torch.cos(angles * 0.5)
        leatherback_pose[:, 6] = torch.sin(angles * 0.5)

        self.Leatherback.write_root_pose_to_sim(leatherback_pose, env_ids)
        self.Leatherback.write_root_velocity_to_sim(leatherback_velocities, env_ids)
        self.Leatherback.write_joint_state_to_sim(joint_positions, joint_velocities, None, env_ids)