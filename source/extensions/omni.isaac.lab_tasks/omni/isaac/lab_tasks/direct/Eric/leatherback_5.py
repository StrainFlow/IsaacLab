# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Train to drive to a waypoint in the environment

from __future__ import annotations

import torch
from collections.abc import Sequence

from .Leatherback import LEATHERBACK_CFG
from .Cone import CONE_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass

@configclass
class LeatherbackEnvCfg(DirectRLEnvCfg):
    
    # env
    decimation = 2              # Decimation (number of time steps between actions)
    episode_length_s = 15.0     # Max each episode should last in seconds
    action_space = 2            # Number of actions the neural network should return    
    observation_space = 11      # Number of observations fed into neural network
    state_space = 0             # Observations to be used in Actor-Critic training

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cone_cfg: VisualizationMarkersCfg = CONE_CFG

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

        self._throttle_state = torch.zeros((self.num_envs,4), dtype=torch.float32, device=self.device)
        self._steering_state = torch.zeros((self.num_envs,2), dtype=torch.float32, device=self.device)

        self._position_error = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self._position_dist = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)

        self._target_positions = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self._markers_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.goal_reached = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)

        # Action Parameters        
        self.throttle_scale = 1
        self.throttle_max = 50.0
        self.steering_scale = 0.1
        self.steering_max = 0.75

        # Boundary parameters
        self.reward_coeff: float = 1.0
        self.position_tolerance: float = 0.2

        # Reward Scales
        self.position_rew_scale = 1
        self.goal_reached_scale = 10

    def _setup_scene(self):

        self.Leatherback = Articulation(self.cfg.robot_cfg)
        self.Cones = VisualizationMarkers(self.cfg.cone_cfg)
        
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
        
        self._throttle_action = actions[:, 0].repeat_interleave(4).reshape((-1, 4)) * self.throttle_scale
        self._throttle_action += self._throttle_state
        self._throttle_action = torch.clamp(self._throttle_action, -self.throttle_max, self.throttle_max)
        self._throttle_state = self._throttle_action

        self._steering_action = actions[:, 1].repeat_interleave(2).reshape((-1, 2)) * self.steering_scale
        self._steering_action += self._steering_state
        self._steering_action = torch.clamp(self._steering_action, -self.steering_max, self.steering_max)
        self._steering_state = self._steering_action

    def _apply_action(self) -> None:        
        self.Leatherback.set_joint_velocity_target(self._throttle_action, joint_ids=self._throttle_dof_idx)
        self.Leatherback.set_joint_position_target(self._steering_action, joint_ids=self._steering_dof_idx)

    def _get_observations(self) -> dict:

        # position error
        self._position_error = self._target_positions[:, :2] - self.Leatherback.data.root_pos_w[:, :2]
        self._position_dist = torch.norm(self._position_error, dim=-1)
        # position error expressed as distance and angular error (to the position)
        heading = self.Leatherback.data.heading_w[:]
        target_heading_w = torch.atan2(
            self._target_positions[:, 1] - self.Leatherback.data.root_pos_w[:, 1],
            self._target_positions[:, 0] - self.Leatherback.data.root_pos_w[:, 0],
        )
        target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))

        obs = torch.cat(
            (
                self._position_dist.unsqueeze(dim=1),
                torch.cos(target_heading_error).unsqueeze(dim=1),
                torch.sin(target_heading_error).unsqueeze(dim=1),
                self.Leatherback.data.root_lin_vel_b[:,0].unsqueeze(dim=1),
                self.Leatherback.data.root_lin_vel_b[:,1].unsqueeze(dim=1),
                self.Leatherback.data.root_lin_vel_b[:,2].unsqueeze(dim=1),
                self.Leatherback.data.root_ang_vel_w[:,0].unsqueeze(dim=1),
                self.Leatherback.data.root_ang_vel_w[:,1].unsqueeze(dim=1),
                self.Leatherback.data.root_ang_vel_w[:,2].unsqueeze(dim=1),
                self._throttle_state[:,0].unsqueeze(dim=1),
                self._steering_state[:,0].unsqueeze(dim=1)
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:

        # position reward
        position_rew = torch.exp(-self._position_dist / self.reward_coeff)      
        self.goal_reached = (self._position_dist < self.position_tolerance).int()  
        
        return (
            position_rew * self.position_rew_scale 
            + self.goal_reached * self.goal_reached_scale
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        
        task_failed = self.episode_length_buf > self.max_episode_length
        
        # Checks if the goal is reached
        task_completed = self.goal_reached   
    
        return task_failed, task_completed

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.Leatherback._ALL_INDICES
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)

        #region Reset Robot
        # Reset from config
        default_state = self.Leatherback.data.default_root_state[env_ids]        # first three are pos, next 4 quats, next 3 vel, next 3 ang vel
        leatherback_pose = default_state[:, :7]                                  # proper way of getting default pose from config file
        leatherback_velocities = default_state[:, 7:]                            # proper way of getting default velocities from config file
        joint_positions = self.Leatherback.data.default_joint_pos[env_ids]       # proper way to get joint positions from config file
        joint_velocities = self.Leatherback.data.default_joint_vel[env_ids]      # proper way to get joint velocities form config file

        leatherback_pose[:, :3] += self.scene.env_origins[env_ids]               # Adds center of each env position to leatherback position

        # Randomize Starting Position
        leatherback_pose[:, :2] += 2.0 * torch.rand((num_reset, 2), dtype=torch.float32, device=self.device)
        
        # Randomize starting Heading
        angles = torch.pi * torch.rand((num_reset,), dtype=torch.float32, device=self.device)
        
        # Isaac Sim quaternions are W-first (w, x, y, z) To rotate about the Z axis, we'll modify the W and Z values
        leatherback_pose[:, 3] = torch.cos(angles * 0.5)
        leatherback_pose[:, 6] = torch.sin(angles * 0.5)

        self.Leatherback.write_root_pose_to_sim(leatherback_pose, env_ids)
        self.Leatherback.write_root_velocity_to_sim(leatherback_velocities, env_ids)
        self.Leatherback.write_joint_state_to_sim(joint_positions, joint_velocities, None, env_ids)
        #endregion Reset Robot

        #region Reset Actions
        self._throttle_state[env_ids] = 0.0
        self._steering_state[env_ids] = 0.0
        #endregion Reset Actions

        #region Reset Goals
        # The position is picked randomly in a square centered on the origin
        self._target_positions[env_ids] = 2.0 * torch.rand((num_reset, 2), dtype=torch.float32, device=self.device) 
        self._target_positions[env_ids] += self.scene.env_origins[env_ids, :2]

        # Update the visual markers
        self._markers_pos[env_ids, :2] = self._target_positions[env_ids]
        self.Cones.visualize(translations=self._markers_pos)
        #endregion Set Goals

        #region Make sure the position error and position dist are up to date after the reset
        self._position_error[env_ids] = (
            self._target_positions[env_ids] - self.Leatherback.data.root_pos_w[env_ids, :2]
        )
        self._position_dist[env_ids] = torch.linalg.norm(self._position_error[env_ids], dim=-1)
        #endregion

        #region Reset Rewards
        self.goal_reached[env_ids] = False
        #endregion Reset Rewards
        

