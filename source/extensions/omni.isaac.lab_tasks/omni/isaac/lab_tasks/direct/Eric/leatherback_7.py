# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Train to drive through a series of waypoints in the environment

from __future__ import annotations

import torch
from collections.abc import Sequence

from .Leatherback import LEATHERBACK_CFG
from .Waypoint import WAYPOINT_CFG
from .Cone import CONE_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
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
    episode_length_s = 30.0     # Max each episode should last in seconds
    action_space = 2            # Number of actions the neural network should return    
    observation_space = 5       # Number of observations fed into neural network
    state_space = 0             # Observations to be used in Actor-Critic training
    env_spacing = 16.0
    num_goals = 10

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    waypoint_cfg: VisualizationMarkersCfg = WAYPOINT_CFG
    
    cone_cfgs = []

    for i in range(num_goals):
        cone_cfg = CONE_CFG.copy()
        cone_cfg.prim_path = f"/World/envs/env_.*/Cone_{i}"
        cone_cfgs.append(cone_cfg)
    
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=env_spacing, replicate_physics=True)
    
class LeatherbackEnv(DirectRLEnv):
    cfg: LeatherbackEnvCfg

    def __init__(self, cfg: LeatherbackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._throttle_dof_idx, _ = self.Leatherback.find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.Leatherback.find_joints(self.cfg.steering_dof_name)

        self._throttle_state = torch.zeros((self.num_envs,4), dtype=torch.float32, device=self.device)
        self._steering_state = torch.zeros((self.num_envs,2), dtype=torch.float32, device=self.device)

        self._goal_reached = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        self.task_completed = torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)

        self._num_goals = self.cfg.num_goals

        self._target_positions = torch.zeros((self.num_envs, self._num_goals, 2), device=self.device, dtype=torch.float32)
        self._markers_pos = torch.zeros((self.num_envs, self._num_goals, 3), device=self.device, dtype=torch.float32)

        self.env_spacing = self.cfg.env_spacing
        self.course_length_coefficient = 2.5
        self.course_width_coefficient = 2.0

        # Reward parameters
        self.position_tolerance: float = 0.2
        self.goal_reached_bonus: float = 10.0
        self.position_progress_weight: float = 1.0
        self.heading_coefficient = 0.25
        self.heading_progress_weight: float = 0.05

        self._target_index = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)

    def _setup_scene(self):

        self.Leatherback = Articulation(self.cfg.robot_cfg)
        self.Waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)
        self.Cones = []

        for cone_cfg in self.cfg.cone_cfgs:
            self.Cones.append(RigidObject(cone_cfg))
        
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
        
        throttle_scale = 1
        throttle_max = 50.0
        steering_scale = 0.1
        steering_max = 0.75
        
        self._throttle_action = actions[:, 0].repeat_interleave(4).reshape((-1, 4)) * throttle_scale
        self._throttle_action += self._throttle_state
        self._throttle_action = torch.clamp(self._throttle_action, -throttle_max, 0.0)
        self._throttle_state = self._throttle_action

        self._steering_action = actions[:, 1].repeat_interleave(2).reshape((-1, 2)) * steering_scale
        self._steering_action += self._steering_state
        self._steering_action = torch.clamp(self._steering_action, -steering_max, steering_max)
        self._steering_state = self._steering_action

    def _apply_action(self) -> None:        
        self.Leatherback.set_joint_velocity_target(-self._throttle_action, joint_ids=self._throttle_dof_idx)
        self.Leatherback.set_joint_position_target(self._steering_state, joint_ids=self._steering_dof_idx)
    
    def _get_observations(self) -> dict:

        # position error
        current_target_positions = self._target_positions[self.Leatherback._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions - self.Leatherback.data.root_pos_w[:, :2]
        self._previous_position_error = self._position_error.clone()
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        
        # heading error
        heading = self.Leatherback.data.heading_w
        target_heading_w = torch.atan2(
            self._target_positions[self.Leatherback._ALL_INDICES, self._target_index, 1]
            - self.Leatherback.data.root_link_pos_w[:, 1],
            self._target_positions[self.Leatherback._ALL_INDICES, self._target_index, 0]
            - self.Leatherback.data.root_link_pos_w[:, 0],
        )
        self.target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        
        obs = torch.cat(
            (
                self._position_error.unsqueeze(dim=1),
                torch.cos(self.target_heading_error).unsqueeze(dim=1),
                torch.sin(self.target_heading_error).unsqueeze(dim=1),
                self._throttle_state[:,0].unsqueeze(dim=1),
                self._steering_state[:,0].unsqueeze(dim=1)
            ),
            dim=-1,
        ) 

        if torch.any(obs.isnan()):
            raise ValueError("Observations cannot be NAN")

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:

        # Position progress
        position_progress_rew = self._previous_position_error - self._position_error

        # Heading Distance
        target_heading_rew = torch.exp(-torch.abs(self.target_heading_error) / self.heading_coefficient)
        
        # Checks if the goal is reached
        goal_reached = self._position_error < self.position_tolerance

        # if the goal is reached, the target index is updated
        self._target_index = self._target_index + goal_reached

        self.task_completed = self._target_index > (self._num_goals - 1)

        self._target_index = self._target_index % self._num_goals

        composite_reward = (
            position_progress_rew * self.position_progress_weight
            + target_heading_rew * self.heading_progress_weight 
            + goal_reached * self.goal_reached_bonus
            )
        
        #region debugging
        # Update Waypoints
        one_hot_encoded = torch.nn.functional.one_hot(self._target_index.long(), num_classes=self._num_goals)
        marker_indices = one_hot_encoded.view(-1).tolist()
        self.Waypoints.visualize(marker_indices=marker_indices)

        if torch.any(composite_reward.isnan()):
            raise ValueError("Rewards cannot be NAN")

        return composite_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        
        task_failed = self.episode_length_buf > self.max_episode_length

        # task_completed is calculated in _get_rewards before target_index is wrapped around

        return task_failed, self.task_completed

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

        # Randomize Starting Position at start of track
        leatherback_pose[:, 0] -= self.env_spacing / 2
        leatherback_pose[:, 1] += 2.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device) * self.course_width_coefficient
        
        # Randomize starting Heading
        angles = torch.pi / 6.0 * torch.rand((num_reset,), dtype=torch.float32, device=self.device)
        
        # Isaac Sim quaternions are W-first (w, x, y, z) To rotate about the Z axis, we'll modify the W and Z values
        leatherback_pose[:, 3] = torch.cos(angles * 0.5)
        leatherback_pose[:, 6] = torch.sin(angles * 0.5)

        self.Leatherback.write_root_pose_to_sim(leatherback_pose, env_ids)
        self.Leatherback.write_root_velocity_to_sim(leatherback_velocities, env_ids)
        self.Leatherback.write_joint_state_to_sim(joint_positions, joint_velocities, None, env_ids)
        #endregion Reset Robot

        #region Reset Cones
        offset = 0.0
        for cone in self.Cones:
            cone_default_state = cone.data.default_root_state[env_ids].clone()
            cone_pose = cone_default_state[:, :7]
            cone_pose[:, :3] += self.scene.env_origins[env_ids]
            cone_pose[:, 0] += offset
            offset += 1.0 
            cone.write_root_pose_to_sim(cone_pose, env_ids)
        #endregion Reset Cones

        #region Reset Actions
        self._throttle_state[env_ids] = 0.0
        self._steering_state[env_ids] = 0.0
        #endregion Reset Actions

        #region Reset Goals
        self._target_positions[env_ids, :, :] = 0.0
        self._markers_pos[env_ids, :, :] = 0.0
        
        spacing = 2 / self._num_goals
        target_positions = torch.arange(-0.8, 1.1, spacing, device=self.device) * self.env_spacing / self.course_length_coefficient
        self._target_positions[env_ids, :len(target_positions), 0] = target_positions
        self._target_positions[env_ids, :, 1] = torch.rand((num_reset, self._num_goals), dtype=torch.float32, device=self.device) * self.course_width_coefficient
        self._target_positions[env_ids, :] += self.scene.env_origins[env_ids, :2].unsqueeze(1)

        self._target_index[env_ids] = 0

        # Update the visual markers
        self._markers_pos[env_ids, :, :2] = self._target_positions[env_ids]
        visualize_pos = self._markers_pos.view(-1, 3)
        self.Waypoints.visualize(translations=visualize_pos)
        #endregion Reset Goals

        #region Make sure the position error and position dist are up to date after the reset
        # reset position error
        current_target_positions = self._target_positions[self.Leatherback._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions[:, :2] - self.Leatherback.data.root_pos_w[:, :2]
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        self._previous_position_error = self._position_error.clone()
        
        # reset heading error
        heading = self.Leatherback.data.heading_w[:]
        target_heading_w = torch.atan2(
            self._target_positions[:, 0, 1] - self.Leatherback.data.root_pos_w[:, 1],
            self._target_positions[:, 0, 0] - self.Leatherback.data.root_pos_w[:, 0],
        )
        self._heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        self._previous_heading_error = self._heading_error.clone()
        #endregion
        

