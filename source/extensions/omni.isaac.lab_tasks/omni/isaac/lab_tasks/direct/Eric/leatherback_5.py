# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Train to drive to a point in the environment

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

        self._throttle_state = torch.zeros((self.num_envs,4), dtype=torch.float32, device=self.device)
        self._steering_state = torch.zeros((self.num_envs,2), dtype=torch.float32, device=self.device)
        self._actions = torch.zeros((self.num_envs,2), dtype=torch.float32, device=self.device)

        self._position_error = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self._position_dist = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self._target_positions = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        self._markers_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self._goal_reached = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)

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
        self._throttle_action += self._throttle_state
        self._throttle_state = self._throttle_action

        self._steering_action = actions[:, 1].repeat_interleave(2).reshape((-1, 2)) * steering_scale
        self._steering_action += self._steering_state
        self._steering_state = self._steering_action

        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()

    def _apply_action(self) -> None:        
        self.Leatherback.set_joint_velocity_target(self._throttle_action, joint_ids=self._throttle_dof_idx)
        self.Leatherback.set_joint_position_target(self._steering_action, joint_ids=self._steering_dof_idx)

    def _get_observations(self) -> dict:
        """
        Computes the observation tensor from the current state of the robot.

        Args:
            robot_data: The current state of the robot.

        self._task_data[:, 0] = The distance between the robot and the target position.
        self._task_data[:, 1] = The cosine of the angle between the robot heading and the target position.
        self._task_data[:, 2] = The sine of the angle between the robot heading and the target position.
        self._task_data[:, 3] = The linear velocity of the robot along the x-axis.
        self._task_data[:, 4] = The linear velocity of the robot along the y-axis.
        self._task_data[:, 5] = The angular velocity of the robot.

        Returns:
            torch.Tensor: The observation tensor."""

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
                target_heading_error.unsqueeze(dim=1),
                self.Leatherback.data.root_lin_vel_b[:, :2].unsqueeze(dim=1),
                self.Leatherback.data.root_ang_vel_w[:, -1].unsqueeze(dim=1),
                self._throttle_state[:,0].unsqueeze(dim=1),
                self._steering_state[:,0].unsqueeze(dim=1)
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """
        Computes the reward for the current state of the robot.

        The observation is given in the robot's frame. The task provides 3 elements:
        - The position of the object in the robot's frame. It is expressed as the distance between the robot and
            the target position, and the angle between the robot's heading and the target position.
        - The linear velocity of the robot in the robot's frame.
        - The angular velocity of the robot in the robot's frame.

        Angle measurements are converted to a cosine and a sine to avoid discontinuities in 0 and 2pi.
        This provides a continuous representation of the angle.

        The observation tensor is composed of the following elements:
        - self._task_data[:, 0]: The distance between the robot and the target position.
        - self._task_data[:, 1]: The cosine of the angle between the robot's heading and the target position.
        - self._task_data[:, 2]: The sine of the angle between the robot's heading and the target position.
        - self._task_data[:, 3]: The linear velocity of the robot along the x-axis.
        - self._task_data[:, 4]: The linear velocity of the robot along the y-axis.
        - self._task_data[:, 5]: The angular velocity of the robot.

        Args:
            current_state (torch.Tensor): The current state of the robot.
            actions (torch.Tensor): The actions taken by the robot.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            torch.Tensor: The reward for the current state of the robot."""


        # Boundary parameters
        self.maximum_robot_distance: float = 10.0
        position_tolerance: float = 0.01

        # Reward Scaling Coeficients
        position_exponential_reward_coeff: float = 0.25
        linear_velocity_min_value: float = 0.5
        linear_velocity_max_value: float = 2.0
        angular_velocity_min_value: float = 0.5
        angular_velocity_max_value: float = 20.0
        boundary_exponential_reward_coeff: float = 1.0
        position_weight: float = 1.0
        linear_velocity_weight: float = -0.05
        angular_velocity_weight: float = -0.05
        boundary_weight: float = -10.0
        rew_action_rate_scale = -0.12
        rew_joint_accel_scale = -2.5e-6

        # boundary distance
        boundary_dist = torch.abs(maximum_robot_distance - self._position_dist)
        # normed linear velocity
        linear_velocity = torch.norm(self.Leatherback.data.root_vel_w[:, :2], dim=-1)
        # normed angular velocity
        angular_velocity = torch.abs(self.Leatherback.data.root_vel_w[:, -1])

        # position reward
        position_rew = torch.exp(-self._position_dist / position_exponential_reward_coeff)
        # linear velocity reward
        linear_velocity_rew = linear_velocity - linear_velocity_min_value
        linear_velocity_rew[linear_velocity_rew < 0] = 0
        linear_velocity_rew[
            linear_velocity_rew > (linear_velocity_max_value - linear_velocity_min_value)
            ] = (linear_velocity_max_value - linear_velocity_min_value)
        # angular velocity reward
        angular_velocity_rew = angular_velocity - angular_velocity_min_value
        angular_velocity_rew[angular_velocity_rew < 0] = 0
        angular_velocity_rew[
            angular_velocity_rew > (angular_velocity_max_value - angular_velocity_min_value)
            ] = (angular_velocity_max_value - angular_velocity_min_value)
        # boundary rew
        boundary_rew = torch.exp(-boundary_dist / boundary_exponential_reward_coeff)

        # Checks if the goal is reached
        goal_is_reached = (self._position_dist < position_tolerance).int()
        self._goal_reached *= goal_is_reached  # if not set the value to 0
        self._goal_reached += goal_is_reached  # if it is add 1

        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        joint_accelerations = torch.sum(torch.square(self.Leatherback.data.joint_acc), dim=1)
        
        # Return the reward by combining the different components and adding the robot rewards
        return (
            position_rew * position_weight
            + linear_velocity_rew * linear_velocity_weight
            + angular_velocity_rew * angular_velocity_weight
            + boundary_rew * boundary_weight
            + action_rate * rew_action_rate_scale
            + joint_accelerations * rew_joint_accel_scale
        )


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates if the platforms should be killed or not.

        Returns:
            torch.Tensor: Whether the platforms should be killed or not."""
        
        reset_after_n_steps_in_tolerance: int = 100

        self._position_error = self._target_positions[:, :2] - self.Leatherback.data.root_pos_w[:, :2]
        self._position_dist = torch.norm(self._position_error, dim=-1)
        ones = torch.ones_like(self._goal_reached, dtype=torch.long)
        task_failed = torch.zeros_like(self._goal_reached, dtype=torch.long)
        task_failed = torch.where(
            self._position_dist > self.maximum_robot_distance,
            ones,
            task_failed,
        )

        task_completed = torch.zeros_like(self._goal_reached, dtype=torch.long)
        task_completed = torch.where(
            self._goal_reached > reset_after_n_steps_in_tolerance,
            ones,
            task_completed,
        )
    
        return task_failed, task_completed

    # TODO: Rework this
    # TODO: Don't forget to reset goal positions
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

        self._throttle_state[env_ids] = 0.0
        self._steering_state[env_ids] = 0.0

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