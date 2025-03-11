# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple waypoint marker."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.utils.assets import REPO_ROOT_PATH

##
# Configuration
##

CONE_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{REPO_ROOT_PATH}/source/assets/robots/traffic_cone.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        )
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
)
"""Configuration for a simple waypoint marker"""

# prim_utils.create_prim("/World/envs/env_0/Obstacles", "Xform")
 
#         rigid_objects = {}
#         low, high = 5, 15
 
#         for i in range(self._task_cfg.max_num_vis_obstacles):
 
#             position = low + (high - low) * torch.rand(3)
#             position[2] = 0.5
 
#             rigid_objects[f"obstacle_{i}"] = RigidObjectCfg(
#                 prim_path=f"/World/envs/env_.*/Obstacles/cylinder_{i}",
#                 spawn=sim_utils.CylinderCfg(
#                     radius=self._task_cfg.obstacle_radius,
#                     height=self._task_cfg.obstacles_height,
#                     rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
#                     mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
#                     collision_props=sim_utils.CollisionPropertiesCfg(),
#                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
#                 ),
#                 init_state=RigidObjectCfg.InitialStateCfg(pos=position),
#             )
 
#         obstacles_cfg = RigidObjectCollectionCfg(rigid_objects=rigid_objects)
 
#         self.obstacles = RigidObjectCollection(obstacles_cfg)

    # def register_sensors(self) -> None:
    #     filters = [f"/World/envs/env_.*/Obstacles/cylinder_{i}" for i in range(self._task_cfg.max_num_vis_obstacles)]
    #     self._robot.activateSensors("contacts", filters)
    #     self._robot.register_sensors()