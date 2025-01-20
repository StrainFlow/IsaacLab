# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple traffic cone marker."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.utils.assets import REPO_ROOT_PATH

##
# Configuration
##

CONE_CFG = VisualizationMarkersCfg(
    prim_path="/World/Visuals/Cones",
    markers={
        "marker1": sim_utils.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)
"""Configuration for a simple traffic cone marker"""
