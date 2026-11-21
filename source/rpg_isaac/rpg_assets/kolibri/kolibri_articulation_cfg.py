from __future__ import annotations

import os
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from rpg_assets import RPG_ASSETS_DIR

##
# Configuration
##

# UrdfConverter 2.3 layout: a root .usd over configuration/ sublayers. The
# payload-based .usda the 3.x importer emits segfaults Newton's USD importer at
# launch, so that format is not shipped for any airframe.
_KOLIBRI_USD = os.path.abspath(os.path.join(RPG_ASSETS_DIR, "meshes/kolibri/usd/high_def/kolibri.usd"))

KOLIBRI_ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Kolibri",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_KOLIBRI_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "fr_joint": 200.0,
            "bl_joint": 200.0,
            "br_joint": 200.0,
            "fl_joint": 200.0,
        },
    ),
    # actuators={
    #     "dummy": ImplicitActuatorCfg(
    #         joint_names_expr=[".*"],
    #         stiffness=0.0,
    #         damping=0.0,
    #     ),
    # },
    actuators={},
)
