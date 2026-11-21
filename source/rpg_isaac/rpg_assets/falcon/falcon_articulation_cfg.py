"""Scene asset for the Falcon quadrotor.

The Falcon USD is generated from ``meshes/falcon/urdf/falcon.urdf`` (see
``meshes/falcon/usd_cfg/high_def/config.yaml`` for the converter settings) and
mirrors the Kolibri bundle: floating base, articulation root on ``body``, four
free-spinning propeller links, convex-hull colliders, and force-type position
drives with 100.0 / 1.0 gains.

Both bundles use the UrdfConverter 2.3 layout — a root ``.usd`` over
``configuration/`` sublayers — because the payload-based ``.usda`` the 3.x
importer emits segfaults Newton's USD importer at launch.

The bundle is binary and lives in the ``rpg_isaac_assets`` submodule, so when it
has not been fetched (``git lfs pull``) this module falls back to the Kolibri
bundle uniformly scaled by the Falcon/Kolibri arm-length ratio. The fallback
only affects what is drawn (and, in engine-coupled mode, the body properties the
engine starts from). In the ``agisim`` backend used by the racing task, rpg's own
Warp integrator owns the dynamics and gate/world collisions are analytic, so the
mesh is purely cosmetic there.
"""

from __future__ import annotations

import os
import warnings

from rpg_assets import RPG_ASSETS_DIR

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

# Falcon arm 0.1122 m (hypot(0.072, 0.086)) / Kolibri arm 0.058 m.
_KOLIBRI_TO_FALCON_SCALE = 1.93

_FALCON_USD = os.path.abspath(os.path.join(RPG_ASSETS_DIR, "meshes/falcon/usd/high_def/falcon.usd"))
_KOLIBRI_USD = os.path.abspath(os.path.join(RPG_ASSETS_DIR, "meshes/kolibri/usd/high_def/kolibri.usd"))

if os.path.isfile(_FALCON_USD):
    _USD_PATH = _FALCON_USD
    _USD_SCALE = None
else:
    warnings.warn(
        f"Falcon USD not found at {_FALCON_USD}; falling back to the Kolibri bundle scaled by"
        f" {_KOLIBRI_TO_FALCON_SCALE}. Fetch the rpg_isaac_assets submodule (git lfs pull), or"
        " regenerate the bundle from meshes/falcon/urdf/falcon.urdf with the settings recorded in"
        " meshes/falcon/usd_cfg/high_def/config.yaml.",
        stacklevel=2,
    )
    _USD_PATH = _KOLIBRI_USD
    _USD_SCALE = (_KOLIBRI_TO_FALCON_SCALE,) * 3

FALCON_ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Falcon",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        scale=_USD_SCALE,
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
    actuators={},
)
