"""Ready-to-use RPG onboard cameras, one per renderer.

Which AOV a renderer emits for colour, and whether it needs per-frame camera
poses, follow from the renderer itself, so they are chosen together with it
rather than declared separately. Matching sim-level RTX settings live in
``rtx_render_presets.py``.
"""

from __future__ import annotations

import torch
from isaaclab_newton.renderers.newton_warp_renderer_cfg import NewtonWarpRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.renderers import RendererCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.math import quat_from_euler_xyz

from .onboard_camera_cfg import RPG_NEWTON_RENDERER_TYPE, OnboardCameraCfg

# Flightmare offboard-2.5mm Kolibri camera fitted to RTX's f-theta model.
# At 224x224 Kit maps the principal point to (112.42, 106.66) and preserves one
# isotropic radial pixel scale from this nominal calibration. The polynomial
# approximates the Double-Sphere calibration to <0.2 deg over a 200-degree FOV.
# Calibrated against fisheye_nominal_*, so it is resolution-independent.
KOLIBRI_RACING_SPAWN = sim_utils.FisheyeCameraCfg(
    focal_length=8.0,
    horizontal_aperture=20.955,
    clipping_range=(0.1, 1000.0),
    fisheye_max_fov=200.0,
    fisheye_nominal_width=960.0,
    fisheye_nominal_height=720.0,
    fisheye_optical_centre_x=481.8,
    fisheye_optical_centre_y=342.8229082615084,
    fisheye_polynomial_a=0.0,
    fisheye_polynomial_b=2.0687882000966075e-3,
    fisheye_polynomial_c=-6.016343263043169e-7,
    fisheye_polynomial_d=3.3464173774034826e-9,
    fisheye_polynomial_e=-5.922148104454601e-12,
    fisheye_polynomial_f=4.030538045985961e-15,
)

KOLIBRI_RACING_MOUNT = CameraCfg.OffsetCfg(
    pos=(0.0, 0.0, 0.0),
    rot=tuple(quat_from_euler_xyz(*torch.deg2rad(torch.tensor([0.0, -30.0, 0.0]))).tolist()),
    convention="world",
)
"""Racing installation: at the body origin, pitched 30 deg down.

The body frame is fixed project-wide: +X forward, +Z up; ``rot`` is intrinsic
XYZ roll, pitch, yaw as an xyzw quaternion.
"""

NEWTON_ALBEDO_RENDERER = NewtonWarpRendererCfg(
    # Routes camera construction to RPGNewtonWarpRenderer; the stock newton_warp
    # raytracer produces neither the calibrated fisheye rays nor the albedo
    # output RPG cameras rely on. See rpg_newton_renderer.py.
    renderer_type=RPG_NEWTON_RENDERER_TYPE,
    enable_textures=True,
    enable_shadows=False,
    enable_ambient_lighting=True,
    enable_backface_culling=True,
    max_distance=1000.0,
    # The task scene owns the lighting rig; albedo needs no fallback light.
    create_default_light=False,
)


def kolibri_racing_camera(renderer_cfg: RendererCfg, colour_aov: str) -> OnboardCameraCfg:
    """Return the Kolibri racing camera wired to one renderer.

    Args:
        renderer_cfg: Which renderer produces the pixels.
        colour_aov: The AOV that renderer emits for colour. It becomes the
            camera's only requested ``data_types`` entry and the source of
            ``observations["rgb"]``; request more AOVs by appending to
            ``data_types`` afterwards.
    """
    return OnboardCameraCfg(
        spawn=KOLIBRI_RACING_SPAWN,
        offset=KOLIBRI_RACING_MOUNT,
        width=224,
        height=224,
        data_types=[colour_aov],
        rgb_aov=colour_aov,
        renderer_cfg=renderer_cfg,
        # The Newton raytracer renders from render_data.camera_transforms, which
        # only Camera._update_poses() refreshes; RTX pays a per-frame pose read
        # it does not need. See CameraCfg.update_latest_camera_pose.
        update_latest_camera_pose=renderer_cfg.renderer_type == RPG_NEWTON_RENDERER_TYPE,
    )


#: RPG Newton/Warp raytracer, calibrated camera rays and unlit albedo. Kitless.
KOLIBRI_NEWTON_ALBEDO = kolibri_racing_camera(NEWTON_ALBEDO_RENDERER, "albedo")

#: Isaac RTX Minimal renderer with textured diffuse shading.
KOLIBRI_RTX_MINIMAL_TEXTURED = kolibri_racing_camera(IsaacRtxRendererCfg(), "simple_shading_diffuse_mdl")

#: Isaac RTX Minimal renderer with constant diffuse shading.
KOLIBRI_RTX_MINIMAL_CONSTANT = kolibri_racing_camera(IsaacRtxRendererCfg(), "simple_shading_constant_diffuse")

#: Isaac RTX Real-Time 2.0 with balanced DLSS ray reconstruction.
KOLIBRI_RTX_REALTIME_BALANCED = kolibri_racing_camera(IsaacRtxRendererCfg(), "rgb")

#: Isaac RTX path tracing with one sample per pixel and short paths.
KOLIBRI_RTX_PATH_TRACING_LOW_SAMPLE = kolibri_racing_camera(IsaacRtxRendererCfg(), "rgb")


# VBDF course camera drone ("offboard_small"): Kalibr double-sphere calibration
# flightrender/configs/camera_config/offboard_small/offboard_small_ds.yaml, 1280x720,
# camera along body x (r_BC = 0) as in the course's tracking env / Habitat renderer.
# (xi, alpha, fx, fy, cx, cy, calib_width, calib_height)
OFFBOARD_SMALL_DS = (-0.007441148905788614, 0.6138265280850498, 599.87676296874, 449.2911037186814, 632.1317557077697, 388.2472267118482, 1280.0, 720.0)

OFFBOARD_SMALL_MOUNT = CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world")  # xyzw identity


def offboard_small_ds_camera(width: int = 1280, height: int = 720, extra_aovs=("instance_segmentation_fast",)) -> OnboardCameraCfg:
    """The course camera drone's lens on the RPG Newton raytracer (albedo + shape ids)."""
    return OnboardCameraCfg(
        spawn=KOLIBRI_RACING_SPAWN,  # USD prim only; the rays come from double_sphere
        offset=OFFBOARD_SMALL_MOUNT,
        width=width,
        height=height,
        data_types=["albedo", *extra_aovs],
        rgb_aov="albedo",
        double_sphere=OFFBOARD_SMALL_DS,
        renderer_cfg=NEWTON_ALBEDO_RENDERER.replace(colorize_instance_segmentation=False),
        update_latest_camera_pose=True,
    )


#: 1280x720 double-sphere camera of the VBDF camera drone, RPG Newton raytracer.
OFFBOARD_SMALL_NEWTON_ALBEDO = offboard_small_ds_camera()
