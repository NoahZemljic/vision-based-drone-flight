import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

# ``renderer_type`` routing camera construction to the RPG Newton raytracer.
RPG_NEWTON_RENDERER_TYPE = "rpg_newton_warp"


@configclass
class OnboardCameraCfg(CameraCfg):
    """The RPG onboard camera.

    The one addition to IsaacLab's CameraCfg is :attr:`rgb_aov`,
    because renderers disagree about what to call rgb output.
    """

    prim_path: str = "/World/envs/env_.*/Robot/body/onboard_cam"
    offset: CameraCfg.OffsetCfg = CameraCfg.OffsetCfg(convention="world")
    spawn: sim_utils.PinholeCameraCfg | sim_utils.FisheyeCameraCfg = sim_utils.PinholeCameraCfg(
        focal_length=8.0, horizontal_aperture=20.955, clipping_range=(0.1, 1000.0)
    )
    data_types: list[str] = []
    width: int = 224
    height: int = 224

    rgb_aov: str = "rgb"
    """Which :attr:`data_types` entry is aliased onto ``observations["rgb"]``.

    The Newton raytracer emits ``albedo`` and RTX Minimal emits
    ``simple_shading_*``; policies read ``rgb`` whichever renderer produced it.
    """

    double_sphere: tuple[float, ...] | None = None
    """Double-sphere lens (Usenko et al. 2018) ``(xi, alpha, fx, fy, cx, cy, calib_width, calib_height)``.

    Kit has no double-sphere projection, so ``spawn`` stays a stock fisheye/pinhole
    prim; only the RPG Newton raytracer honours this field and casts exact
    double-sphere rays (intrinsics scaled from the calibration resolution to
    ``width``/``height``). ``None`` = use the ``spawn`` projection.
    """
