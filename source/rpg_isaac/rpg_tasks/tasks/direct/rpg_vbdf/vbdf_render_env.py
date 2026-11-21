"""Synthetic drone-detection images (VBDF Task 1) with the RPG Newton raytracer.

rpg_isaac counterpart of the course's Habitat renderer (flightrender ``drone_render_node``
+ ``collect_drone_images.py``): a camera drone (invisible) with the offboard_small
double-sphere lens looks at an actor drone placed in the arena stage; the raytracer's
shape-id output gives the actor's silhouette, hence an exact bounding box (the course
thresholded a second render against a white background).
"""

from __future__ import annotations

import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.utils.configclass import configclass

from rpg_assets import RPG_ASSETS_DIR
from rpg_assets.kolibri.kolibri_articulation_cfg import KOLIBRI_ROBOT_CFG
from rpg_tasks.tasks.direct.rpg_vbdf.vbdf_env_cfg import RPGVbdfHoverEnvCfg, RPGVbdfTrackingEnvCfg
from rpg_tasks.tasks.direct.rpg_vbdf.vbdf_hover_env import RPGVbdfHoverEnv
from rpg_tasks.tasks.direct.rpg_vbdf.vbdf_tracking_env import RPGVbdfTrackingEnv
from rpg_tasks.tasks.direct.rpg_vbdf.vbdf_tracking_cfg import (
    RPGVbdfBboxTrackingEnvCfg,
    RPGVbdfStateTrackingEnvCfg,
)
from rpg_tasks.tasks.direct.rpg_vbdf.vbdf_bbox_tracking_env import RPGVbdfBboxTrackingEnv
from rpg_tasks.tasks.direct.rpg_vbdf.vbdf_state_tracking_env import RPGVbdfStateTrackingEnv
from rpg_vision.camera_presets import offboard_small_ds_camera

STAGES_DIR = os.path.join(RPG_ASSETS_DIR, "meshes", "stages")
# background stages (course: ENV_ARENA / ENV_WINTI / ENV_EMPTY); winti from blender_objects/winti/*.blend
STAGE_USD = {
    "arena": os.path.join(STAGES_DIR, "arena.usd"),
    "winti": os.path.join(STAGES_DIR, "winti", "winti.usdc"),  # winti_1k_decimated.blend
    "winti_notrack": os.path.join(STAGES_DIR, "winti_notrack", "winti_notrack.usdc"),  # winti_1k_large_notrack.blend, decimated 0.05
    "empty": None,
}
# DroneRendererRandom sampling volume per stage [xmin, ymin, zmin, xmax, ymax, zmax] (world frame of the stage)
STAGE_BOX = {
    "arena": [-3.0, -3.0, 0.25, 3.0, 3.0, 2.5],  # matches physics world_bounds (6x6x2.25) so crash==render
    "winti": [-4.0, -6.0, 2.0, 16.0, 6.0, 4.0],  # the course's box (collect_drone_images.py defaults)
    "winti_notrack": [-4.0, -6.0, 2.0, 16.0, 6.0, 4.0],
    "empty": [-3.0, -3.0, 0.25, 3.0, 3.0, 2.5],
}


def resolve_stage(stage: str):
    """'arena' | 'winti' | 'empty' | path to a USD -> (usd path or None, default sampling box)."""
    if stage in STAGE_USD:
        return STAGE_USD[stage], STAGE_BOX[stage]
    if os.path.isfile(stage):
        return stage, STAGE_BOX["arena"]
    raise ValueError(f"unknown stage {stage!r}: {sorted(STAGE_USD)} or a USD path")
# the course's actor drone (blender_objects/drone/offboard_4S_vision.blend -> glb -> convert_mesh.py, see detection README)
OFFBOARD_4S_VISION_USD = os.path.join(RPG_ASSETS_DIR, "meshes", "offboard_4S_vision", "usd", "offboard_4S_vision.usd")

OFFBOARD_4S_VISION_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Actor",
    spawn=sim_utils.UsdFileCfg(
        usd_path=OFFBOARD_4S_VISION_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
)


def resolve_actor(actor_drone: str) -> RigidObjectCfg | ArticulationCfg:
    """'offboard_4S_vision' | 'kolibri' | path to a drone USD (scripts/tools/convert_mesh.py output)."""
    if actor_drone == "offboard_4S_vision":
        return OFFBOARD_4S_VISION_CFG
    if actor_drone == "kolibri":
        return KOLIBRI_ROBOT_CFG.replace(prim_path="/World/envs/env_.*/Actor")
    if os.path.isfile(actor_drone):
        return OFFBOARD_4S_VISION_CFG.replace(spawn=OFFBOARD_4S_VISION_CFG.spawn.replace(usd_path=actor_drone))
    raise ValueError(f"unknown actor_drone {actor_drone!r}: 'offboard_4S_vision', 'kolibri' or a USD path")


@configclass
class RPGVbdfRenderEnvCfg(RPGVbdfHoverEnvCfg):
    """Base env + onboard double-sphere camera + a visible actor drone (Task 1)."""

    image_width: int = 1280
    image_height: int = 720
    stage: str = "arena"  # "arena" | "winti" | "winti_notrack" | "empty" | USD path
    actor_drone: str = "offboard_4S_vision"  # the course actor | "kolibri" | path to a drone USD
    scene_num_envs: int = 4

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = self.scene_num_envs
        self.onboard_camera = offboard_small_ds_camera(self.image_width, self.image_height)
        self.robot.spawn.visible = False  # the camera drone must not see itself
        self.actor: RigidObjectCfg | ArticulationCfg = resolve_actor(self.actor_drone)


class RPGVbdfRenderEnv(RPGVbdfHoverEnv):
    """Poses in, images + actor masks out; no dynamics are stepped."""

    def _setup_scene(self):
        usd, _ = resolve_stage(self.cfg.stage)
        if usd is not None and os.path.exists(usd):
            cfg = sim_utils.UsdFileCfg(usd_path=usd)
            cfg.func("/World/stage", cfg, translation=(0.0, 0.0, 0.0))
            print(f"[INFO] Loaded stage: {usd}")
        if isinstance(self.cfg.actor, RigidObjectCfg):
            self.actor = RigidObject(self.cfg.actor)
            super()._setup_scene()
            self.scene.rigid_objects["actor"] = self.actor
        else:
            self.actor = Articulation(self.cfg.actor)
            super()._setup_scene()
            self.scene.articulations["actor"] = self.actor

    def _actor_shape_mask(self, seg: torch.Tensor) -> torch.Tensor:
        """Pixels whose shape belongs to an actor body (shape ids -> model bodies)."""
        if not hasattr(self, "_actor_shape_ids"):
            from isaaclab_newton.physics.newton_manager import NewtonManager

            model = NewtonManager._model
            labels = list(model.body_label)
            actor_bodies = {i for i, name in enumerate(labels) if "/Actor" in name or "Actor/" in name}
            shape_body = model.shape_body.numpy()
            ids = [i for i, b in enumerate(shape_body) if int(b) in actor_bodies]
            self._actor_shape_ids = torch.tensor(ids, device=self.device, dtype=seg.dtype)
            print(f"[RPGVbdfRender] actor: {len(actor_bodies)} bodies, {len(ids)} shapes")
        return torch.isin(seg, self._actor_shape_ids)

    @torch.no_grad()
    def render_poses(self, cam_pos, cam_att, actor_pos, actor_att):
        """cam/actor pos (N,3), att wxyz (N,4) -> albedo uint8 (N,H,W,3), actor mask (N,H,W) bool."""
        n = self.num_envs
        qs = self.quad_state
        qs.pos[:] = cam_pos
        qs.att[:] = cam_att / torch.linalg.norm(cam_att, dim=-1, keepdim=True)
        qs.vel[:] = 0.0
        qs.ome[:] = 0.0
        self.quadsim.refresh_caches()
        pose = torch.cat([actor_pos, actor_att[:, [1, 2, 3, 0]]], dim=-1).to(self.device, torch.float32)  # xyzw
        self.actor.write_root_pose_to_sim(pose)
        self.actor.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device))
        self.scene.write_data_to_sim()
        self.bridge.push_root_state()
        self._sync_render_state()
        self.sim.render()
        self.scene.update(dt=self.physics_dt)
        out = self._onboard_camera.data.output
        rgb = out["albedo"][..., :3]
        seg = out["instance_segmentation_fast"]
        if seg.dim() == 4:
            seg = seg[..., 0]
        return rgb, self._actor_shape_mask(seg)

    def _rs_weights(self, steps: int, height: int, shutter_ratio: float, rolling_ratio: float) -> torch.Tensor:
        """flightmare habitat_utils rs_tensor (top-to-bottom): per (sub-step, row) trapezoid
        weight, normalised so the per-pixel mean over sub-steps is an exposure-weighted average."""
        key = (steps, height, shutter_ratio, rolling_ratio)
        if getattr(self, "_rs_key", None) != key:
            import numpy as np

            dT = 1.0 / (steps - 1)
            offset = rolling_ratio / (rolling_ratio + shutter_ratio)
            duration = shutter_ratio / (rolling_ratio + shutter_ratio)
            T = np.arange(steps)[:, None] * dT
            t0 = np.arange(height)[None, :] / (height - 1) * offset
            t1 = t0 + duration
            w = np.zeros((steps, height))
            w = np.where((T > t0 - dT) & (T < t0), (T - (t0 - dT)) / dT, w)
            w = np.where((T >= t0) & (T <= t1), 1.0, w)
            w = np.where((T > t1) & (T < t1 + dT), (t1 + dT - T) / dT, w)
            w *= steps / w.sum(axis=0, keepdims=True)
            self._rs_key, self._rs_tensor = key, torch.as_tensor(w, dtype=torch.float32, device=self.device)
        return self._rs_tensor

    @staticmethod
    def _rotate_wxyz(att: torch.Tensor, omega: torch.Tensor, dt: float) -> torch.Tensor:
        """Rotate wxyz quaternions (N,4) by world-frame rates omega (N,3) over dt: exp(omega*dt/2) * att."""
        half = 0.5 * dt * omega
        ang = torch.linalg.norm(half, dim=-1, keepdim=True)
        axis = half / ang.clamp_min(1e-12)
        dq = torch.cat([torch.cos(ang), axis * torch.sin(ang)], dim=-1)
        w1, v1, w2, v2 = dq[:, :1], dq[:, 1:], att[:, :1], att[:, 1:]
        return torch.cat([w1 * w2 - (v1 * v2).sum(-1, keepdim=True), w1 * v2 + w2 * v1 + torch.cross(v1, v2, dim=-1)], dim=-1)

    @torch.no_grad()
    def render_poses_blurred(
        self,
        cam_pos,
        cam_att,
        actor_pos,
        actor_att,
        cam_vel=None,
        actor_vel=None,
        cam_omega=None,
        actor_omega=None,
        steps: int = 16,
        shutter_angle: float = 67.5,
        rolling_angle: float = 0.0,
        frame_dt: float = 1 / 30,
    ):
        """render_poses with the course's multi-sample shutter model (flightmare habitat_utils
        real_motionblur + rolling_shutter, config_drone_images.yaml defaults) — plus sub-step
        poses for the actor, so a fast actor smears relative to the background.

        Poses are end-of-exposure; velocities [m/s] and world-frame rates [rad/s] (all (N,3),
        None = static) are integrated backwards over the exposure window. ``frame_dt`` is the
        duration of 360 deg of shutter+rolling. Returns (rgb float (N,H,W,3) in [0,255], mask
        union (N,H,W) bool)."""
        n, dev = self.num_envs, self.device
        zeros = torch.zeros(n, 3, device=dev)
        cam_vel, actor_vel = (zeros if v is None else v for v in (cam_vel, actor_vel))
        cam_omega, actor_omega = (zeros if v is None else v for v in (cam_omega, actor_omega))
        sr, rr = shutter_angle / 360.0, rolling_angle / 360.0
        assert sr + rr > 0, "enable at least one of shutter_angle / rolling_angle"
        rs = self._rs_weights(steps, self.cfg.image_height, sr, rr) if rr > 0 else None
        tinterp = [(-(sr + rr) + (sr + rr) * j / (steps - 1)) * frame_dt for j in range(steps)]  # habitat tinterp
        acc, mask = 0.0, torch.zeros(n, self.cfg.image_height, self.cfg.image_width, dtype=torch.bool, device=dev)
        for j, t in enumerate(tinterp):
            rgb, m = self.render_poses(
                cam_pos + cam_vel * t,
                self._rotate_wxyz(cam_att, cam_omega, t),
                actor_pos + actor_vel * t,
                self._rotate_wxyz(actor_att, actor_omega, t),
            )
            acc = acc + (rgb.float() * rs[j][None, :, None, None] if rs is not None else rgb.float())
            mask |= m
        return acc / steps, mask


def _spawn_stage(stage: str) -> None:
    usd, _ = resolve_stage(stage)
    if usd is not None and os.path.exists(usd):
        cfg = sim_utils.UsdFileCfg(usd_path=usd)
        cfg.func("/World/stage", cfg, translation=(0.0, 0.0, 0.0))


class ActorVizMixin:
    """Adds a stage + a visible actor mesh that mirrors the virtual reference, so the
    Newton/kit viewers show both drones (the policy still sees whatever the env's
    observations contain)."""

    def _setup_scene(self):
        _spawn_stage(self.cfg.stage)
        self.actor = RigidObject(self.cfg.actor)
        super()._setup_scene()
        self.scene.rigid_objects["actor"] = self.actor

    def _advance_actor(self) -> None:
        super()._advance_actor()
        pose = torch.cat([self.other_pos, self.other_att[:, [1, 2, 3, 0]]], dim=-1).to(self.device, torch.float32)  # xyzw
        self.actor.write_root_pose_to_sim(pose)
        self.actor.write_root_velocity_to_sim(torch.zeros(self.num_envs, 6, device=self.device))
        self.scene.write_data_to_sim()


@configclass
class RPGVbdfTrackingVizEnvCfg(RPGVbdfTrackingEnvCfg):
    """The tracking task with a visible actor mesh + stage, for viewers / video recording."""

    stage: str = "arena"
    actor_drone: str = "offboard_4S_vision"

    def __post_init__(self):
        super().__post_init__()
        self.actor: RigidObjectCfg = resolve_actor(self.actor_drone)
        # fixed wide-angle side shot covering the whole flight box (Newton GL culls the
        # arena's back faces, so a camera outside the room looks straight in)
        self.viewer.eye = (0.0, -10.0, 5.5)
        self.viewer.lookat = (0.0, 0.0, 1.5)
        self.video_recorder.horiz_fov_deg = 90.0


class RPGVbdfTrackingVizEnv(ActorVizMixin, RPGVbdfTrackingEnv):
    pass


@configclass
class RPGVbdfHoverVizEnvCfg(RPGVbdfHoverEnvCfg):
    """Hover task with a background stage, for viewers / video recording."""

    stage: str = "arena"


class RPGVbdfHoverVizEnv(RPGVbdfHoverEnv):
    def _setup_scene(self):
        _spawn_stage(self.cfg.stage)
        super()._setup_scene()


# --- fresh Task-2 trackers (state-based + vision/bbox), Viz variants -------------------
# Both trackers share TrackingMixin (which supplies ``other_pos`` / ``other_att`` and the
# ``_advance_actor`` hook the ActorVizMixin mirrors), so a single mixin works for both and
# the observation layout is inherited UNCHANGED (SB = 25-dim, BB = 53-dim) so that the
# trained checkpoints load against the Viz env exactly as against the non-Viz env. The only
# difference is the added stage + visible actor mesh + fixed 3rd-person camera.

# Retuned 3rd-person fixed shot (see coordinator sign-off notes): slightly LOWER on z than
# the original (0,-10,5.5)->(0,0,1.5), and yawed a little toward the scene's +x ("right" as
# seen from a camera standing on the -y wall).
#
# v2 (render v2 clips): the previous (1.5,-12,3.2)->(0.5,0,1.2) still read as top-down-ish.
# - MORE DOWN / less bird's-eye: drop the eye z 3.2 -> 2.6 and raise the lookat z 1.2 -> 1.5,
#   so the view-ray drop over the ~12 m standoff shrinks from 2.0 m to ~1.1 m (a flatter,
#   more driver-level pitch instead of looking down at the room).
# - Yawed to the camera's RIGHT: for a camera on the -y wall facing +y with up=+z, the
#   camera's right is +x, so push the lookat x (1.9) past the eye x (1.6) to turn the view
#   gently toward +x.
# v3 (analysis commit 445577c "corrected raked 3rd-person camera"): camera up above the
# wall/roof line (z=4.0 > roof 2.5) and back (7.0 m past the -y wall), looking DOWN into the
# open 6x6 room at room-centre / mid-flight height (3,0,1.375) -> a RAKED three-quarter view
# (floor + walls readable, whole 6x6 room inside the 90-deg frustum). NOT the old near-flat
# sliver (1.6,-12,2.6)->(1.9,0,1.5, ~5.2-deg declination). Keep horiz_fov 90.
VIZ_EYE = (3.0, -7.0, 4.0)
VIZ_LOOKAT = (3.0, 0.0, 1.375)


def _offboard_pov_camera(tilt_deg: float = 15.0, width: int = 1280, height: int = 720):
    """The tracking drone's own double-sphere fisheye (POV) camera, mounted nose-forward
    with an ``tilt_deg`` upward tilt (the real drone tilts its camera 15 deg up, course
    sheet 3.1.1). ``offboard_small_ds_camera`` gives the identity "world"-convention mount
    (forward +X, up +Z); pitch +deg rotates +X toward +Z = up."""
    import torch

    from isaaclab.utils.math import quat_from_euler_xyz

    cam = offboard_small_ds_camera(width, height)
    # FIX (lane-2, 2026-09-15): a POSITIVE euler pitch here (quat_from_euler_xyz(0,+15,0) in
    # "world" convention, then converted world->opengl by Camera.__init__) actually points the
    # rendered optical axis 15 deg DOWN (verified empirically: quat_w_world -> optical axis
    # [0.966, 0, -0.259]). The course intent (alt_offset = desired_dist*sin(15), "tilted 15 deg
    # up") requires the optical axis to point UP ([0.966, 0, +0.259]), so the euler pitch sign
    # must be NEGATED. Keep the caller's `tilt_deg` positive (= "tilt up") and negate here.
    up_tilt = tuple(quat_from_euler_xyz(*torch.deg2rad(torch.tensor([0.0, -tilt_deg, 0.0]))).tolist())
    cam.offset = cam.offset.replace(rot=up_tilt)
    return cam


def _apply_tracking_viz_cfg(cfg, *, pov: bool) -> None:
    """Shared __post_init__ tail: visible actor + retuned 3rd-person shot (+ optional POV)."""
    cfg.actor = resolve_actor(cfg.actor_drone)
    cfg.viewer.eye = VIZ_EYE
    cfg.viewer.lookat = VIZ_LOOKAT
    cfg.video_recorder.horiz_fov_deg = 90.0
    if pov:
        cfg.onboard_camera = _offboard_pov_camera(cfg.pov_tilt_deg)
        # POV: hide the ego drone's own mesh so it doesn't self-occlude the fisheye camera
        # (the sibling fork sets robot.spawn.visible=False for exactly this).
        cfg.robot.spawn.visible = False


@configclass
class RPGVbdfStateTrackingVizEnvCfg(RPGVbdfStateTrackingEnvCfg):
    """State-based tracker with a visible actor mesh + stage, for viewers / video recording."""

    stage: str = "arena"
    actor_drone: str = "offboard_4S_vision"
    pov: bool = False
    pov_tilt_deg: float = 15.0

    def __post_init__(self):
        super().__post_init__()
        _apply_tracking_viz_cfg(self, pov=self.pov)


class RPGVbdfStateTrackingVizEnv(ActorVizMixin, RPGVbdfStateTrackingEnv):
    pass


@configclass
class RPGVbdfBboxTrackingVizEnvCfg(RPGVbdfBboxTrackingEnvCfg):
    """Vision (bbox) tracker with a visible actor mesh + stage, for viewers / video recording."""

    stage: str = "arena"
    actor_drone: str = "offboard_4S_vision"
    pov: bool = False
    pov_tilt_deg: float = 15.0

    def __post_init__(self):
        super().__post_init__()
        _apply_tracking_viz_cfg(self, pov=self.pov)


class RPGVbdfBboxTrackingVizEnv(ActorVizMixin, RPGVbdfBboxTrackingEnv):
    pass


