"""The RPG Newton warp raytracer (kit-less onboard cameras).

Extends IsaacLab's stock NewtonWarpRenderer with what vision training needs:

- calibrated polynomial f-theta fisheye rays, or exact double-sphere rays when the
  camera sets :attr:`OnboardCameraCfg.double_sphere`, selected per camera; pinhole cameras keep the stock ray generator, which
  is all upstream supports;
- textured-shape colors reset to white (Newton's default per-shape palette
  multiplies into textures and tints instances differently);
- unlit albedo output with a light-gray clear color;
- PhysX shadow-state sync in ``update_transforms`` (upstream rebinds instead
  of copying at num_envs=1, see MIGRATION_ISAACLAB_3.0.md section 8b).

Registered under its own factory backend key — the stock ``newton`` renderer
stays untouched.
"""

from __future__ import annotations

import math

import newton
import torch
import warp as wp
from isaaclab_newton.physics.newton_manager import NewtonManager
from isaaclab_newton.renderers.newton_warp_renderer import NewtonWarpRenderer, RenderData
from isaaclab_newton.renderers.newton_warp_renderer_cfg import NewtonWarpRendererCfg

from isaaclab.renderers import Renderer
from isaaclab.renderers import renderer as _renderer_factory_module
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.scene_data import SceneDataFormat
from isaaclab.sim import SimulationContext
from isaaclab.utils.warp.warp_math import convert_camera_frame_orientation_convention_wp


@wp.kernel(enable_backward=False)
def _compute_fisheye_polynomial_camera_rays(
    width: int,
    height: int,
    nominal_width: float,
    nominal_height: float,
    optical_center_x: float,
    optical_center_y: float,
    max_fov_radians: float,
    polynomial_a: float,
    polynomial_b: float,
    polynomial_c: float,
    polynomial_d: float,
    polynomial_e: float,
    polynomial_f: float,
    output: wp.array(dtype=wp.vec3f, ndim=4),
):
    """Generate OpenGL camera-space rays for Kit's polynomial f-theta model."""
    camera_index, py, px = wp.tid()
    optical_center_output_x = optical_center_x * float(width) / nominal_width
    optical_center_output_y = optical_center_y * float(height) / nominal_height
    radial_scale = wp.max(nominal_width / float(width), nominal_height / float(height))
    delta_x = (float(px) + 0.5 - optical_center_output_x) * radial_scale
    delta_y_down = (float(py) + 0.5 - optical_center_output_y) * radial_scale
    radius = wp.sqrt(delta_x * delta_x + delta_y_down * delta_y_down)
    theta = polynomial_a + radius * (
        polynomial_b
        + radius * (polynomial_c + radius * (polynomial_d + radius * (polynomial_e + radius * polynomial_f)))
    )

    direction = wp.vec3f(0.0, 0.0, -1.0)
    if radius > 1.0e-8:
        radial_sine = wp.sin(theta) / radius
        direction = wp.vec3f(delta_x * radial_sine, -delta_y_down * radial_sine, -wp.cos(theta))
    if theta < 0.0 or theta > 0.5 * max_fov_radians:
        direction = wp.vec3f(0.0)

    output[camera_index, py, px, 0] = wp.vec3f(0.0)
    output[camera_index, py, px, 1] = direction


@wp.kernel(enable_backward=False)
def _compute_double_sphere_camera_rays(
    width: int,
    height: int,
    xi: float,
    alpha: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    output: wp.array(dtype=wp.vec3f, ndim=4),
):
    """OpenGL camera-space rays of the double-sphere model (Usenko, Demmel, Cremers 2018).

    Unprojection: m = ((u-cx)/fx, (v-cy)/fy), r2 = |m|^2,
    mz = (1 - alpha^2 r2) / (alpha sqrt(1 - (2 alpha - 1) r2) + 1 - alpha),
    d = (mz xi + sqrt(mz^2 + (1 - xi^2) r2)) / (mz^2 + r2) * (mx, my, mz) - (0, 0, xi).
    Pixels outside the valid domain (r2 > 1/(2 alpha - 1) for alpha > 0.5) get a zero ray.
    """
    camera_index, py, px = wp.tid()
    mx = (float(px) + 0.5 - cx) / fx
    my = (float(py) + 0.5 - cy) / fy
    r2 = mx * mx + my * my
    direction = wp.vec3f(0.0)
    valid = True
    if alpha > 0.5:
        if r2 > 1.0 / (2.0 * alpha - 1.0):
            valid = False
    if valid:
        root = 1.0 - (2.0 * alpha - 1.0) * r2
        if root < 0.0:
            root = 0.0
        mz = (1.0 - alpha * alpha * r2) / (alpha * wp.sqrt(root) + 1.0 - alpha)
        k = (mz * xi + wp.sqrt(mz * mz + (1.0 - xi * xi) * r2)) / (mz * mz + r2)
        dx = k * mx
        dy = k * my
        dz = k * mz - xi
        n = wp.sqrt(dx * dx + dy * dy + dz * dz)
        # optical (x right, y down, z forward) -> OpenGL (x right, y up, -z forward)
        direction = wp.vec3f(dx / n, -dy / n, -dz / n)
    output[camera_index, py, px, 0] = wp.vec3f(0.0)
    output[camera_index, py, px, 1] = direction


class RPGRenderData(RenderData):
    """Newton camera data with projection-matched rays and persistent pose buffers."""

    def __init__(self, newton_sensor: newton.sensors.SensorTiledCamera, spec: CameraRenderSpec):
        super().__init__(newton_sensor, spec)
        self.camera_spawn_cfg = spec.cfg.spawn
        self.double_sphere = getattr(spec.cfg, "double_sphere", None)
        self.converted_orientations = None

    def update(self, positions, orientations, intrinsics) -> None:
        if self.converted_orientations is None:
            self.converted_orientations = wp.empty_like(orientations)
        convert_camera_frame_orientation_convention_wp(
            src=orientations,
            dst=self.converted_orientations,
            origin="world",
            target="opengl",
            device=self.newton_sensor.model.device,
        )

        if self.camera_transforms is None:
            self.camera_transforms = wp.empty(
                (1, self.newton_sensor.model.world_count),
                dtype=wp.transformf,
                device=self.newton_sensor.model.device,
            )
        wp.launch(
            RenderData._update_transforms,
            self.newton_sensor.model.world_count,
            [positions, self.converted_orientations, self.camera_transforms],
            device=self.newton_sensor.model.device,
        )

        if self.camera_rays is not None:
            return
        if self.double_sphere is not None:
            xi, alpha, fx, fy, cx, cy, calib_w, calib_h = [float(v) for v in self.double_sphere]
            sx = self.width / calib_w
            sy = self.height / calib_h
            self.camera_rays = wp.empty(
                (self.num_cameras, self.height, self.width, 2),
                dtype=wp.vec3f,
                device=self.newton_sensor.model.device,
            )
            wp.launch(
                kernel=_compute_double_sphere_camera_rays,
                dim=(self.num_cameras, self.height, self.width),
                inputs=[
                    self.width,
                    self.height,
                    xi,
                    alpha,
                    fx * sx,
                    fy * sy,
                    cx * sx,
                    cy * sy,
                    self.camera_rays,
                ],
                device=self.newton_sensor.model.device,
            )
            return
        if getattr(self.camera_spawn_cfg, "projection_type", "pinhole") == "fisheyePolynomial":
            self.camera_rays = wp.empty(
                (self.num_cameras, self.height, self.width, 2),
                dtype=wp.vec3f,
                device=self.newton_sensor.model.device,
            )
            wp.launch(
                kernel=_compute_fisheye_polynomial_camera_rays,
                dim=(self.num_cameras, self.height, self.width),
                inputs=[
                    self.width,
                    self.height,
                    self.camera_spawn_cfg.fisheye_nominal_width,
                    self.camera_spawn_cfg.fisheye_nominal_height,
                    self.camera_spawn_cfg.fisheye_optical_centre_x,
                    self.camera_spawn_cfg.fisheye_optical_centre_y,
                    math.radians(self.camera_spawn_cfg.fisheye_max_fov),
                    self.camera_spawn_cfg.fisheye_polynomial_a,
                    self.camera_spawn_cfg.fisheye_polynomial_b,
                    self.camera_spawn_cfg.fisheye_polynomial_c,
                    self.camera_spawn_cfg.fisheye_polynomial_d,
                    self.camera_spawn_cfg.fisheye_polynomial_e,
                    self.camera_spawn_cfg.fisheye_polynomial_f,
                    self.camera_rays,
                ],
                device=self.newton_sensor.model.device,
            )
            return

        first_focal_length = intrinsics.torch[:, 1, 1][0:1]
        fov_radians_all = 2.0 * torch.atan(self.height / (2.0 * first_focal_length))
        fov_warp = wp.from_torch(fov_radians_all, dtype=wp.float32)
        self.camera_rays = self.newton_sensor.utils.compute_pinhole_camera_rays(self.width, self.height, fov_warp)


class RPGNewtonWarpRenderer(NewtonWarpRenderer):
    """Newton warp raytracer for RPG onboard cameras.

    Supports both projections: the camera's ``spawn`` type selects which,
    and :class:`RPGRenderData` generates polynomial f-theta or pinhole rays
    accordingly.
    """

    cfg: NewtonWarpRendererCfg

    def create_render_data(self, spec: CameraRenderSpec) -> RPGRenderData:
        render_data = RPGRenderData(self.newton_sensor, spec)
        # Newton's default PIXEL_PRIORITY traversal spreads a warp's threads
        # across distant pixels; TILED keeps each warp inside one 16x8 tile so
        # neighboring rays walk the same BVH nodes. Byte-identical albedo
        # output, measured ~40% faster at 224x224. TILED asserts tile-aligned
        # image dimensions, so keep the stock order otherwise.
        from newton._src.sensors.warp_raytrace.types import RenderOrder

        render_config = self.newton_sensor.render_config
        if render_data.width % 16 == 0 and render_data.height % 8 == 0:
            render_config.render_order = int(RenderOrder.TILED)
            render_config.tile_width = 16
            render_config.tile_height = 8
        else:
            render_config.render_order = int(RenderOrder.PIXEL_PRIORITY)
        return render_data

    def initialize(self) -> None:
        super().initialize()
        model = self.newton_sensor.model
        if model.shape_count == 0:
            return
        # Newton multiplies textures by its per-shape default palette color,
        # tinting each instance differently. Textured meshes must stay untinted.
        import numpy as np

        shape_colors = model.shape_color.numpy()
        whitened = 0
        for index, source in enumerate(model.shape_source):
            if source is not None and getattr(source, "texture", None) is not None:
                if not np.allclose(shape_colors[index], 1.0):
                    shape_colors[index] = (1.0, 1.0, 1.0)
                    whitened += 1
        if whitened:
            model.shape_color.assign(shape_colors)
            print(f"[INFO] RPG Newton renderer: reset {whitened} textured shape colors to white")

    def update_transforms(self) -> None:
        """Sync PhysX shadow state without relying on an IsaacLab source patch."""
        sim = SimulationContext.instance()
        sim.physics_manager.forward()
        scene_data_provider = NewtonManager.get_scene_data_provider()
        if NewtonManager._backend_is_newton(scene_data_provider):
            return
        NewtonManager._ensure_visualization_model()
        if NewtonManager._state_0 is None or NewtonManager._model is None or NewtonManager._state_0.body_q is None:
            return
        if NewtonManager._scene_data is None:
            NewtonManager._scene_data = SceneDataFormat.Transform()
        if NewtonManager._scene_data_mapping is None:
            body_paths = NewtonManager._resolve_scene_data_body_paths(
                list(NewtonManager._model.body_label), scene_data_provider.usd_stage
            )
            NewtonManager._scene_data_mapping = scene_data_provider.create_mapping(body_paths)
        NewtonManager._scene_data.transforms = NewtonManager._state_0.body_q
        scene_data_provider.get_transforms(
            NewtonManager._scene_data,
            mapping=NewtonManager._scene_data_mapping,
            allow_passthrough=False,
        )

    def render(self, render_data: RPGRenderData) -> None:
        newton_state: newton.State = NewtonManager.get_state()
        if self.newton_sensor.model.shape_count > 0:
            newton.geometry.refit_bvh_shape(self.newton_sensor.model, newton_state)
        self.newton_sensor.update(
            newton_state,
            render_data.camera_transforms,
            render_data.camera_rays,
            color_image=render_data.outputs.color_image,
            hdr_color_image=render_data.outputs.hdr_color_image,
            albedo_image=render_data.outputs.albedo_image,
            depth_image=render_data.outputs.depth_image,
            normal_image=render_data.outputs.normals_image,
            shape_index_image=render_data.outputs.instance_segmentation_image,
            clear_data=newton.sensors.SensorTiledCamera.ClearData(clear_color=0xFFEEEEEE, clear_albedo=0xFFEEEEEE),
        )
        if render_data.ppisp_pipeline is not None:
            render_data.ppisp_pipeline.apply(render_data._ppisp_hdr_source, render_data._ppisp_rgba_dest)


# Register under a dedicated factory backend key. A profile's
# renderer_cfg.renderer_type carries this string, which is what routes camera
# construction here instead of to the stock "newton" registration. Pre-seeding
# the registry also stops the factory from attempting to import a nonexistent
# isaaclab_rpg_newton package. The scene-data requirements table is keyed by the
# same string — register there too, or the newton-model requirement is
# silently skipped for this renderer.
from isaaclab.physics import scene_data_requirements as _scene_data_requirements

_renderer_factory_module._RENDERER_TYPE_TO_BACKEND["rpg_newton_warp"] = "rpg_newton"
_scene_data_requirements._RENDERER_REQUIREMENTS["rpg_newton_warp"] = _scene_data_requirements._RENDERER_REQUIREMENTS[
    "newton_warp"
]
Renderer.register("rpg_newton", RPGNewtonWarpRenderer)
