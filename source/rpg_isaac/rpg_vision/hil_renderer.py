"""HIL camera renderer.

Drives the onboard camera from external poses (mocap/AgiSim) or replayed
traces, without advancing vehicle dynamics.
"""

from __future__ import annotations

import time

import torch
import warp as wp


@wp.kernel(enable_backward=False)
def _fused_camera_transform_kernel(
    body_position: wp.vec3f,
    body_orientation: wp.quatf,  # body pose, world frame, xyzw; need not be normalized
    mount_position: wp.vec3f,
    mount_opengl: wp.quatf,  # camera mount composed with the world->OpenGL convention
    out_transforms: wp.array(dtype=wp.transformf, ndim=2),
):
    q = wp.normalize(body_orientation)
    position = body_position + wp.quat_rotate(q, mount_position)
    out_transforms[0, 0] = wp.transformf(position, q * mount_opengl)


@wp.kernel(enable_backward=False)
def _fused_camera_transform_kernel_buffered(
    # Same math as above, but the pose comes from device buffers so a CUDA
    # graph capture of this launch stays valid for every subsequent pose.
    body_position: wp.array(dtype=wp.vec3f),
    body_orientation: wp.array(dtype=wp.quatf),  # world frame, xyzw; need not be normalized
    mount_position: wp.vec3f,
    mount_opengl: wp.quatf,
    out_transforms: wp.array(dtype=wp.transformf, ndim=2),
):
    q = wp.normalize(body_orientation[0])
    position = body_position[0] + wp.quat_rotate(q, mount_position)
    out_transforms[0, 0] = wp.transformf(position, q * mount_opengl)


class HILCameraRenderer:
    """Render the onboard camera at externally supplied vehicle poses.

    The default ``robot`` mode mirrors the complete vehicle state and renders
    from the attached onboard camera, matching student training. ``barebone``
    drives the Newton raster directly from the body pose.
    """

    def __init__(
        self,
        env,
        pose_sync_mode: str = "robot",
    ):
        self.env = env.unwrapped
        self.camera = self.env._onboard_camera
        if self.camera is None:
            raise RuntimeError("The environment has no onboard camera")
        if pose_sync_mode not in ("robot", "barebone"):
            raise ValueError("pose_sync_mode must be 'robot' or 'barebone'")
        if pose_sync_mode == "robot" and not self.env.using_agisim:
            raise ValueError("robot pose synchronization requires an AgiSim backend")
        self.pose_sync_mode = pose_sync_mode
        self.device = torch.device(self.env.device)
        self.dtype = self.env.dtype

        offset = self.env.cfg.onboard_camera.offset
        if offset.convention != "world":
            raise ValueError(
                f"HIL rendering expects a camera offset in the 'world' convention; got {offset.convention!r}"
            )
        self._mount_position = torch.tensor([offset.pos], dtype=torch.float32, device=self.device)
        self._mount_orientation = torch.tensor([offset.rot], dtype=torch.float32, device=self.device)
        if pose_sync_mode == "barebone":
            self._setup_barebone()

    def _setup_barebone(self) -> None:
        """One fused kernel: body pose -> renderer camera transform, then raster.

        Skips the quad-state write, root-state push, forward kinematics, and the
        Camera sensor machinery entirely. Nothing but the camera transform is
        advanced, so this is only valid when every other rendered body is static.
        """
        from rpg_vision.rpg_newton_renderer import RPGNewtonWarpRenderer

        from isaaclab.utils.warp.warp_math import convert_camera_frame_orientation_convention_wp

        renderer = self.camera._renderer
        render_data = self.camera._render_data
        if not isinstance(renderer, RPGNewtonWarpRenderer):
            raise ValueError("barebone pose synchronization requires the RPG Newton renderer")
        # Populate camera_rays/camera_transforms once through the stock path.
        seed_pos = wp.from_torch(torch.zeros((1, 3), dtype=torch.float32, device=self.device), dtype=wp.vec3f)
        seed_ori = wp.from_torch(
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=self.device), dtype=wp.quatf
        )
        render_data.update(seed_pos, seed_ori, None)
        # Fold the world->OpenGL convention into the mount quaternion by pushing
        # the mount through the stock conversion routine (right-multiplication).
        mount_wp = wp.from_torch(self._mount_orientation.contiguous(), dtype=wp.quatf)
        converted = wp.empty_like(mount_wp)
        convert_camera_frame_orientation_convention_wp(
            src=mount_wp, dst=converted, origin="world", target="opengl", device=str(self.device)
        )
        mount_opengl = wp.to_torch(converted).cpu().numpy()[0]
        self._bb_mount_opengl = wp.quatf(*(float(x) for x in mount_opengl))
        self._bb_mount_position = wp.vec3f(*(float(x) for x in self._mount_position[0].tolist()))
        self._bb_render_data = render_data
        self._bb_renderer = renderer
        self._bb_rgb = self.camera.data.output["rgb"].torch
        # CUDA-graph state: pose staging buffers the captured graph reads from.
        # The graph itself is captured lazily on the first render, after the
        # megakernel and every render buffer have been warmed through the
        # ordinary submit path.
        self._bb_pose_position = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        self._bb_pose_orientation = torch.zeros((1, 4), dtype=torch.float32, device=self.device)
        self._bb_pose_position_wp = wp.from_torch(self._bb_pose_position, dtype=wp.vec3f)
        self._bb_pose_orientation_wp = wp.from_torch(self._bb_pose_orientation, dtype=wp.quatf)
        self._bb_graph = None
        self._bb_graph_attempted = False

    def _bb_submit(self) -> None:
        """Enqueue the pose kernel and the raster; the captured graph replays this."""
        wp.launch(
            _fused_camera_transform_kernel_buffered,
            dim=1,
            inputs=[
                self._bb_pose_position_wp,
                self._bb_pose_orientation_wp,
                self._bb_mount_position,
                self._bb_mount_opengl,
            ],
            outputs=[self._bb_render_data.camera_transforms],
            device=str(self.device),
        )
        self._bb_renderer.render(self._bb_render_data)

    def _bb_ensure_graph(self) -> bool:
        """Capture the per-frame GPU work into a CUDA graph once; False if unsupported.

        Replaying one graph instead of re-submitting the megakernel drops the
        CPU-side cost per frame from milliseconds to microseconds, which both
        shortens the frame and frees the GIL for the ROS state-subscriber
        thread while the GPU renders.
        """
        if self._bb_graph is not None:
            return True
        if self._bb_graph_attempted or self.device.type != "cuda":
            return False
        self._bb_graph_attempted = True
        try:
            self._bb_submit()  # warm every kernel and buffer outside capture
            wp.synchronize_device(str(self.device))
            with wp.ScopedCapture(device=str(self.device)) as capture:
                self._bb_submit()
            self._bb_graph = capture.graph
            return True
        except Exception as exc:  # noqa: BLE001 - any capture failure falls back
            print(f"[WARN] HIL renderer: CUDA graph capture failed ({exc}); using per-frame launches", flush=True)
            self._bb_graph = None
            return False

    def _render_barebone(self, body_position, body_orientation_xyzw, timings, profile):
        start = time.perf_counter()
        position = torch.as_tensor(body_position, device=self.device, dtype=torch.float32).reshape(1, 3)
        orientation = torch.as_tensor(body_orientation_xyzw, device=self.device, dtype=torch.float32).reshape(1, 4)
        self._bb_pose_position.copy_(position)
        self._bb_pose_orientation.copy_(orientation)
        self._profile_sync(profile)
        timings["pose_sync"] = time.perf_counter() - start
        start = time.perf_counter()
        if self._bb_ensure_graph():
            wp.capture_launch(self._bb_graph)
        else:
            self._bb_submit()
        wp.synchronize_device(str(self.device))
        timings["render"] = time.perf_counter() - start
        timings["camera"] = 0.0
        return self._bb_rgb, timings

    def render(
        self,
        body_position: torch.Tensor,
        body_orientation_xyzw: torch.Tensor,
        body_velocity: torch.Tensor | None = None,
        body_omega: torch.Tensor | None = None,
        *,
        profile: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Render RGB at an externally supplied body pose.

        Args:
            body_position: Body world position [m], shape ``[1, 3]``.
            body_orientation_xyzw: Body world orientation, shape ``[1, 4]``.
            body_velocity: Optional world linear velocity [m/s], shape ``[1, 3]``.
            body_omega: Optional body angular velocity [rad/s], shape ``[1, 3]``.
            profile: Whether to synchronize CUDA around phases for timing.

        Returns:
            A CUDA RGB tensor and phase durations in seconds.
        """
        timings: dict[str, float] = {}
        if self.pose_sync_mode == "barebone":
            return self._render_barebone(body_position, body_orientation_xyzw, timings, profile)

        start = time.perf_counter()
        position = body_position.to(device=self.device, dtype=torch.float32).reshape(1, 3)
        orientation = body_orientation_xyzw.to(device=self.device, dtype=torch.float32).reshape(1, 4)
        orientation = torch.nn.functional.normalize(orientation, dim=1)
        timings["input_prepare"] = time.perf_counter() - start

        self._profile_sync(profile)
        start = time.perf_counter()
        quad_state = self.env.quad_state
        quad_state.pos.copy_(position.to(dtype=self.dtype))
        quad_state.att.copy_(orientation[:, [3, 0, 1, 2]].to(dtype=self.dtype))
        if body_velocity is not None:
            quad_state.vel.copy_(body_velocity.to(device=self.device, dtype=self.dtype).reshape(1, 3))
        if body_omega is not None:
            quad_state.ome.copy_(body_omega.to(device=self.device, dtype=self.dtype).reshape(1, 3))
        self.env.quadsim.refresh_caches()
        self.env.bridge.push_root_state()
        self.env._sync_render_state()
        self._profile_sync(profile)
        timings["pose_sync"] = time.perf_counter() - start

        start = time.perf_counter()
        self.env.sim.render()
        self._profile_sync(profile)
        timings["render"] = time.perf_counter() - start

        start = time.perf_counter()
        self.camera.update(dt=self.env.cfg.ctrl_dt)
        timings["sensor_update"] = time.perf_counter() - start
        output_start = time.perf_counter()
        rgb = self.camera.data.output["rgb"][..., :3]
        timings["camera_output"] = time.perf_counter() - output_start
        self._profile_sync(profile)
        timings["camera"] = time.perf_counter() - start
        return rgb, timings

    def _profile_sync(self, enabled: bool) -> None:
        if enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
