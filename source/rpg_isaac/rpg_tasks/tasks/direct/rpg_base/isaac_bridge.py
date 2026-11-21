"""Thin adapter between the (framework-agnostic) rpg QuadrotorSimulator and IsaacLab.

All reads/writes that cross the simulator <-> engine boundary live here:

- :meth:`IsaacBridge.pull_root_state` — engine root buffers -> quad_state
  (engine-coupled mode, every substep)
- :meth:`IsaacBridge.push_root_state` — quad_state -> engine
  (resets; agisim render/sensor sync)
- :meth:`IsaacBridge.write_wrench` — quad_state.force_torque -> engine wrench
  input (engine-coupled mode, every substep)
- :meth:`IsaacBridge.push_body_params` — randomized mass/inertia batches ->
  engine body properties (engine-coupled mode, on domain randomization)
"""

import torch
import warp as wp


class IsaacBridge:
    """Binds an rpg :class:`QuadState` to an IsaacLab articulation.

    Root-state binding (engine-coupled mode): IsaacLab 3.0 exposes
    ``root_link_pose_w`` / ``root_com_vel_w`` over pointer-stable GPU buffers
    (device pointer fixed across steps; Newton rebinds only on a full
    ``PHYSICS_READY`` reset), so a single bind at construction stays valid —
    each pull rewrites the same memory in place and the captured ingest graph
    reads fresh data on replay.
    """

    def __init__(self, robot, quad_state, body_id, param_server=None):
        self.robot = robot
        self.quad_state = quad_state
        self.param_server = param_server
        self.body_id = body_id
        self._wrench_graph = None
        self._torch_physx_wrench_view = None
        self._push_root_kernel = None
        self._wp_state = None
        self._root_pose_wp = None
        self._root_vel_wp = None

        if not quad_state.using_agisim:
            data = robot.data
            quad_state.bind_external_root_state(
                data.root_link_pose_w.torch, data.root_com_vel_w.torch
            )
            # Isaac Sim 5's dense Warp-fronted articulation wrench submission
            # is substantially slower than the Torch-fronted submission for
            # this workload.  Reuse the same PhysX backend with a Torch
            # frontend, while retaining Warp for RPG's captured scatter.
            if type(robot).__module__.startswith("isaaclab_physx"):
                self._setup_torch_physx_wrench_view()
            self._capture_wrench_graph()

    def _setup_torch_physx_wrench_view(self) -> None:
        """Create a Torch frontend over the existing PhysX articulation backend.

        IsaacLab 3.0 owns the normal Warp-fronted ``root_view``.  Constructing a
        second PhysX backend view after simulation start is rejected by PhysX,
        so this lightweight view deliberately shares that same backend and only
        changes how the force/torque tensor descriptors are produced.

        The wrench storage remains Warp-owned.  ``wp.to_torch`` creates cached
        zero-copy tensor views; no force or torque data is copied per substep.
        """
        from omni.physics.tensors.api import ArticulationView
        from omni.physics.tensors.frontend_torch import FrontendTorch

        root_view = self.robot.root_view
        backend = getattr(root_view, "_backend", None)
        if backend is None:
            raise RuntimeError("PhysX articulation root view has no initialized backend")

        device = torch.device(str(self.quad_state.device))
        if device.type != "cuda":
            raise RuntimeError(
                "The Torch-fronted PhysX wrench fast path currently requires a CUDA device"
            )
        frontend = FrontendTorch(0 if device.index is None else device.index)
        self._torch_physx_wrench_view = ArticulationView(
            backend,
            frontend,
            getattr(root_view, "_param_device", -1),
        )

        view = self._torch_physx_wrench_view
        if view.count != self.robot.num_instances or view.max_links != self.robot.num_bodies:
            raise RuntimeError(
                "Torch-fronted PhysX wrench view does not match the IsaacLab articulation: "
                f"view=({view.count} instances, {view.max_links} links), "
                f"robot=({self.robot.num_instances} instances, {self.robot.num_bodies} links)"
            )

        # Dense because PhysX's articulation force API has no link-index input.
        # Only ``body_id`` is written by the captured Warp graph; every other
        # link remains zero for the lifetime of these buffers.
        self._physx_wrench_force_wp = wp.zeros(
            (view.count, view.max_links), dtype=wp.vec3f, device=str(device)
        )
        self._physx_wrench_torque_wp = wp.zeros(
            (view.count, view.max_links), dtype=wp.vec3f, device=str(device)
        )
        self._physx_wrench_force_torch = wp.to_torch(self._physx_wrench_force_wp)
        self._physx_wrench_torque_torch = wp.to_torch(self._physx_wrench_torque_wp)
        self._physx_wrench_indices = torch.arange(
            view.count, dtype=torch.int32, device=device
        )

        print(
            "[RPG_Simulator] PhysX external wrench: captured Warp scatter -> "
            f"zero-copy Torch articulation submission ({view.count}x{view.max_links}, "
            f"body link {self.body_id})"
        )

    def pull_root_state(self) -> None:
        """Engine root state -> quad_state (pos/att/vel/ome + R/vel_b caches).

        The property access is the engine pull: on PhysX it triggers
        ``get_root_transforms()`` / ``get_root_velocities()`` into the stable
        buffers the ingest graph reads (Newton refreshes them in place).
        Timestamp-gated: ``scene.update(dt)`` bumps the sim timestamp right
        before this is called, so the access re-pulls.
        """
        data = self.robot.data
        data.root_link_pose_w  # noqa: B018  property access = engine pull into the stable buffer
        data.root_com_vel_w  # noqa: B018
        self.quad_state.ingest_external_root_state()

    def push_root_state(self, env_ids=None) -> None:
        """quad_state -> engine root state (pose + world-frame velocities)."""
        if env_ids is None and self._setup_push_root_state():
            wp.launch(
                self._push_root_kernel,
                dim=self.quad_state.num_quads,
                inputs=[self._wp_state, self.quad_state.wp_R],
                outputs=[self._root_pose_wp, self._root_vel_wp],
                device=str(self.quad_state.device),
            )
            # Mask writes take warp arrays directly and are the graph-safe API;
            # the deprecated write_root_state_to_sim() forces a torch round trip.
            self.robot.write_root_link_pose_to_sim_mask(root_pose=self._root_pose_wp)
            self.robot.write_root_com_velocity_to_sim_mask(root_velocity=self._root_vel_wp)
            return

        qs = self.quad_state
        ids = slice(None) if env_ids is None else env_ids
        root_state = qs.state[ids, :13].clone()
        # rpg stores attitude as wxyz; IsaacLab 3.0 expects xyzw.
        root_state[:, 3:7] = root_state[:, [4, 5, 6, 3]]
        # rpg ome is body-frame; IsaacLab expects world-frame angular velocity.
        root_state[:, 10:13] = torch.einsum("bij,bj->bi", qs.R[ids], qs.ome[ids])
        # Engine buffers are float32; rpg-side state may be float64.
        self.robot.write_root_state_to_sim(
            root_state.to(torch.float32), env_ids=env_ids
        )

    def _setup_push_root_state(self) -> bool:
        """Build the fused all-env root-state kernel; False if unsupported.

        Replaces a clone, a quaternion gather/scatter, an einsum and a dtype
        cast with one kernel writing the engine's transform/spatial-vector
        buffers directly.
        """
        if self._push_root_kernel is not None:
            return True
        if self._push_root_kernel is False:
            return False
        if not all(
            hasattr(self.robot, name)
            for name in ("write_root_link_pose_to_sim_mask", "write_root_com_velocity_to_sim_mask")
        ):
            self._push_root_kernel = False
            return False

        qs = self.quad_state
        device = str(qs.device)
        # Kernel annotations use these locals; needs eager annotation evaluation,
        # so this module must not adopt `from __future__ import annotations`.
        in_dtype = wp.float64 if qs.state.dtype == torch.float64 else wp.float32
        mat33_t = qs._mat33
        vec3_t = wp.types.vector(length=3, dtype=in_dtype)

        @wp.kernel(enable_backward=False)  # type: ignore[misc,valid-type]
        def push_root_kernel(
            state: wp.array(dtype=in_dtype, ndim=2),  # type: ignore[valid-type]
            R: wp.array(dtype=mat33_t),  # type: ignore[valid-type]  world<-body
            pose_out: wp.array(dtype=wp.transformf),  # type: ignore[valid-type]
            vel_out: wp.array(dtype=wp.spatial_vectorf),  # type: ignore[valid-type]
        ):
            env = wp.tid()
            # rpg stores attitude as wxyz; wp.transformf carries xyzw.
            pose_out[env] = wp.transformf(
                wp.vec3f(
                    wp.float32(state[env, 0]),
                    wp.float32(state[env, 1]),
                    wp.float32(state[env, 2]),
                ),
                wp.quatf(
                    wp.float32(state[env, 4]),
                    wp.float32(state[env, 5]),
                    wp.float32(state[env, 6]),
                    wp.float32(state[env, 3]),
                ),
            )
            # rpg ome is body-frame; IsaacLab expects world-frame angular velocity.
            omega_w = R[env] * vec3_t(state[env, 10], state[env, 11], state[env, 12])
            vel_out[env] = wp.spatial_vectorf(
                wp.float32(state[env, 7]),
                wp.float32(state[env, 8]),
                wp.float32(state[env, 9]),
                wp.float32(omega_w[0]),
                wp.float32(omega_w[1]),
                wp.float32(omega_w[2]),
            )

        self._wp_state = wp.from_torch(qs.state, dtype=in_dtype)
        self._root_pose_wp = wp.empty(qs.num_quads, dtype=wp.transformf, device=device)
        self._root_vel_wp = wp.empty(qs.num_quads, dtype=wp.spatial_vectorf, device=device)
        self._push_root_kernel = push_root_kernel
        return True

    def push_body_params(
        self, env_ids=None, *, mass: bool = True, inertia: bool = True
    ) -> None:
        """Randomized param-server mass/inertia batches -> engine body params.

        Only needed in engine-coupled mode (the engine integrates with them).
        Engine buffers are float32; rpg-side batches may be float64.
        """
        ps = self.param_server
        if env_ids is None:
            env_ids = torch.arange(len(ps.mass_batch), device=self.quad_state.device)
        body_id = self.body_id if isinstance(self.body_id, int) else self.body_id[0]
        if mass:
            self.robot.set_masses_index(
                masses=ps.mass_batch[env_ids].reshape(-1, 1).to(torch.float32),
                body_ids=[body_id],
                env_ids=env_ids,
            )
        if inertia:
            self.robot.set_inertias_index(
                inertias=ps.J_batch[env_ids].reshape(-1, 1, 9).to(torch.float32),
                body_ids=[body_id],
                env_ids=env_ids,
            )

    def write_wrench(self) -> None:
        """Submit ``quad_state.force_torque`` to the engine."""
        wp.capture_launch(self._wrench_graph)
        if self._torch_physx_wrench_view is not None:
            self._torch_physx_wrench_view.apply_forces_and_torques_at_position(
                force_data=self._physx_wrench_force_torch,
                torque_data=self._physx_wrench_torque_torch,
                position_data=None,
                indices=self._physx_wrench_indices,
                is_global=False,
            )
        else:
            # We wrote the composer's input buffers directly (bypassing set_*), so flag it
            # active + dirty for IsaacLab 3.0's lazy compose-on-read in write_data_to_sim.
            composer = self.robot.permanent_wrench_composer
            composer._active = True
            composer._dirty = True

    def _capture_wrench_graph(self):
        target_name = (
            "Torch-fronted PhysX articulation buffers"
            if self._torch_physx_wrench_view is not None
            else "WrenchComposer"
        )
        print(f"[RPG_Simulator] Compiling Fused Wrench Transfer Graph ({target_name})...")
        qs = self.quad_state
        device = str(qs.device)

        if self._torch_physx_wrench_view is not None:
            # These buffers are submitted directly to PhysX.  They already hold
            # body-frame values and only the selected drone-body slot is nonzero.
            local_force_b = self._physx_wrench_force_wp
            local_torque_b = self._physx_wrench_torque_wp
        else:
            composer = self.robot.permanent_wrench_composer
            composer._active = True

            # IsaacLab 3.0: write the WrenchComposer body-frame INPUT buffers (was the old
            # output buffers ``_composed_*_b``). rpg's force/torque are body-frame, applied at
            # the body link, so they map to ``_local_force_b`` / ``_local_torque_b``. The
            # composer composes inputs -> outputs lazily on read (see ``write_wrench``).
            local_force_b = composer._local_force_b
            local_torque_b = composer._local_torque_b

        # Kernel annotations use these locals; needs eager annotation evaluation,
        # so this module must not adopt `from __future__ import annotations`.
        in_dtype = (
            wp.float64 if qs.force_torque.dtype == torch.float64 else wp.float32
        )
        vec3_t = wp.types.vector(length=3, dtype=in_dtype)
        mat33_t = qs._mat33
        self.wp_force_torque = wp.from_torch(qs.force_torque, dtype=in_dtype)

        # Newton applies the composer's out_force_b verbatim as a WORLD vector (it skips the
        # body->world rotation PhysX/OvPhysX do via is_global=False). rpg's wrench is body-frame,
        # so for Newton we rotate it to world (R @ f, R @ tau) in-kernel; the other backends keep
        # the body wrench and rotate in their apply. Both end up applying the same world wrench.
        rotate_to_world = "isaaclab_newton" in type(self.robot).__module__

        if rotate_to_world:

            @wp.kernel(enable_backward=False)  # type: ignore[misc,valid-type]
            def wrench_kernel(
                ft: wp.array(dtype=in_dtype, ndim=2),  # type: ignore[valid-type]
                R: wp.array(dtype=mat33_t),  # type: ignore[valid-type]  world<-body
                f_out: wp.array2d(dtype=wp.vec3f),  # type: ignore[valid-type]
                t_out: wp.array2d(dtype=wp.vec3f),  # type: ignore[valid-type]
                body_idx: int,
            ):
                env = wp.tid()
                Rm = R[env]
                fw = Rm * vec3_t(ft[env, 0], ft[env, 1], ft[env, 2])
                tw = Rm * vec3_t(ft[env, 3], ft[env, 4], ft[env, 5])
                f_out[env, body_idx] = wp.vec3f(wp.float32(fw[0]), wp.float32(fw[1]), wp.float32(fw[2]))
                t_out[env, body_idx] = wp.vec3f(wp.float32(tw[0]), wp.float32(tw[1]), wp.float32(tw[2]))

            launch_inputs = [
                self.wp_force_torque,
                qs.wp_R,
                local_force_b,
                local_torque_b,
                self.body_id,
            ]
        else:

            @wp.kernel(enable_backward=False)  # type: ignore[misc,valid-type]
            def wrench_kernel(
                ft: wp.array(dtype=in_dtype, ndim=2),  # type: ignore[valid-type]
                f_out: wp.array2d(dtype=wp.vec3f),  # type: ignore[valid-type]
                t_out: wp.array2d(dtype=wp.vec3f),  # type: ignore[valid-type]
                body_idx: int,
            ):
                env = wp.tid()
                f_out[env, body_idx] = wp.vec3f(
                    wp.float32(ft[env, 0]), wp.float32(ft[env, 1]), wp.float32(ft[env, 2])
                )
                t_out[env, body_idx] = wp.vec3f(
                    wp.float32(ft[env, 3]), wp.float32(ft[env, 4]), wp.float32(ft[env, 5])
                )

            launch_inputs = [
                self.wp_force_torque,
                local_force_b,
                local_torque_b,
                self.body_id,
            ]

        self.wrench_kernel = wrench_kernel
        self.wrench_kernel.module.load(device=device)

        wp.capture_begin(device=device)
        wp.launch(
            self.wrench_kernel,
            dim=qs.num_quads,
            inputs=launch_inputs,
            device=device,
        )
        self._wrench_graph = wp.capture_end(device=device)
