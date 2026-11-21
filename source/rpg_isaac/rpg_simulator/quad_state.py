# from __future__ import annotations

import torch
import warp as wp
from collections.abc import Sequence

import rpg_simulator.utils.math_utils as math
from rpg_simulator.kernels import layout as L
from rpg_simulator.kernels import quad_state as K
from rpg_simulator.kernels.registry import ensure_loaded
from rpg_simulator.utils.graph import LaunchGraph


class QuadState:
    """State representation for a quadrotor."""

    def __init__(
        self,
        init_battery_voltage: float,
        num_quads=1,
        using_agisim=False,
        device="cuda",
        dtype=torch.float32,
    ):
        self.num_quads = num_quads
        self.using_agisim = using_agisim
        self.device = device
        self.dtype = dtype
        self.init_battery_voltage = float(init_battery_voltage)

        assert dtype in [
            torch.float32,
            torch.float64,
        ], "dtype must be either torch float32 or float64"

        self.state = torch.zeros(self.num_quads, 33, device=device, dtype=dtype)
        self.derivative = torch.zeros(self.num_quads, 33, device=device, dtype=dtype)

        if not using_agisim:
            self.force_torque = torch.zeros(
                self.num_quads, 6, device=device, dtype=dtype
            )

        self.pos_idx = slice(0, 3)
        self.att_idx = slice(3, 7)
        self.vel_idx = slice(7, 10)
        self.ome_idx = slice(10, 13)
        self.acc_idx = slice(13, 16)
        self.tau_idx = slice(16, 19)
        self.mot_idx = slice(19, 23)
        self.motdes_idx = slice(23, 27)
        self.E_idx = slice(27, 28)
        self.ucap_idx = slice(28, 29)
        self.ubat_idx = slice(29, 30)
        self.bfli_idx = slice(30, 33)
        # the kernels bake this layout as compile-time constants
        assert (
            self.pos_idx.start,
            self.att_idx.start,
            self.vel_idx.start,
            self.ome_idx.start,
            self.acc_idx.start,
            self.tau_idx.start,
            self.mot_idx.start,
            self.motdes_idx.start,
            self.E_idx.start,
            self.ucap_idx.start,
            self.ubat_idx.start,
            self.bfli_idx.start,
            self.bfli_idx.stop,
        ) == (L.POS, L.ATT, L.VEL, L.OME, L.ACC, L.TAU, L.MOT, L.MOTDES, L.E, L.UCAP, L.UBAT, L.BFLI, L.STATE_DIM)

        self.pos = self.state[:, self.pos_idx]
        self.att = self.state[:, self.att_idx]
        self.att[:, 0] = 1.0  # initialize to identity quaternion
        self.vel = self.state[:, self.vel_idx]
        self.ome = self.state[:, self.ome_idx]
        self.acc = self.state[:, self.acc_idx]
        self.tau = self.state[:, self.tau_idx]
        self.mot = self.state[:, self.mot_idx]
        self.motdes = self.state[:, self.motdes_idx]
        self.E = self.state[:, self.E_idx]
        self.ucap = self.state[:, self.ucap_idx]
        self.ubat = self.state[:, self.ubat_idx]
        self.ubat[:] = self.init_battery_voltage
        self.bfli = self.state[:, self.bfli_idx]

        self.d_pos = self.derivative[:, self.pos_idx]
        self.d_att = self.derivative[:, self.att_idx]
        self.d_vel = self.derivative[:, self.vel_idx]
        self.d_ome = self.derivative[:, self.ome_idx]
        self.d_acc = self.derivative[:, self.acc_idx]
        self.d_tau = self.derivative[:, self.tau_idx]
        self.d_mot = self.derivative[:, self.mot_idx]
        self.d_motdes = self.derivative[:, self.motdes_idx]
        self.d_E = self.derivative[:, self.E_idx]
        self.d_ucap = self.derivative[:, self.ucap_idx]
        self.d_ubat = self.derivative[:, self.ubat_idx]
        self.d_bfli = self.derivative[:, self.bfli_idx]

        if not using_agisim:
            self.force = self.force_torque[:, :3]
            self.torque = self.force_torque[:, 3:6]
            self.z_force_torque = self.force_torque[:, 2:]

        self.vel_b = torch.zeros((self.num_quads, 3), device=device, dtype=dtype)
        self.R = torch.eye(3, device=device, dtype=dtype).repeat(self.num_quads, 1, 1)

        #### Warp kernel setup ####
        wp.init()
        self.wp_dtype = wp.float32 if dtype == torch.float32 else wp.float64
        self._vec3 = wp.types.vector(length=3, dtype=self.wp_dtype)
        self._vec4 = wp.types.vector(length=4, dtype=self.wp_dtype)
        self._mat33 = wp.types.matrix(shape=(3, 3), dtype=self.wp_dtype)

        self.wp_att = wp.from_torch(self.att, dtype=self._vec4)
        self.wp_vel = wp.from_torch(self.vel, dtype=self._vec3)
        self.wp_R = wp.from_torch(self.R, dtype=self._mat33)
        self.wp_vel_b = wp.from_torch(self.vel_b, dtype=self._vec3)

        if not self.using_agisim:
            self.wp_pos = wp.from_torch(self.pos, dtype=self._vec3)
            self.wp_ome = wp.from_torch(self.ome, dtype=self._vec3)

        self.wp_device = (
            self.device.split(":")[0]
            if isinstance(self.device, str)
            else str(self.device).split(":")[0]
        )

        kernels = K.KERNELS[dtype]
        self._kernel_cache = kernels["cache_derived"]
        self._kernel_external = kernels["ingest_external"]
        ensure_loaded(self.wp_device, dtype, self._kernel_cache.module)
        self._cache_graph = None
        # external-engine root buffers; engine-coupled mode only
        # (see bind_external_root_state / ingest_external_root_state)
        self.wp_root_transforms = None
        self.wp_root_vels = None
        self._ingest_graph = None

    @classmethod
    def from_state(
        cls,
        state: torch.Tensor,
        using_agisim: bool,
        deriv: torch.Tensor | None = None,
        init_battery_voltage: float | None = None,
    ):
        """Defaults ``init_battery_voltage`` to the ``ubat`` already in ``state``."""
        dtype = state.dtype
        if state.ndim != 2 or state.shape[1] != 33:
            raise ValueError(
                f"State tensor must have shape (num_quads, 33), but got {state.shape}"
            )
        num_quads = state.shape[0]
        quad_state = cls(
            num_quads=num_quads,
            using_agisim=using_agisim,
            device=state.device,
            dtype=dtype,
            init_battery_voltage=0.0,
        )
        quad_state.state.copy_(state)
        quad_state.init_battery_voltage = (
            float(quad_state.ubat[0])
            if init_battery_voltage is None
            else float(init_battery_voltage)
        )
        if deriv is not None:
            if deriv.shape != state.shape:
                raise ValueError(
                    f"Derivative tensor must have the same shape as state tensor, but got {deriv.shape}"
                )
            quad_state.derivative.copy_(deriv)
        quad_state.cache_state()
        return quad_state

    def bind_external_root_state(
        self, root_pose: torch.Tensor, root_com_vel: torch.Tensor
    ) -> None:
        """Bind the external engine's pointer-stable root buffers — once.

        - ``root_pose``: (N, 7) ``[px,py,pz, qx,qy,qz,qw]`` (quat xyzw)
        - ``root_com_vel``: (N, 6) ``[vx,vy,vz, ox,oy,oz]``, world frame

        The ingest graph captures these device pointers, so the engine must
        rewrite the same memory in place; rebind after any reallocation.
        """
        assert not self.using_agisim, "external root binding is engine-coupled only"
        self.wp_root_transforms = wp.from_torch(root_pose, dtype=self.wp_dtype)
        self.wp_root_vels = wp.from_torch(root_com_vel, dtype=self.wp_dtype)
        self._ingest_graph = None

    def ingest_external_root_state(self) -> None:
        """Convert the bound engine root buffers into the rpg state layout.

        Writes pos / att (wxyz) / vel and the derived caches (R, vel_b,
        body-frame ome) via a captured-once, replayed CUDA graph. The caller
        must refresh the bound buffers first (engine pull).
        """
        if self._ingest_graph is None:
            assert (
                self.wp_root_transforms is not None
            ), "call bind_external_root_state first"
            self._ingest_graph = LaunchGraph(self.wp_device, self._launch_external_ingest)
        self._ingest_graph()

    def _launch_external_ingest(self):
        wp.launch(
            kernel=self._kernel_external,
            dim=self.num_quads,
            inputs=[
                self.wp_root_transforms,
                self.wp_root_vels,
                self.wp_pos,
                self.wp_att,
                self.wp_vel,
                self.wp_ome,
                self.wp_vel_b,
                self.wp_R,
            ],
            device=self.wp_device,
        )

    def reset_state(
        self,
        env_idx: Sequence[int] | None = None,
        quad_pos=None,
        quad_att=None,
        quad_vel=None,
        quad_ome=None,
    ) -> None:
        local_env_ids = slice(None) if env_idx is None else env_idx
        self.state[local_env_ids, :] = 0.0
        self.state[local_env_ids, self.att_idx.start] = 1.0  # identity quaternion
        self.state[local_env_ids, self.ubat_idx] = self.init_battery_voltage
        self.derivative[local_env_ids, :] = 0.0

        # Optionally set custom initial conditions
        if quad_pos is not None:
            self.state[local_env_ids, self.pos_idx] = quad_pos
        if quad_att is not None:
            self.state[local_env_ids, self.att_idx] = quad_att
        if quad_vel is not None:
            self.state[local_env_ids, self.vel_idx] = quad_vel
        if quad_ome is not None:
            self.state[local_env_ids, self.ome_idx] = quad_ome

        # Keep cached tensors consistent immediately after reset so observations
        # in the same step see the updated attitude perturbation.
        self.att[local_env_ids] = math.normalize(self.att[local_env_ids])
        self.R[local_env_ids] = math.matrix_from_quat(self.att[local_env_ids])
        self.vel_b[local_env_ids] = torch.einsum(
            "bij,bj->bi", self.R[local_env_ids].transpose(1, 2), self.vel[local_env_ids]
        )

        if not self.using_agisim:
            self.force_torque[local_env_ids, :] = 0.0

    def reset_battery(self, env_idx=None) -> None:
        local_env_ids = slice(None) if env_idx is None else env_idx
        self.state[local_env_ids, self.ubat_idx] = self.init_battery_voltage
        self.state[local_env_ids, self.E_idx] = 0.0
        self.state[local_env_ids, self.ucap_idx] = 0.0

    def cuda_launch(self):
        """Recompute the R / vel_b caches from att / vel (raw kernel launch)."""
        wp.launch(
            kernel=self._kernel_cache,
            dim=self.num_quads,
            inputs=[self.wp_att, self.wp_vel, self.wp_R, self.wp_vel_b],
            device=self.wp_device,
        )

    def cache_state(self):
        """Recompute the R / vel_b caches via a captured-once, replayed graph.

        Pure — operates on rpg-owned buffers only. In engine-coupled mode the
        root state itself is imported separately via
        :meth:`ingest_external_root_state` (which also refreshes the caches).
        """
        if self._cache_graph is None:
            self._cache_graph = LaunchGraph(self.wp_device, self.cuda_launch)
        self._cache_graph()
