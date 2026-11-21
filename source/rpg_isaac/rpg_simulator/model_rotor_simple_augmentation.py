# from __future__ import annotations  # doesn't work with warp's @kernel type annotations, which require concrete types

import torch
import warp as wp

# wp.config.verbose = True
# wp.config.verify_fp = True  # Optional: adds bounds checking and NaN checks

from dataclasses import MISSING
from isaaclab.utils.configclass import configclass

from rpg_simulator.kernels import model_rotor_simple_augmentation as K
from rpg_simulator.kernels.registry import ensure_loaded
from rpg_simulator.quad_state import QuadState
from rpg_simulator.quadrotor_params import QuadParamServer
from rpg_simulator.utils.graph import LaunchGraph


# ======================Simulation module parameters======================
@configclass
class RotorSimpleAugParams:
    """Parameters for simple rotor augmented model."""

    scale_force: float = 1.0
    scale_torque: float = 1.0

    fx_offset: float = 0.0
    fx_m_mean: float = 0.0
    fx_ang_y: float = 0.0
    fx_vx: float = 0.0
    fx_vx_sqr: float = 0.0
    fx_vx_cub: float = 0.0
    fx_vx_m_mean: float = 0.0  # vx * m_mean

    fy_offset: float = 0.0
    fy_m_mean: float = 0.0
    fy_ang_x: float = 0.0
    fy_vy: float = 0.0
    fy_vy_sqr: float = 0.0
    fy_vy_cub: float = 0.0
    fy_vy_m_mean: float = 0.0  # vy * m_mean

    fz_offset: float = 0.0
    fz_vhor: float = 0.0
    fz_vz: float = 0.0
    fz_m_mean: float = 0.0
    fz_vhor_vz: float = 0.0  # vhor * vz
    fz_vhor_m_mean: float = 0.0  # vhor * m_mean
    fz_vz_m_mean: float = 0.0  # vz * m_mean
    fz_vhor_sqr: float = 0.0
    fz_vhor_vz_m_mean: float = 0.0  # vhor * vz * m_mean
    fz_vz_cub: float = 0.0

    tx_offset: float = 0.0
    tx_m_mean: float = 0.0
    tx_vy: float = 0.0
    tx_vy_m_mean: float = 0.0  # vy * m_mean

    ty_offset: float = 0.0
    ty_m_mean: float = 0.0
    ty_vx: float = 0.0
    ty_vx_m_mean: float = 0.0  # vx * m_mean

    tz_offset: float = 0.0
    tz_vx: float = 0.0
    tz_vy: float = 0.0


class ModelRotorSimpleAugmentation:
    """Warp-accelerated rotor augmentation model for quadrotors."""

    def __init__(
        self,
        params: RotorSimpleAugParams,
        quad_params: QuadParamServer,
        quad_state: QuadState,
        num_env: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):

        self.device = device
        self.num_env = num_env
        self.dtype = dtype
        self.wp_dtype = wp.float32 if dtype == torch.float32 else wp.float64
        self.quad_state = quad_state

        wp.init()

        if self.quad_state.using_agisim:
            self.R_handle = wp.from_torch(self.quad_state.R, dtype=self.wp_dtype)
            self.deriv_handle = wp.from_torch(
                self.quad_state.derivative, dtype=self.wp_dtype
            )

        # Tensor handles for zero-copy access in kernels
        self.state_handle = wp.from_torch(quad_state.state, dtype=self.wp_dtype)
        self.vel_b_handle = wp.from_torch(quad_state.vel_b, dtype=self.wp_dtype)
        if not self.quad_state.using_agisim:
            self.force_torque_handle = wp.from_torch(
                quad_state.force_torque, dtype=self.wp_dtype
            )

        self.wp_mass = wp.from_torch(
            quad_params.mass_batch, dtype=self.wp_dtype
        )  # [N] per-env mass
        self.wp_J = wp.from_torch(
            quad_params.J_batch, dtype=self.wp_dtype
        )  # [N,3,3] per-env inertia

        self.quad_params = quad_params

        # per-instance fit coefficients, read by the kernel at runtime
        self.params = torch.zeros(K.P_SIZE, device=device, dtype=dtype)
        self._fill_params(params)
        self.params_handle = wp.from_torch(self.params, dtype=self.wp_dtype)

        kernels = K.KERNELS[dtype]
        if self.quad_state.using_agisim:
            self.aug_kernel = kernels["aug_kernel_agisim"]
        else:
            self.aug_kernel = kernels["aug_kernel_engine"]
        ensure_loaded(self.device, dtype, self.aug_kernel.module)

        self._graph = None

    def cuda_launch(self, quad_state: QuadState):
        if self.quad_state.using_agisim:
            wp.launch(
                kernel=self.aug_kernel,
                dim=self.num_env,
                inputs=[
                    self.state_handle,
                    self.vel_b_handle,
                    self.R_handle,
                    self.deriv_handle,
                    self.params_handle,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=self.aug_kernel,
                dim=self.num_env,
                inputs=[
                    self.state_handle,
                    self.vel_b_handle,
                    self.force_torque_handle,
                    self.wp_mass,
                    self.wp_J,
                    self.params_handle,
                ],
                device=self.device,
            )

    def run(self, quad_state: QuadState):
        if self._graph is None:
            self._graph = LaunchGraph(self.device, lambda: self.cuda_launch(quad_state))
        self._graph()

    def _fill_params(self, p: RotorSimpleAugParams):
        t = self.params
        t[K.P_SCALE_FORCE] = p.scale_force
        t[K.P_SCALE_TORQUE] = p.scale_torque
        t[K.P_FX_VX] = p.fx_vx
        t[K.P_FX_M_MEAN] = p.fx_m_mean
        t[K.P_FX_VX_M] = p.fx_vx_m_mean
        t[K.P_FX_VX_SQR] = p.fx_vx_sqr
        t[K.P_FX_VX_CUB] = p.fx_vx_cub
        t[K.P_FY_VY] = p.fy_vy
        t[K.P_FY_M_MEAN] = p.fy_m_mean
        t[K.P_FY_VY_M] = p.fy_vy_m_mean
        t[K.P_FY_VY_SQR] = p.fy_vy_sqr
        t[K.P_FY_VY_CUB] = p.fy_vy_cub
        t[K.P_FZ_OFFSET] = p.fz_offset
        t[K.P_FZ_VHOR] = p.fz_vhor
        t[K.P_FZ_VZ] = p.fz_vz
        t[K.P_FZ_M_MEAN] = p.fz_m_mean
        t[K.P_FZ_VHOR_VZ] = p.fz_vhor_vz
        t[K.P_FZ_VHOR_M] = p.fz_vhor_m_mean
        t[K.P_FZ_VZ_M] = p.fz_vz_m_mean
        t[K.P_FZ_VHOR_SQR] = p.fz_vhor_sqr
        t[K.P_FZ_VHOR_VZ_M] = p.fz_vhor_vz_m_mean
        t[K.P_FZ_VZ_CUB] = p.fz_vz_cub
        t[K.P_TX_VY] = p.tx_vy
        t[K.P_TX_M_MEAN] = p.tx_m_mean
        t[K.P_TX_VY_M] = p.tx_vy_m_mean
        t[K.P_TY_VX] = p.ty_vx
        t[K.P_TY_M_MEAN] = p.ty_m_mean
        t[K.P_TY_VX_M] = p.ty_vx_m_mean
        t[K.P_TZ_VX] = p.tz_vx
        t[K.P_TZ_VY] = p.tz_vy
