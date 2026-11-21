# from __future__ import annotations  # doesn't work with warp's @kernel type annotations, which require concrete types

import torch
import warp as wp
from dataclasses import MISSING
from isaaclab.utils.configclass import configclass

from rpg_simulator.kernels import model_motor as K
from rpg_simulator.kernels.registry import ensure_loaded
from rpg_simulator.quad_state import QuadState
from rpg_simulator.quadrotor_params import QuadParamServer
from rpg_simulator.utils.graph import LaunchGraph


# ======================Simulation module parameters======================
@configclass
class MotorParams:
    time_constant: float = MISSING  # Motor time constant [s]
    inertia: float = MISSING  # Rotor inertia [kgm^2]
    pwr_coeff_drag: float = MISSING  # Power coefficient for drag torque [Nm/(rad/s)^2]
    pwr_coeff_linear: float = (
        MISSING  # Power coefficient for linear torque [Nm/(rad/s)]
    )
    pwr_coeff_cubic: float = MISSING  # Power coefficient for cubic torque [Nm]
    pwr_coeff_sixth: float = (
        MISSING  # Power coefficient for sixth-order torque [Nm/(rad/s)^2]
    )


# ======================Simulation module======================
class ModelMotor:
    """Warp-accelerated model for motor dynamics (Zero-Allocation)."""

    def __init__(
        self,
        motor_params: MotorParams,
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
        self.tau = motor_params.time_constant

        wp.init()

        # Check state type once
        self.is_agisim = quad_state.using_agisim
        self.quad_state = quad_state

        # Tensor handles for zero-copy access inside the Warp kernel
        self.state_handle = wp.from_torch(quad_state.state, dtype=self.wp_dtype)
        self.deriv_handle = wp.from_torch(quad_state.derivative, dtype=self.wp_dtype)
        if not self.is_agisim:
            self.force_torque_handle = wp.from_torch(
                quad_state.force_torque, dtype=self.wp_dtype
            )

        self.wp_J_inv = wp.from_torch(
            quad_params.J_inv_batch, dtype=self.wp_dtype
        )  # [N,3,3] per-env inverse inertia

        # per-instance coefficients, read by the kernels at runtime
        p = motor_params
        self.params = torch.zeros(K.P_SIZE, device=device, dtype=dtype)
        self.params[K.P_DRAG] = p.pwr_coeff_drag
        self.params[K.P_LIN] = p.pwr_coeff_linear
        self.params[K.P_CUB] = p.pwr_coeff_cubic
        self.params[K.P_SIX] = p.pwr_coeff_sixth
        self.params[K.P_INV_TAU] = 0.0 if self.tau == 0.0 else 1.0 / self.tau
        self.params[K.P_INERTIA] = p.inertia
        self.params_handle = wp.from_torch(self.params, dtype=self.wp_dtype)

        # 3. Pick the kernel for the current configuration
        kernels = K.KERNELS[dtype]
        if self.tau == 0.0:
            self.motor_kernel = kernels["kernel_tau_0"]
        elif self.is_agisim:
            self.motor_kernel = kernels["kernel_tau_gt_0_agisim"]
        else:
            self.motor_kernel = kernels["kernel_tau_gt_0_engine"]
        ensure_loaded(self.device, dtype, self.motor_kernel.module)

        self._graph = None

    def cuda_launch(self, quad_State: QuadState):
        if self.tau == 0.0:
            # Path A: Instant Response
            wp.launch(
                kernel=self.motor_kernel,
                dim=self.num_env,
                inputs=[self.state_handle, self.deriv_handle, self.params_handle],
                device=self.device,
            )
        else:
            # Path B: Standard Response
            if self.is_agisim:
                wp.launch(
                    kernel=self.motor_kernel,
                    dim=self.num_env,
                    inputs=[self.state_handle, self.deriv_handle, self.wp_J_inv, self.params_handle],
                    device=self.device,
                )
            else:
                wp.launch(
                    kernel=self.motor_kernel,
                    dim=self.num_env,
                    inputs=[
                        self.state_handle,
                        self.deriv_handle,
                        self.force_torque_handle,
                        self.params_handle,
                    ],
                    device=self.device,
                )

    def run(self, quad_state):
        """Execute motor dynamics via CUDA graph replay."""
        if self._graph is None:
            self._graph = LaunchGraph(self.device, lambda: self.cuda_launch(quad_state))
        self._graph()
