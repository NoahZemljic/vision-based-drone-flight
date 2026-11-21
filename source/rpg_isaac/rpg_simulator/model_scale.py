import torch
import warp as wp

from rpg_simulator.kernels import model_scale as K
from rpg_simulator.kernels.registry import ensure_loaded
from rpg_simulator.quad_state import QuadState
from rpg_simulator.quadrotor_params import QuadParamServer
from rpg_simulator.utils.graph import LaunchGraph


class ModelScale:
    """Warp-accelerated force/torque scaling model (Zero-Allocation, CUDA Graph).

    In the AgiSim path the derivative tensor (d_vel, d_ome) contains world-frame
    accelerations that already include gravity.  We rotate into the body frame,
    strip gravity, scale the propulsive components, rotate back and re-add gravity.

    In the PhysX (Isaac) path the force_torque tensor only contains propulsive
    forces (body frame) — gravity is handled by PhysX — so we simply scale the
    components in-place.
    """

    def __init__(
        self,
        quad_params: QuadParamServer,
        quad_state: QuadState,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device
        self.num_env = quad_state.num_quads
        self.dtype = dtype
        self.wp_dtype = wp.float32 if dtype == torch.float32 else wp.float64
        self.quad_state = quad_state
        self.quad_params = quad_params

        wp.init()

        # ---- Warp handles for batch params (mutable for domain randomization) ----
        self.wp_scale_drag = wp.from_torch(
            quad_params.scale_drag_batch, dtype=self.wp_dtype
        )
        self.wp_scale_thrust = wp.from_torch(
            quad_params.scale_thrust_batch, dtype=self.wp_dtype
        )
        self.wp_scale_torque = wp.from_torch(
            quad_params.scale_torque_batch, dtype=self.wp_dtype
        )

        self.is_agisim = quad_state.using_agisim

        if self.is_agisim:
            self.wp_mass = wp.from_torch(quad_params.mass_batch, dtype=self.wp_dtype)
            self.d_vel_handle = wp.from_torch(quad_state.d_vel, dtype=self.wp_dtype)
            self.d_ome_handle = wp.from_torch(quad_state.d_ome, dtype=self.wp_dtype)
            self.R_handle = wp.from_torch(quad_state.R, dtype=self.wp_dtype)
        else:
            self.force_torque_handle = wp.from_torch(
                quad_state.force_torque, dtype=self.wp_dtype
            )

        kernels = K.KERNELS[dtype]
        self.scale_kernel = kernels["scale_agisim"] if self.is_agisim else kernels["scale_physx"]
        ensure_loaded(self.device, dtype, self.scale_kernel.module)

        self._graph = None

    def cuda_launch(self, quad_state: QuadState):
        if self.is_agisim:
            wp.launch(
                kernel=self.scale_kernel,
                dim=self.num_env,
                inputs=[
                    self.d_vel_handle,
                    self.d_ome_handle,
                    self.R_handle,
                    self.wp_mass,
                    self.wp_scale_drag,
                    self.wp_scale_thrust,
                    self.wp_scale_torque,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=self.scale_kernel,
                dim=self.num_env,
                inputs=[
                    self.force_torque_handle,
                    self.wp_scale_drag,
                    self.wp_scale_thrust,
                    self.wp_scale_torque,
                ],
                device=self.device,
            )

    def run(self, quad_state: QuadState):
        """Execute scaling via CUDA graph replay."""
        if self._graph is None:
            self._graph = LaunchGraph(self.device, lambda: self.cuda_launch(quad_state))
        self._graph()
