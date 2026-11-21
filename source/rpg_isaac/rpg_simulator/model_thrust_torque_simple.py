import torch
import warp as wp

from rpg_simulator.kernels import model_thrust_torque_simple as K
from rpg_simulator.kernels.registry import ensure_loaded
from rpg_simulator.quadrotor_params import QuadParamServer
from rpg_simulator.utils.graph import LaunchGraph


class ModelThrustTorqueSimple:
    """Warp-accelerated thrust and torque model (Zero-Allocation, CUDA Graph)."""

    def __init__(
        self,
        quad_params: QuadParamServer,
        quad_state,  # Pass the quad_state directly to init for handles
        num_env: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):

        self.device = device
        self.num_env = num_env
        self.dtype = dtype
        self.wp_dtype = wp.float32 if dtype == torch.float32 else wp.float64

        wp.init()

        # Extract constants
        self.thrust_map = quad_params.thrust_map
        self.B_allocation_T = quad_params.B_allocation.T.contiguous()

        # Check AgiSim state structure
        self.is_agisim = getattr(quad_state, "using_agisim", False)

        # Create memory handles
        self.mot_handle = wp.from_torch(quad_state.mot, dtype=self.wp_dtype)
        self.wp_mass = wp.from_torch(
            quad_params.mass_batch, dtype=self.wp_dtype
        )  # Mass as a batch tensor for AgiSim kernel
        self.wp_J_inv = wp.from_torch(
            quad_params.J_inv_batch, dtype=self.wp_dtype
        )  # J_inv as a batch tensor [N,3,3] for AgiSim kernel

        if self.is_agisim:
            self.d_vel_handle = wp.from_torch(quad_state.d_vel, dtype=self.wp_dtype)
            self.d_ome_handle = wp.from_torch(quad_state.d_ome, dtype=self.wp_dtype)
            self.R_handle = wp.from_torch(
                quad_state.R, dtype=self.wp_dtype
            )  # R is [N,3,3] already contiguous
        else:
            self.z_force_torque_handle = wp.from_torch(
                quad_state.z_force_torque, dtype=self.wp_dtype
            )

        # per-instance thrust map + allocation, read by the kernels at runtime
        self.params = torch.zeros(K.P_SIZE, device=device, dtype=dtype)
        self.params[K.P_C0] = self.thrust_map[0]
        self.params[K.P_C1] = self.thrust_map[1]
        self.params[K.P_C2] = self.thrust_map[2]
        B_T = self.B_allocation_T.cpu().tolist()
        for i in range(4):
            for j in range(4):
                self.params[K.P_B + 4 * i + j] = B_T[i][j]
        self.params_handle = wp.from_torch(self.params, dtype=self.wp_dtype)

        kernels = K.KERNELS[dtype]
        self.thrust_kernel = kernels["kernel_agisim"] if self.is_agisim else kernels["kernel_standard"]
        ensure_loaded(self.device, dtype, self.thrust_kernel.module)

        self._graph = None

    def cuda_launch(self, quad_state):
        if self.is_agisim:
            wp.launch(
                kernel=self.thrust_kernel,
                dim=self.num_env,
                inputs=[
                    self.mot_handle,
                    self.R_handle,
                    self.d_vel_handle,
                    self.d_ome_handle,
                    self.wp_mass,
                    self.wp_J_inv,
                    self.params_handle,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=self.thrust_kernel,
                dim=self.num_env,
                inputs=[self.mot_handle, self.z_force_torque_handle, self.params_handle],
                device=self.device,
            )

    def run(self, quad_state):
        """Execute thrust mapping via CUDA graph replay."""
        if self._graph is None:
            self._graph = LaunchGraph(self.device, lambda: self.cuda_launch(quad_state))
        self._graph()
