import torch
import warp as wp

from rpg_simulator.kernels import integrator_euler as K
from rpg_simulator.kernels.registry import ensure_loaded
from rpg_simulator.quad_state import QuadState
from rpg_simulator.utils.graph import LaunchGraph


class IntegratorEuler:
    """Euler integrator (Fully Fused GPU Graph)."""

    def __init__(
        self,
        dt: float,
        quad_state: QuadState,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.dt = dt
        self.device = device
        self.dtype = dtype
        self.wp_dtype = wp.float32 if dtype == torch.float32 else wp.float64
        self.num_envs = quad_state.num_quads
        self.state_dim = quad_state.state.shape[1]

        # Only AgiSim relies on the integrator for kinematics.
        # If using Isaac Sim, the physics engine handles integration.
        self.using_agisim = quad_state.using_agisim

        wp.init()

        # ------------------------------------------------------------------
        # 1. Pre-allocate Buffers (Zero Allocation Loop)
        # ------------------------------------------------------------------
        # We need temporary storage for the initial state snapshot to compute acceleration later.
        self.state_init = wp.zeros(
            (self.num_envs, self.state_dim), dtype=self.wp_dtype, device=device
        )

        # ------------------------------------------------------------------
        # 2. Handles to Live Tensors
        # ------------------------------------------------------------------
        self.state_handle = wp.from_torch(quad_state.state, dtype=self.wp_dtype)
        self.deriv_handle = wp.from_torch(quad_state.derivative, dtype=self.wp_dtype)

        # For AgiSim Accel calc
        if self.using_agisim:
            self.vel_idx_start = quad_state.vel_idx.start
            self.acc_idx_start = quad_state.acc_idx.start

        # ------------------------------------------------------------------
        # 3. Kernels
        # ------------------------------------------------------------------
        kernels = K.KERNELS[dtype]
        self.copy_kernel = kernels["copy_k"]
        self.euler_step_kernel = kernels["euler_step_k"]
        if self.using_agisim:
            self.accel_kernel = kernels["accel_k"]
        ensure_loaded(self.device, dtype, self.copy_kernel.module)

        # Lazy Init Graph
        self.graph = None

    def step(self, state: QuadState, deriv_func) -> None:
        """
        Execute one Euler step using a fused CUDA graph.
        'deriv_func' is expected to launch the physics pipeline graph.
        """
        if self.graph is None:
            self.graph = LaunchGraph(self.device, lambda: self._launch_sequence(deriv_func))
        self.graph()

    def _launch_sequence(self, deriv_func):
        """Records the sequence: Save -> Deriv -> Substep -> Accel."""

        # 1. Snapshot State: state_init = state
        self._launch_copy(self.state_handle, self.state_init)

        # 2. Calculate Derivative
        # deriv_func writes to self.deriv_handle (quad_state.derivative)
        deriv_func()

        # 3. Apply Euler Step: state = init + deriv * dt
        # We can write directly back to state_handle
        wp.launch(
            self.euler_step_kernel,
            dim=(self.num_envs, self.state_dim),
            inputs=[self.state_init, self.deriv_handle, self.dt, self.state_handle],
            device=self.device,
        )

        # 4. Finalize
        if self.using_agisim:
            # AgiSim: Calculate acceleration (v_new - v_old) / dt
            wp.launch(
                kernel=self.accel_kernel,
                dim=self.num_envs,
                inputs=[self.state_init, self.state_handle, self.dt],
                device=self.device,
            )

        # Note: For Euler, we don't need to average force_torque over multiple substeps like RK4.
        # The deriv_func naturally writes the correct final force_torque into the tensor.


    # ------------------ Launch Helpers ------------------
    def _launch_copy(self, src, dst):
        wp.launch(
            self.copy_kernel,
            dim=(dst.shape[0], dst.shape[1]),
            inputs=[src, dst],
            device=self.device,
        )
