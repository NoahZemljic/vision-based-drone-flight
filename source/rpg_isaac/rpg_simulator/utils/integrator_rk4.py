import torch
import warp as wp

from rpg_simulator.kernels import integrator_rk4 as K
from rpg_simulator.kernels.registry import ensure_loaded
from rpg_simulator.quad_state import QuadState
from rpg_simulator.utils.graph import LaunchGraph


class IntegratorRK4:
    """Runge-Kutta 4th order integrator (Fully Fused GPU Graph)."""

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
        self.force_dim = 6

        self.using_agisim = quad_state.using_agisim

        wp.init()

        # ------------------------------------------------------------------
        # 1. Pre-allocate Buffers (Zero Allocation Loop)
        # ------------------------------------------------------------------
        # We need temporary storage for the initial state snapshot and the 4 slopes (k1-k4)

        # State Snapshots
        self.state_init = wp.zeros(
            (self.num_envs, self.state_dim), dtype=self.wp_dtype, device=device
        )

        # Derivative Slopes (k buffers)
        # We pre-allocate these to avoid .clone() inside the step loop
        self.k1 = wp.zeros(
            (self.num_envs, self.state_dim), dtype=self.wp_dtype, device=device
        )
        self.k2 = wp.zeros(
            (self.num_envs, self.state_dim), dtype=self.wp_dtype, device=device
        )
        self.k3 = wp.zeros(
            (self.num_envs, self.state_dim), dtype=self.wp_dtype, device=device
        )
        self.k4 = wp.zeros(
            (self.num_envs, self.state_dim), dtype=self.wp_dtype, device=device
        )

        # Buffers for Force/Torque (Only if not using Agisim)
        if not self.using_agisim:
            self.ft_init = wp.zeros(
                (self.num_envs, self.force_dim), dtype=self.wp_dtype, device=device
            )
            self.k1_ft = wp.zeros(
                (self.num_envs, self.force_dim), dtype=self.wp_dtype, device=device
            )
            self.k2_ft = wp.zeros(
                (self.num_envs, self.force_dim), dtype=self.wp_dtype, device=device
            )
            self.k3_ft = wp.zeros(
                (self.num_envs, self.force_dim), dtype=self.wp_dtype, device=device
            )
            self.k4_ft = wp.zeros(
                (self.num_envs, self.force_dim), dtype=self.wp_dtype, device=device
            )

        # ------------------------------------------------------------------
        # 2. Handles to Live Tensors
        # ------------------------------------------------------------------
        self.state_handle = wp.from_torch(quad_state.state, dtype=self.wp_dtype)
        self.deriv_handle = wp.from_torch(quad_state.derivative, dtype=self.wp_dtype)

        if not self.using_agisim:
            self.ft_handle = wp.from_torch(quad_state.force_torque, dtype=self.wp_dtype)

        # For AgiSim Accel calc
        if self.using_agisim:
            self.vel_idx_start = quad_state.vel_idx.start
            self.acc_idx_start = quad_state.acc_idx.start

        # ------------------------------------------------------------------
        # 3. Kernels
        # ------------------------------------------------------------------
        kernels = K.KERNELS[dtype]
        self.copy_kernel = kernels["copy_k"]
        self.substep_kernel = kernels["substep_k"]
        self.final_kernel = kernels["final_k"]
        if self.using_agisim:
            self.accel_kernel = kernels["accel_k"]
        ensure_loaded(self.device, dtype, self.copy_kernel.module)

        # Lazy Init Graph
        self.graph = None

    def step(self, state: QuadState, deriv_func) -> None:
        """
        Execute one RK4 step using a fused CUDA graph.
        'deriv_func' is expected to launch the physics pipeline graph.
        """
        if self.graph is None:
            self.graph = LaunchGraph(self.device, lambda: self._launch_sequence(deriv_func))
        self.graph()

    def _launch_sequence(self, deriv_func):
        """Records the sequence: Save -> (Deriv -> k -> Substep)x4 -> Combine."""

        # 1. Snapshot State: state_init = state
        self._launch_copy(self.state_handle, self.state_init)

        # ================== Stage 1 ==================
        # Calculate k1 at initial state
        # Note: deriv_func writes to self.deriv_handle (quad_state.derivative)
        deriv_func()
        self._launch_copy(self.deriv_handle, self.k1)
        if not self.using_agisim:
            self._launch_copy(self.ft_handle, self.k1_ft)

        # Setup state for k2: state = init + k1 * dt/2
        self._launch_substep(self.state_init, self.k1, 0.5 * self.dt, self.state_handle)

        # ================== Stage 2 ==================
        deriv_func()
        self._launch_copy(self.deriv_handle, self.k2)
        if not self.using_agisim:
            self._launch_copy(self.ft_handle, self.k2_ft)

        # Setup state for k3: state = init + k2 * dt/2
        self._launch_substep(self.state_init, self.k2, 0.5 * self.dt, self.state_handle)

        # ================== Stage 3 ==================
        deriv_func()
        self._launch_copy(self.deriv_handle, self.k3)
        if not self.using_agisim:
            self._launch_copy(self.ft_handle, self.k3_ft)

        # Setup state for k4: state = init + k3 * dt
        self._launch_substep(self.state_init, self.k3, self.dt, self.state_handle)

        # ================== Stage 4 ==================
        deriv_func()
        self._launch_copy(self.deriv_handle, self.k4)
        if not self.using_agisim:
            self._launch_copy(self.ft_handle, self.k4_ft)

        # ================== Finalize ==================
        # state = init + dt/6 * (k1 + 2k2 + 2k3 + k4)
        if self.using_agisim:
            wp.launch(
                kernel=self.final_kernel,
                dim=self.num_envs,
                inputs=[
                    self.state_init,
                    self.k1,
                    self.k2,
                    self.k3,
                    self.k4,
                    self.dt,
                    self.state_handle,
                ],
                device=self.device,
            )
            # AgiSim: Calculate acceleration (v_new - v_old) / dt
            wp.launch(
                kernel=self.accel_kernel,
                dim=self.num_envs,
                inputs=[self.state_init, self.state_handle, self.dt],
                device=self.device,
            )
        else:
            # Update state
            wp.launch(
                kernel=self.final_kernel,
                dim=self.num_envs,
                inputs=[
                    self.state_init,
                    self.k1,
                    self.k2,
                    self.k3,
                    self.k4,
                    self.dt,
                    self.state_handle,
                ],
                device=self.device,
            )
            # Update force/torque average
            # We reuse final_kernel logic but passing FT buffers
            wp.launch(
                kernel=self.final_kernel,
                dim=self.num_envs,
                inputs=[
                    self.ft_init,
                    self.k1_ft,
                    self.k2_ft,
                    self.k3_ft,
                    self.k4_ft,
                    1.0,
                    self.ft_handle,
                ],
                device=self.device,
            )


    # ------------------ Launch Helpers ------------------
    def _launch_copy(self, src, dst):
        # We flatten dimensions to 2D for generic copy kernel
        wp.launch(
            self.copy_kernel,
            dim=(dst.shape[0], dst.shape[1]),
            inputs=[src, dst],
            device=self.device,
        )

    def _launch_substep(self, init, k, dt, out):
        wp.launch(
            self.substep_kernel,
            dim=(out.shape[0], out.shape[1]),
            inputs=[init, k, dt, out],
            device=self.device,
        )
