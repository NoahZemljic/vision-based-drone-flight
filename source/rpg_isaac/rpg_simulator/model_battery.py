import torch
import warp as wp

from dataclasses import MISSING
from isaaclab.utils.configclass import configclass
from rpg_simulator.kernels import model_battery as K
from rpg_simulator.kernels.registry import ensure_loaded
from rpg_simulator.quad_state import QuadState
from rpg_simulator.utils.graph import LaunchGraph


# ======================Simulation module parameters======================
@configclass
class BatteryParams:
    cell_series: float = MISSING
    cell_parallel: float = MISSING
    cell_capacity: float = MISSING  # [Ah]
    u_0: float = MISSING  # Open circuit voltage [V]
    time_constant: float = MISSING
    time_constant_short: float = MISSING
    power_coefficient: float = MISSING
    energy_coefficients_lin: float = MISSING
    energy_coefficients_square: float = MISSING
    energy_coefficients_cube: float = MISSING
    r_coeff_power: float = MISSING
    r_coeff_capacity: float = MISSING
    r_min: float = MISSING


# ======================Simulation module======================
class ModelBattery:
    """Warp-accelerated model for battery dynamics (Zero-Allocation, CUDA Graph)."""

    def __init__(
        self,
        battery_params: BatteryParams,
        quad_state: QuadState,
        num_env: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):

        self.device = device
        self.num_env = num_env
        self.dtype = dtype
        self.wp_dtype = wp.float32 if dtype == torch.float32 else wp.float64
        self.params = battery_params

        wp.init()

        self.E_handle = wp.from_torch(quad_state.E, dtype=self.wp_dtype)
        self.d_E_handle = wp.from_torch(quad_state.d_E, dtype=self.wp_dtype)
        self.ucap_handle = wp.from_torch(quad_state.ucap, dtype=self.wp_dtype)
        self.ubat_handle = wp.from_torch(quad_state.ubat, dtype=self.wp_dtype)
        self.d_ucap_handle = wp.from_torch(quad_state.d_ucap, dtype=self.wp_dtype)
        self.d_ubat_handle = wp.from_torch(quad_state.d_ubat, dtype=self.wp_dtype)

        # per-instance coefficients (pre-folded), read by the kernel at runtime
        p = battery_params
        total_cells = p.cell_series * p.cell_parallel * p.cell_capacity
        self.params_table = torch.zeros(K.P_SIZE, device=device, dtype=dtype)
        self.params_table[K.P_NORM_FACTOR] = -1e-3 / total_cells
        self.params_table[K.P_POWER_NORM] = -1.0 / total_cells
        self.params_table[K.P_INV_TC] = 1.0 / p.time_constant
        self.params_table[K.P_INV_TC_SHORT] = 1.0 / p.time_constant_short
        self.params_table[K.P_C_LIN] = p.energy_coefficients_lin
        self.params_table[K.P_C_SQ] = p.energy_coefficients_square
        self.params_table[K.P_C_CUBE] = p.energy_coefficients_cube
        self.params_table[K.P_U0] = p.u_0
        self.params_table[K.P_NEG_4_RMIN] = -4.0 * p.r_min
        self.params_table[K.P_PWR_COEFF] = p.power_coefficient
        self.params_table[K.P_U0_CELLS] = p.u_0 * p.cell_series
        self.params_table[K.P_HALF_CELLS] = 0.5 * p.cell_series
        self.params_handle = wp.from_torch(self.params_table, dtype=self.wp_dtype)

        self.battery_kernel = K.KERNELS[dtype]["battery_kernel"]
        ensure_loaded(self.device, dtype, self.battery_kernel.module)

        self._graph = None

    def cuda_launch(self, quad_state=None):
        wp.launch(
            kernel=self.battery_kernel,
            dim=self.num_env,
            inputs=[
                self.E_handle,
                self.d_E_handle,
                self.ucap_handle,
                self.ubat_handle,
                self.d_ucap_handle,
                self.d_ubat_handle,
                self.params_handle,
            ],
            device=self.device,
        )

    def run(self, quad_state: QuadState):
        """Execute battery dynamics via CUDA graph replay."""
        if self._graph is None:
            self._graph = LaunchGraph(self.device, lambda: self.cuda_launch(quad_state))
        self._graph()
