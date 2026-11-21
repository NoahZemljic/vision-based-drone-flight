import os
import torch
import warp as wp
from dataclasses import MISSING
from isaaclab.utils.configclass import configclass

import rpg_assets
from rpg_simulator.kernels import low_level_controller_betaflight as K
from rpg_simulator.kernels.registry import ensure_loaded
from rpg_simulator.low_level_controller_base import LowLevelControllerBase
from rpg_simulator.quadrotor_params import QuadParamServer
from rpg_simulator.quad_state import QuadState
from rpg_simulator.utils.thrust_map import ThrustMap
from rpg_simulator.utils.graph import LaunchGraph


# ======================Simulation module parameters======================
@configclass
class BetaFlightParams:
    # PID:
    kp_roll: float = MISSING
    ki_roll: float = MISSING
    kd_roll: float = MISSING
    kp_pitch: float = MISSING
    ki_pitch: float = MISSING
    kd_pitch: float = MISSING
    kp_yaw: float = MISSING
    ki_yaw: float = MISSING
    kd_yaw: float = MISSING

    # throttle:
    min_throttle_pwm: float = MISSING
    max_throttle_pwm: float = MISSING
    dshot_offset: float = MISSING

    # filters:
    gyro_lowpass_freq: float = MISSING
    dterm_lowpass_freq: float = MISSING

    # omega:
    omega_cmd_sqrt: float = MISSING
    omega_cmd_lin: float = MISSING
    omega_volt: float = MISSING
    omega_offset: float = MISSING

    thrust_map: str = MISSING  # csv file path

    # Convert the collective-thrust command with the nominal mass instead of the
    # per-env randomized one, so mass randomization stays observable to the policy.
    use_nominal_mass: bool = False

    # Some Hardcoded parameters from Betaflight
    PWM_MAX_VAL: float = 2000
    PWM_MIN_VAL: float = 1000
    DSHOT_MAX_VAL: float = 2048
    DSHOT_MIN_VAL: float = 48
    SBUS_MIN_VAL: float = 192
    SBUS_MAX_VAL: float = 1792
    p_gain_scaling: float = 1.818e-3
    i_gain_scaling: float = 12.5e-3
    d_gain_scaling: float = -31.25e-6
    i_gain_limit: float = 0.5

    def __post_init__(self):
        self.PWM_RANGE = self.PWM_MAX_VAL - self.PWM_MIN_VAL
        self.DSHOT_RANGE = self.DSHOT_MAX_VAL - self.DSHOT_MIN_VAL
        self.SBUS_VAL_RANGE = self.SBUS_MAX_VAL - self.SBUS_MIN_VAL

        # Validation checks...
        assert (
            self.min_throttle_pwm >= self.PWM_MIN_VAL
        ), "min_throttle_pwm must be >= PWM_MIN_VAL"
        assert (
            self.max_throttle_pwm <= self.PWM_MAX_VAL
        ), "max_throttle_pwm must be <= PWM_MAX_VAL"
        assert (
            self.min_throttle_pwm < self.max_throttle_pwm
        ), "min_throttle_pwm must be < max_throttle_pwm"
        assert 0.0 <= self.dshot_offset <= 1.0, "dshot_offset must be in [0.0, 1.0]"
        assert (
            self.omega_cmd_sqrt > 0 and self.omega_cmd_lin > 0
        ), "omega_cmd_sqrt and omega_cmd_lin must be > 0"
        assert (
            self.omega_volt > 0 and self.omega_offset < 0
        ), "omega_volt must be > 0 and omega_offset must be < 0"


# ======================Simulation module======================
class LowLevelControllerBetaFlight(LowLevelControllerBase):
    """BetaFlight low-level controller – Fully GPU-accelerated (Set + Run)."""

    def __init__(
        self,
        params: BetaFlightParams,
        quad_params: QuadParamServer,
        quad_state: QuadState,
        cmd_tensor: torch.Tensor,  # <--- The persistent source tensor from CmdQueueWarp
        ctrl_dt: float,
        device: str = "cuda",
        dtype=torch.float32,
    ):
        super().__init__(quad_params, quad_state, device, dtype=dtype)
        self.params = params
        self.wp_dtype = wp.float32 if self.dtype == torch.float32 else wp.float64
        self.ctrl_dt_val = float(ctrl_dt)
        self.wp_device = device

        wp.init()

        # 1. Setup Internal Command Buffer
        # This is the destination for set_command and source for llc_run
        self.cmd = torch.zeros_like(cmd_tensor)

        # 2. Warp Handles
        self.wp_state = wp.from_torch(self.quad_state.state, dtype=self.wp_dtype)
        self.wp_ome = wp.from_torch(self.quad_state.ome, dtype=self.wp_dtype)
        # Mass the controller believes it is flying. mass_batch is mutated in
        # place by the randomizer, so the warp view stays live; the nominal
        # tensor is a separate constant buffer that is never written.
        if params.use_nominal_mass:
            self.llc_mass = torch.full_like(
                self.quad_params.mass_batch, self.quad_params.nominal_mass
            )
        else:
            self.llc_mass = self.quad_params.mass_batch
        self.wp_mass_batch = wp.from_torch(self.llc_mass, dtype=self.wp_dtype)
        self.wp_collective_thrust_min_batch = wp.from_torch(
            self.quad_params.collective_thrust_min_batch, dtype=self.wp_dtype
        )
        self.wp_collective_thrust_max_batch = wp.from_torch(
            self.quad_params.collective_thrust_max_batch, dtype=self.wp_dtype
        )

        # Source (Queue) and Destination (Controller) Handles
        self.wp_source_cmd = wp.from_torch(cmd_tensor, dtype=self.wp_dtype)
        self.wp_internal_cmd = wp.from_torch(self.cmd, dtype=self.wp_dtype)

        # 3. PID Gains & Limits
        self.kp = torch.tensor(
            [params.kp_roll, params.kp_pitch, params.kp_yaw], device=device, dtype=dtype
        )
        self.ki = torch.tensor(
            [params.ki_roll, params.ki_pitch, params.ki_yaw], device=device, dtype=dtype
        )
        self.omega_max = float(quad_params.motor_omega_max)
        self.omega_min = float(quad_params.motor_omega_min)
        self.body_omega_max = torch.tensor(
            quad_params.omega_max, device=device, dtype=dtype
        )

        # 4. Thrust Map
        tmap_file = os.path.join(
            os.path.dirname(rpg_assets.__file__), params.thrust_map
        )
        assert os.path.isfile(tmap_file), f"Thrust map file {tmap_file} does not exist."
        self.thrust_map_obj = ThrustMap(tmap_file, device=device, dtype=dtype)
        self.wp_thrust_map = wp.from_torch(
            self.thrust_map_obj.thrust_map.contiguous(), dtype=self.wp_dtype
        )

        # 5. Per-instance param table + kernels
        self._build_params()
        kernels = K.KERNELS[dtype]
        self.set_cmd_kernel = kernels["set_cmd_kernel"]
        self.llc_kernel = kernels["llc_kernel"]
        ensure_loaded(self.wp_device, dtype, self.set_cmd_kernel.module, self.llc_kernel.module)

        # =================================================================
        # GRAPH 1: SET COMMAND (Clamping & Copying)
        # =================================================================
        self.set_cmd_graph = LaunchGraph(self.wp_device, self._launch_set_cmd)
        self.llc_graph = LaunchGraph(self.wp_device, self._launch_llc)

    def _launch_set_cmd(self):
        wp.launch(
            kernel=self.set_cmd_kernel,
            dim=self.num_quads,
            inputs=[
                self.wp_source_cmd,
                self.wp_internal_cmd,
                self.wp_collective_thrust_min_batch,
                self.wp_collective_thrust_max_batch,
                self.params_handle,
            ],
            device=self.wp_device,
        )

    def _launch_llc(self):
        wp.launch(
            kernel=self.llc_kernel,
            dim=self.num_quads,
            inputs=[
                self.wp_internal_cmd,
                self.wp_ome,
                self.wp_state,
                self.wp_thrust_map,
                self.wp_mass_batch,
                self.params_handle,
            ],
            device=self.wp_device,
        )

    def set_command(self, cmd=None, env_ids=None):
        """
        Execute command safety layer via CUDA graph replay.
        Note: Arguments are ignored as the graph is bound to the persistent tensors.
        """
        self.set_cmd_graph()

    def _build_params(self):
        """Fill the per-instance param table read by the kernels at runtime."""
        p = self.params
        tm = self.thrust_map_obj
        t = torch.zeros(K.P_SIZE, device=self.device, dtype=self.dtype)

        t[K.P_OMEGA_MAX_X] = self.body_omega_max[0].item()
        t[K.P_OMEGA_MAX_Y] = self.body_omega_max[1].item()
        t[K.P_OMEGA_MAX_Z] = self.body_omega_max[2].item()
        t[K.P_NEG_OMEGA_MAX_X] = -self.body_omega_max[0].item()
        t[K.P_NEG_OMEGA_MAX_Y] = -self.body_omega_max[1].item()
        t[K.P_NEG_OMEGA_MAX_Z] = -self.body_omega_max[2].item()

        t[K.P_THRUST_MIN] = float(tm.thrust_min.item())
        t[K.P_VOLTAGE_MIN] = float(tm.voltage_min.item())
        t[K.P_T_SCALE_THRUST] = float(tm.n_col / (tm.thrust_max.item() - tm.thrust_min.item()))
        t[K.P_T_SCALE_VOLTAGE] = float(tm.n_row / (tm.voltage_max.item() - tm.voltage_min.item()))
        t[K.P_INV_DELTA_THRUST] = float(1.0 / tm.delta_thrust)
        t[K.P_INV_DELTA_VOLTAGE] = float(1.0 / tm.delta_voltage)
        t[K.P_MAX_I_T] = tm.n_col - 2
        t[K.P_MAX_I_V] = tm.n_row - 2

        t[K.P_SBUS_MIN_VAL] = float(p.SBUS_MIN_VAL)
        t[K.P_INV_SBUS_RANGE] = float(1.0 / p.SBUS_VAL_RANGE)
        t[K.P_BFL_DSHOT_OFFSET] = float(p.dshot_offset)
        t[K.P_INV_BFL_DSHOT_RANGE] = float(1.0 / (1.0 - p.dshot_offset))

        _min_throttle_f = float((p.min_throttle_pwm - p.PWM_MIN_VAL) / p.PWM_RANGE)
        _max_throttle_f = float((p.max_throttle_pwm - p.PWM_MIN_VAL) / p.PWM_RANGE)
        _dshot_offset_f = float(p.dshot_offset * p.DSHOT_RANGE)
        _dshot_slope_f = float(p.DSHOT_MAX_VAL - _dshot_offset_f)
        t[K.P_DSHOT_OFFSET] = _dshot_offset_f
        t[K.P_DSHOT_SLOPE] = _dshot_slope_f
        t[K.P_MIN_DS] = _min_throttle_f * _dshot_slope_f + _dshot_offset_f
        t[K.P_MAX_DS] = _max_throttle_f * _dshot_slope_f + _dshot_offset_f

        t[K.P_CTRL_DT] = float(self.ctrl_dt_val)
        t[K.P_I_GAIN_LIMIT] = float(p.i_gain_limit)
        t[K.P_NEG_I_GAIN_LIMIT] = -float(p.i_gain_limit)

        t[K.P_KP_S0] = float(p.kp_roll * p.p_gain_scaling)
        t[K.P_KP_S1] = float(p.kp_pitch * p.p_gain_scaling)
        t[K.P_KP_S2] = float(p.kp_yaw * p.p_gain_scaling)
        t[K.P_KI_S0] = float(p.ki_roll * p.i_gain_scaling)
        t[K.P_KI_S1] = float(p.ki_pitch * p.i_gain_scaling)
        t[K.P_KI_S2] = float(p.ki_yaw * p.i_gain_scaling)

        # allocation matrix rows (throttle, roll, pitch, yaw)
        B = [[1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, -1.0], [1.0, -1.0, 1.0, 1.0], [1.0, 1.0, -1.0, 1.0]]
        for i, base in enumerate((K.P_B0, K.P_B1, K.P_B2, K.P_B3)):
            t[base : base + 4] = torch.tensor(B[i], device=self.device, dtype=self.dtype)

        t[K.P_OMEGA_OFFSET] = float(p.omega_offset)
        t[K.P_OMEGA_VOLT] = float(p.omega_volt)
        t[K.P_OMEGA_CMD_LIN] = float(p.omega_cmd_lin)
        t[K.P_OMEGA_CMD_SQRT] = float(p.omega_cmd_sqrt)
        t[K.P_OMEGA_MIN] = float(self.omega_min)
        t[K.P_OMEGA_MAX] = float(self.omega_max)

        self.param_table = t
        self.params_handle = wp.from_torch(self.param_table, dtype=self.wp_dtype)

    def run(self):
        """Execute low-level controller logic via CUDA graph replay."""
        self.llc_graph()
