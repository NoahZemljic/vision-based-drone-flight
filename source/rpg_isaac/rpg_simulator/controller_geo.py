import torch
import warp as wp

from dataclasses import MISSING
from isaaclab.utils.configclass import configclass

from rpg_simulator.kernels import controller_geo as K
from rpg_simulator.kernels.registry import ensure_loaded
from rpg_simulator.quadrotor_params import QuadrotorParams
from rpg_simulator.quad_state import QuadState
from rpg_simulator.utils.graph import LaunchGraph


@configclass
class GeoControllerParams:
    kp_acc: torch.tensor = torch.tensor([10.0, 10.0, 20.0], dtype=torch.float64)
    kd_acc: torch.tensor = torch.tensor([4.0, 4.0, 6.0], dtype=torch.float64)
    kp_rate: torch.tensor = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    kp_att_xy: float = 10.0
    kp_att_z: float = 2.0
    filter_sampling_frequency: float = 100.0
    filter_cutoff_frequency: float = 6.0

    p_err_max: torch.tensor = torch.tensor([0.6, 0.6, 0.3], dtype=torch.float64)
    v_err_max: torch.tensor = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)


class GeometricController:
    """Warp-accelerated Geometric controller (Zero-Allocation, Fused Kernel)."""

    def __init__(
        self,
        geo_params: GeoControllerParams,
        quad_params: QuadrotorParams,
        quad_state: QuadState,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):

        self.device = device
        self.num_quads = quad_state.num_quads
        self.dtype = dtype
        self.wp_dtype = wp.float32 if dtype == torch.float32 else wp.float64

        wp.init()

        # Define Dynamic Warp Vector Types to match PyTorch dtype (float32 vs float64)
        self._vec3 = wp.types.vector(length=3, dtype=self.wp_dtype)
        self._vec4 = wp.types.vector(length=4, dtype=self.wp_dtype)

        # -----------------------------------------------------------
        # 1. Memory Allocation (Input Buffers)
        # -----------------------------------------------------------
        # We pre-allocate buffers for the desired state.
        # get_command() will copy data here before the kernel runs.
        self.pos_des = torch.zeros((self.num_quads, 3), device=device, dtype=dtype)
        self.vel_des = torch.zeros((self.num_quads, 3), device=device, dtype=dtype)
        self.acc_des = torch.zeros((self.num_quads, 3), device=device, dtype=dtype)
        self.att_des = torch.zeros((self.num_quads, 4), device=device, dtype=dtype)
        self.att_des[:, 0] = 1.0  # Init identity quaternion [1, 0, 0, 0]

        # Output Command Buffer (Thrust, Omega_x, Omega_y, Omega_z)
        self.cmd = torch.zeros((self.num_quads, 4), device=device, dtype=dtype)

        # -----------------------------------------------------------
        # 2. Warp Handles
        # -----------------------------------------------------------
        # Note: We must use the dynamic vector types (self._vec3) to wrap float64 tensors
        self.wp_pos_des = wp.from_torch(self.pos_des, dtype=self._vec3)
        self.wp_vel_des = wp.from_torch(self.vel_des, dtype=self._vec3)
        self.wp_acc_des = wp.from_torch(self.acc_des, dtype=self._vec3)
        self.wp_att_des = wp.from_torch(self.att_des, dtype=self._vec4)
        self.wp_cmd = wp.from_torch(self.cmd, dtype=self._vec4)

        # QuadState handles
        # We assume quad_state has contiguous tensors we can wrap
        self.wp_pos = wp.from_torch(quad_state.pos, dtype=self._vec3)
        self.wp_vel = wp.from_torch(quad_state.vel, dtype=self._vec3)
        self.wp_att = wp.from_torch(quad_state.att, dtype=self._vec4)

        # -----------------------------------------------------------
        # 3. Per-instance gains (read by the kernel) + kernel
        # -----------------------------------------------------------
        gp = geo_params
        self.params = torch.zeros(K.P_SIZE, device=device, dtype=dtype)
        self.params[K.P_KP_ACC : K.P_KP_ACC + 3] = torch.as_tensor(gp.kp_acc, dtype=dtype)
        self.params[K.P_KD_ACC : K.P_KD_ACC + 3] = torch.as_tensor(gp.kd_acc, dtype=dtype)
        self.params[K.P_P_ERR_MAX : K.P_P_ERR_MAX + 3] = torch.as_tensor(gp.p_err_max, dtype=dtype)
        self.params[K.P_V_ERR_MAX : K.P_V_ERR_MAX + 3] = torch.as_tensor(gp.v_err_max, dtype=dtype)
        self.params[K.P_T_ATT_X] = gp.kp_att_xy
        self.params[K.P_T_ATT_Y] = gp.kp_att_xy
        self.params[K.P_T_ATT_Z] = gp.kp_att_z
        self.params_handle = wp.from_torch(self.params, dtype=self.wp_dtype)

        self.ctrl_kernel = K.KERNELS[dtype]["geo_ctrl_kernel"]
        ensure_loaded(self.device, dtype, self.ctrl_kernel.module)

        # Graph capture (Lazy init)
        self._graph = None

    def _launch(self):
        wp.launch(
            kernel=self.ctrl_kernel,
            dim=self.num_quads,
            inputs=[
                self.wp_pos,
                self.wp_vel,
                self.wp_att,
                self.wp_pos_des,
                self.wp_vel_des,
                self.wp_acc_des,
                self.wp_att_des,
                self.wp_cmd,
                self.params_handle,
            ],
            device=self.device,
        )

    def get_command(
        self,
        pos_des=None,
        vel_des=None,
        acc_des=None,
        att_des=None,
        ome_des=None,
        alpha_des=None,
    ):
        """
        Compute control command using Warp kernel.
        Inputs should be PyTorch tensors.
        """
        # 1. Update Input Buffers (Async Copy)
        # Note: We use .copy_() or similar to update the pre-allocated tensors
        # that our Warp handles point to.

        # Helper to handle None inputs
        if pos_des is not None:
            self.pos_des.copy_(pos_des)
        else:
            self.pos_des.zero_()

        if vel_des is not None:
            self.vel_des.copy_(vel_des)
        else:
            self.vel_des.zero_()

        if acc_des is not None:
            self.acc_des.copy_(acc_des)
        else:
            self.acc_des.zero_()

        if att_des is not None:
            self.att_des.copy_(att_des)
        else:
            self.att_des.zero_()
            self.att_des[:, 0] = 1.0

        # 2. Graph Capture (Lazy)
        if self._graph is None:
            self._graph = LaunchGraph(self.device, self._launch)
        self._graph()

        # 4. Return Output
        # The output is directly in self.cmd (PyTorch tensor) due to zero-copy
        return self.cmd
