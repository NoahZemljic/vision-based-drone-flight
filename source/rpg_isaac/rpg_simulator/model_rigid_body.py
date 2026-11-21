import torch
import warp as wp

from rpg_simulator.kernels import model_rigid_body as K
from rpg_simulator.kernels.registry import ensure_loaded
from rpg_simulator.quad_state import QuadState
from rpg_simulator.quadrotor_params import QuadParamServer
from rpg_simulator.utils.graph import LaunchGraph


class ModelRigidBody:
    """Warp-accelerated model for rigid body dynamics (Zero-Allocation, CUDA Graph).
    Only used in conjunction with custom integrators (RK4, Euler) that do not rely on Isaac Sim.
    """

    def __init__(
        self,
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

        wp.init()

        # 1. Define Vector Types
        self._vec3_type = wp.types.vector(length=3, dtype=self.wp_dtype)
        self._vec4_type = wp.types.vector(length=4, dtype=self.wp_dtype)

        # Per-env inertia tensor handles [N,3,3]
        self.wp_J = wp.from_torch(quad_params.J_batch, dtype=self.wp_dtype)
        self.wp_J_inv = wp.from_torch(quad_params.J_inv_batch, dtype=self.wp_dtype)

        # 2. Map Memory Handles (Zero-Allocation)
        self.vel_handle = wp.from_torch(quad_state.vel, dtype=self._vec3_type)
        self.att_handle = wp.from_torch(quad_state.att, dtype=self._vec4_type)
        self.ome_handle = wp.from_torch(quad_state.ome, dtype=self._vec3_type)

        self.d_pos_handle = wp.from_torch(quad_state.d_pos, dtype=self._vec3_type)
        self.d_att_handle = wp.from_torch(quad_state.d_att, dtype=self._vec4_type)
        self.d_ome_handle = wp.from_torch(quad_state.d_ome, dtype=self._vec3_type)

        # 3. Pick the kernel (no per-instance scalars: J comes in per env)
        self.rb_kernel = K.KERNELS[dtype]["rb_kernel"]
        ensure_loaded(self.device, dtype, self.rb_kernel.module)

        self._graph = None

    def cuda_launch(self, quad_state: QuadState):
        wp.launch(
            kernel=self.rb_kernel,
            dim=self.num_env,
            inputs=[
                self.vel_handle,
                self.att_handle,
                self.ome_handle,
                self.d_pos_handle,
                self.d_att_handle,
                self.d_ome_handle,
                self.wp_J,
                self.wp_J_inv,
            ],
            device=self.device,
        )

    def run(self, quad_state: QuadState):
        if self._graph is None:
            self._graph = LaunchGraph(self.device, lambda: self.cuda_launch(quad_state))
        self._graph()
