import torch
import warp as wp

from rpg_simulator.quad_state import QuadState
from rpg_simulator.utils.graph import LaunchGraph


class ModelInit:
    """Warp-accelerated initialization module (Zero-Allocation, CUDA Graph)."""

    def __init__(
        self,
        quad_state: QuadState,
        num_env: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):

        self.device = device
        self.num_env = num_env
        self.wp_dtype = wp.float32 if dtype == torch.float32 else wp.float64

        wp.init()

        self.is_agisim = quad_state.using_agisim

        # Map Memory Handles (Zero-Allocation)
        self.deriv_handle = wp.from_torch(quad_state.derivative, dtype=self.wp_dtype)

        if not self.is_agisim:
            self.force_torque_handle = wp.from_torch(
                quad_state.force_torque, dtype=self.wp_dtype
            )

        self._graph = None

    def cuda_launch(self, quad_state=None):
        # Main function, allows Master Capture in parent graph
        self.deriv_handle.zero_()
        if not self.is_agisim:
            self.force_torque_handle.zero_()

    def run(self, quad_state: QuadState):
        """Execute hardware memory clearing via CUDA graph replay."""
        if self._graph is None:
            self._graph = LaunchGraph(self.device, lambda: self.cuda_launch(quad_state))
        self._graph()
