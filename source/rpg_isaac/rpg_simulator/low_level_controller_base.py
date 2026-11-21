from __future__ import annotations

# make abstract base class for llc
from abc import ABC, abstractmethod
from rpg_simulator.quad_state import QuadState
from rpg_simulator.quadrotor_params import QuadParamServer
import torch


class LowLevelControllerBase(ABC):
    """Abstract base class for low-level controllers."""

    def __init__(
        self,
        quad_params: QuadParamServer,
        quad_state: QuadState,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.quad_params = quad_params
        self.quad_state = quad_state
        self.device = device
        self.dtype = dtype
        self.num_quads = quad_state.num_quads
        self.cmd = torch.zeros(
            (self.num_quads, 4), device=device, dtype=dtype
        )  # collective thrust and body rates
        self.motor_omega_des = torch.zeros(
            (self.num_quads, 4), device=device, dtype=dtype
        )

    @abstractmethod
    def set_command(self, cmd: torch.Tensor, env_ids: torch.Tensor) -> None:
        pass

    def get_motor_commands(self) -> torch.Tensor:
        self.run()
        # assert torch.isfinite(self.motor_omega_des).all(), "Non-finite motor commands computed!"

    @abstractmethod
    def run(self) -> torch.Tensor:
        """Compute motor speed commands given desired thrust and torques.

        Args:
            body_w (torch.Tensor): Current body angular velocities (num_quads, 3).
            ctbr_cmd (torch.Tensor): Desired collective thrust and body rates (num_quads, 4).

        Returns:
            torch.Tensor: Motor speed commands (num_quads, 4).
        """
        pass
