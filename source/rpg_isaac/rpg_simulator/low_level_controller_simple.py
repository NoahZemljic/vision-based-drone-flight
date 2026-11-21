from __future__ import annotations

from dataclasses import MISSING
from isaaclab.utils.configclass import configclass
import torch

from rpg_simulator.quadrotor_params import QuadParamServer
from rpg_simulator.low_level_controller_base import LowLevelControllerBase
from rpg_simulator.quad_state import QuadState


# ======================Simulation module parameters======================
@configclass
class LowLevelControllerSimpleParams:
    """Configuration for a simple low-level controller."""

    p_roll: float = 20.0
    p_pitch: float = 20.0
    p_yaw: float = 41.0


# ======================Simulation module======================
class LowLevelControllerSimple(LowLevelControllerBase):
    """A simple low-level controller for quadrotors."""

    def __init__(
        self,
        params: LowLevelControllerSimpleParams,
        quad_params: QuadParamServer,
        quad_state: QuadState,
        cmd_tensor: torch.Tensor,
        device: str = "cuda",
        dtype=torch.float32,
    ):
        super().__init__(quad_params, quad_state, device, dtype=dtype)
        self.Kinv_ang_vel_tau = torch.diag(
            torch.tensor([params.p_roll, params.p_pitch, params.p_yaw], dtype=dtype)
        ).to(device)
        self.J = torch.diag(quad_params.nominal_inertia).to(device).to(dtype)
        self.B_allocation_inv = quad_params.B_allocation_inv.to(device).to(dtype)
        self.mass = quad_params.nominal_mass
        self.thrust_map = torch.tensor(
            quad_params.thrust_map, device=device, dtype=dtype
        )
        self.act_min = torch.tensor(
            [
                4.0 * quad_params.thrust_min / quad_params.nominal_mass,
                -quad_params.omega_max[0],
                -quad_params.omega_max[1],
                -quad_params.omega_max[2],
            ],
            dtype=dtype,
        ).to(device)
        self.act_max = torch.tensor(
            [
                4.0 * quad_params.thrust_max / quad_params.nominal_mass,
                quad_params.omega_max[0],
                quad_params.omega_max[1],
                quad_params.omega_max[2],
            ],
            dtype=dtype,
        ).to(device)
        self.thrust_min, self.thrust_max = (
            quad_params.thrust_min,
            quad_params.thrust_max,
        )

        self.cmd_tensor = cmd_tensor

    def set_command(self, env_ids) -> None:
        """Set the desired collective thrust and body rates command.

        Args:
            cmd (torch.Tensor): Desired collective thrust and body rates (num_env, 4).
        """
        # assert torch.isfinite(cmd).all(), "Command contains non-finite values."
        cmd = torch.clamp(self.cmd_tensor[env_ids], self.act_min, self.act_max)
        self.cmd[env_ids] = cmd

    def run(self) -> torch.Tensor:
        """Compute motor speed commands given desired thrust and torques.

        This is a wrapper around the JIT-compiled implementation for better performance.

        Args:
            body_w: Current body angular velocity in body frame, shape (num_env, 3).
            ctbr_cmd: Desired collective thrust and body rates, shape (num_env, 4).
        Returns:
            Motor speed commands, shape (num_env, 4).
        """

        body_w = self.quad_state.ome

        # Call the JIT-compiled static method with instance variables
        self.motor_omega_des = self._compute_motor_commands_jit(
            body_w=body_w,
            ctbr_cmd=self.cmd,
            mass=self.mass,
            J=self.J,
            Kinv_ang_vel_tau=self.Kinv_ang_vel_tau,
            B_allocation_inv=self.B_allocation_inv,
            thrust_min=self.thrust_min,
            thrust_max=self.thrust_max,
            thrust_map=self.thrust_map,
        )
        self.quad_state.motdes.copy_(self.motor_omega_des)

    def motor_thrust_to_omega(self, motor_thrusts):
        scale = 1.0 / (2.0 * self.thrust_map[0])
        offset = -self.thrust_map[1] / scale
        root = torch.sqrt(
            self.thrust_map[1] ** 2
            - 4.0 * self.thrust_map[0] * (self.thrust_map[2] - motor_thrusts)
        )
        return offset + scale * root

    def clamp_ctbr(self, ctbr: torch.Tensor) -> torch.Tensor:
        """
        Clamp collective thrust body rate.
        Called only when new command arrives.
        """
        return torch.clamp(ctbr, self.act_min, self.act_max)

    # @staticmethod
    @torch.jit.script
    def _compute_motor_commands_jit(
        body_w: torch.Tensor,
        ctbr_cmd: torch.Tensor,
        mass: float,
        J: torch.Tensor,
        Kinv_ang_vel_tau: torch.Tensor,
        B_allocation_inv: torch.Tensor,
        thrust_min: float,
        thrust_max: float,
        thrust_map: torch.Tensor,
    ) -> torch.Tensor:
        """JIT-compiled motor command computation for performance.

        Args:
            body_w: Current body angular velocity in body frame, shape (num_env, 3).
            ctbr_cmd: Desired collective thrust and body rates, shape (num_env, 4).
            mass: Quadrotor mass [kg].
            J: Inertia matrix, shape (3, 3).
            Kinv_ang_vel_tau: Angular velocity control gains, shape (3, 3).
            B_allocation_inv: Inverse control allocation matrix, shape (4, 4).
            thrust_min: Minimum thrust per motor [N].
            thrust_max: Maximum thrust per motor [N].
            thrust_map: Thrust map coefficients, shape (3,).

        Returns:
            Motor speed commands, shape (num_env, 4).
        """
        # Compute forces and torques
        force = mass * ctbr_cmd[:, 0:1]  # (num_env, 1)
        omega_err = ctbr_cmd[:, 1:4] - body_w  # (num_env, 3)

        # Body torque computation
        # Note: Using batch matrix multiplication for vectorized computation
        body_torque_des = torch.bmm(
            torch.bmm(
                J.unsqueeze(0).expand(body_w.shape[0], -1, -1),
                Kinv_ang_vel_tau.unsqueeze(0).expand(body_w.shape[0], -1, -1),
            ),
            omega_err.unsqueeze(-1),
        ).squeeze(-1) + torch.cross(
            body_w,
            torch.bmm(
                J.unsqueeze(0).expand(body_w.shape[0], -1, -1), body_w.unsqueeze(-1)
            ).squeeze(-1),
            dim=1,
        )

        # Combine thrust and torque
        thrust_torque = torch.cat([force, body_torque_des], dim=1)  # (num_env, 4)

        # Convert to motor thrusts
        motor_thrusts = torch.bmm(
            B_allocation_inv.unsqueeze(0).expand(body_w.shape[0], -1, -1),
            thrust_torque.unsqueeze(-1),
        ).squeeze(
            -1
        )  # (num_env, 4)

        # Clamp motor thrusts
        motor_thrusts = torch.clamp(motor_thrusts, thrust_min, thrust_max)

        # Convert motor thrusts to angular velocities
        scale = 1.0 / (2.0 * thrust_map[0])
        offset = -thrust_map[1] / scale
        root = torch.sqrt(
            thrust_map[1] ** 2 - 4.0 * thrust_map[0] * (thrust_map[2] - motor_thrusts)
        )
        motor_omegas = offset + scale * root

        return motor_omegas
