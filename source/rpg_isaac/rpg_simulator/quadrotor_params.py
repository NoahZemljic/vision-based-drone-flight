from __future__ import annotations

from dataclasses import MISSING
from isaaclab.utils.configclass import configclass
import torch
import warp as wp

CELL_VOLTAGE_FULL: float = 4.12  # [V] full LiPo cell, u_0 in every agilicious pack
BATTERY_TYPES: dict[str, int] = {"4s": 4, "4S": 4, "6s": 6, "6S": 6}


@configclass
class QuadrotorParams:
    mass: float = MISSING  # [kg]

    # Motor positions relative to center of mass [m]
    tbm_fr: list[float] = MISSING  # Front right motor position
    tbm_bl: list[float] = MISSING  # Back left motor position
    tbm_bl: list[float] = MISSING  # Back left motor position
    tbm_br: list[float] = MISSING  # Back right motor position
    tbm_fl: list[float] = MISSING  # Front left motor position

    inertia: list[float] = MISSING  # Inertia matrix diagonal entries [kgm^2]

    # Motor specifications
    motor_omega_min: float = MISSING  # [rad/s]
    motor_omega_max: float = MISSING  # [rad/s]
    motor_tau: float = MISSING  # Motor time constant [s]
    omega_max: list[float] = MISSING  # Max angular velocity [rad/s]

    # Thrust and torque
    thrust_map: list[float] = MISSING  # Thrust coefficients [N/(rad/s)^2, N/(rad/s), N]
    kappa: float = MISSING  # Torque-to-thrust ratio [Nm/N]
    thrust_min: float = MISSING  # [N]
    thrust_max: float = MISSING  # [N] per motor
    # Per-motor thrust the policy is allowed to command, when the airframe is
    # flown below what its motors can deliver. None commands up to thrust_max.
    derated_thrust_max: float | None = None  # [N] per motor
    rotors_config: str = MISSING  # "cross" or "plus"

    # Battery
    battery_type: str = MISSING  # cells in series, one of BATTERY_TYPES

    # Aerodynamics
    aero_coeff_1: list[float] = MISSING  # [N/(v/m)]
    aero_coeff_3: list[float] = MISSING  # [N/(v/m)]
    aero_coeff_h: float = MISSING  # [N/(v^2/m^2)]

    # Scaling factors
    scale_thrust: float = MISSING
    scale_torque: float = MISSING
    scale_drag: float = MISSING

    # Methods to compute derived properties
    @property
    def J(self) -> torch.Tensor:
        """Inertia matrix diagonal entries (same as inertia)."""
        assert len(self.inertia) == 3, "Inertia must be a list of three elements."
        return torch.diag(torch.tensor(self.inertia, dtype=torch.float64))

    @property
    def B_allocation(self) -> torch.Tensor:
        """Control allocation matrix B."""
        assert self.rotors_config == "cross", "Only 'cross' rotors_config is supported."
        rotor_coords = torch.stack(
            [
                torch.tensor(self.tbm_fr, dtype=torch.float64),
                torch.tensor(self.tbm_bl, dtype=torch.float64),
                torch.tensor(self.tbm_br, dtype=torch.float64),
                torch.tensor(self.tbm_fl, dtype=torch.float64),
            ]
        )

        rotor_coords = torch.stack(
            [
                torch.tensor(self.tbm_fr, dtype=torch.float64),
                torch.tensor(self.tbm_bl, dtype=torch.float64),
                torch.tensor(self.tbm_br, dtype=torch.float64),
                torch.tensor(self.tbm_fl, dtype=torch.float64),
            ]
        )

        x = rotor_coords[:, 0]
        y = rotor_coords[:, 1]

        return torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [y[0], y[1], y[2], y[3]],
                [-x[0], -x[1], -x[2], -x[3]],
                [-self.kappa, -self.kappa, self.kappa, self.kappa],
            ],
            dtype=torch.float64,
        )

    @property
    def B_allocation_inv(self) -> torch.Tensor:
        """Inverse of control allocation matrix B."""
        return torch.linalg.inv(self.B_allocation)

    def __post_init__(self):
        self.collective_thrust_min = 4.0 * self.thrust_min / self.mass
        self.collective_thrust_max = 4.0 * self.thrust_max / self.mass

        if self.battery_type not in BATTERY_TYPES:
            raise ValueError(
                f"Unsupported battery_type {self.battery_type!r}; supported types"
                f" are {sorted(BATTERY_TYPES)}"
            )
        self.cell_series = BATTERY_TYPES[self.battery_type]
        self.rest_voltage = self.cell_series * CELL_VOLTAGE_FULL


class QuadParamServer:
    """
    A simple host of quad param tensors which can be conveniently randomized for training.
    """

    def __init__(
        self,
        quad_params: QuadrotorParams,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.device = device
        self.dtype = dtype

        self.nominal_mass = quad_params.mass
        self.tbm_fr = quad_params.tbm_fr
        self.tbm_bl = quad_params.tbm_bl
        self.tbm_br = quad_params.tbm_br
        self.tbm_fl = quad_params.tbm_fl
        self.nominal_inertia = torch.tensor(
            quad_params.inertia, device=self.device, dtype=self.dtype
        )
        self.motor_omega_min = quad_params.motor_omega_min
        self.motor_omega_max = quad_params.motor_omega_max
        self.motor_tau = quad_params.motor_tau
        self.omega_max = quad_params.omega_max
        self.thrust_map = quad_params.thrust_map
        self.kappa = quad_params.kappa
        self.thrust_min = quad_params.thrust_min
        self.thrust_max = quad_params.thrust_max
        self.aero_coeff_1 = quad_params.aero_coeff_1
        self.aero_coeff_3 = quad_params.aero_coeff_3
        self.aero_coeff_h = quad_params.aero_coeff_h
        self.nominal_scale_thrust = quad_params.scale_thrust
        self.nominal_scale_torque = quad_params.scale_torque
        self.nominal_scale_drag = quad_params.scale_drag
        self.collective_thrust_min = quad_params.collective_thrust_min
        self.collective_thrust_max = quad_params.collective_thrust_max

        self.B_allocation = quad_params.B_allocation.to(device, dtype=dtype)
        self.B_allocation_inv = quad_params.B_allocation_inv.to(device, dtype=dtype)

        # mass related tensors
        self.mass_batch = torch.tensor(
            quad_params.mass, device=device, dtype=dtype
        ).repeat(num_envs)
        self.collective_thrust_min_batch = torch.tensor(
            quad_params.collective_thrust_min, device=device, dtype=dtype
        ).repeat(num_envs)
        self.collective_thrust_max_batch = torch.tensor(
            quad_params.collective_thrust_max, device=device, dtype=dtype
        ).repeat(num_envs)

        self.inertia_batch = torch.tensor(
            quad_params.inertia, device=device, dtype=dtype
        ).repeat(num_envs, 1)
        self.J_batch = torch.diag_embed(self.inertia_batch)
        self.J_inv_batch = torch.diag_embed(
            1 / self.inertia_batch
        )  # Assuming diagonal inertia, so inverse is just reciprocal of diagonal entries
        self.scale_drag_batch = torch.tensor(
            quad_params.scale_drag, device=device, dtype=dtype
        ).repeat(num_envs)
        self.scale_thrust_batch = torch.tensor(
            quad_params.scale_thrust, device=device, dtype=dtype
        ).repeat(num_envs)
        self.scale_torque_batch = torch.tensor(
            quad_params.scale_torque, device=device, dtype=dtype
        ).repeat(num_envs)

    def apply_random_quad_mass(
        self,
        mass_random_coeff: float,
        env_ids: torch.Tensor | None = None,
    ):
        """
        Applies random mass accross environments
        random_mass = nominal_mass * (1 + coeff * Uniform(-1, 1))

        If env_ids is provided, only applies to those envs. Only the rpg-side
        batches are updated; mirroring into a host engine is the caller's job
        (IsaacLab: IsaacBridge.push_body_params).
        """
        if mass_random_coeff == 0.0:
            return

        if env_ids is None:
            env_ids = torch.arange(len(self.mass_batch), device=self.device)
        random_factor = 1.0 + mass_random_coeff * (
            2.0 * torch.rand(env_ids.numel(), device=self.device, dtype=self.dtype)
            - 1.0
        )
        self.mass_batch[env_ids] = self.nominal_mass * random_factor
        self.collective_thrust_min_batch[env_ids] = (
            4.0 * self.thrust_min / self.mass_batch[env_ids]
        )
        self.collective_thrust_max_batch[env_ids] = (
            4.0 * self.thrust_max / self.mass_batch[env_ids]
        )

    def apply_random_quad_inertia(
        self,
        inertia_random_coeff: float,
        env_ids: torch.Tensor | None = None,
    ):
        """
        Applies random inertia accross environments
        random_inertia = nominal_inertia * (1 + coeff * Uniform(-1, 1)), sampled
        independently per axis (agilicious draws a fresh value per diagonal entry).
        """
        if inertia_random_coeff == 0.0:
            return

        if env_ids is None:
            env_ids = torch.arange(len(self.mass_batch), device=self.device)
        random_factor = 1.0 + inertia_random_coeff * (
            2.0 * torch.rand(env_ids.numel(), 3, device=self.device, dtype=self.dtype)
            - 1.0
        )
        self.inertia_batch[env_ids] = self.nominal_inertia * random_factor
        self.J_batch.copy_(torch.diag_embed(self.inertia_batch))
        self.J_inv_batch.copy_(torch.diag_embed(1 / self.inertia_batch))

    def apply_random_drag(
        self, drag_random_coeff: float, env_ids: torch.Tensor | None = None
    ):
        """
        Applies random drag scaling accross environments
        random_drag_scale = nominal_drag_scale * (1 + coeff * Uniform(-1, 1))
        """
        if drag_random_coeff == 0.0:
            return

        if env_ids is None:
            env_ids = torch.arange(len(self.scale_drag_batch), device=self.device)
        random_factor = 1.0 + drag_random_coeff * (
            2.0 * torch.rand(env_ids.numel(), device=self.device, dtype=self.dtype)
            - 1.0
        )
        self.scale_drag_batch[env_ids] = self.nominal_scale_drag * random_factor

    def apply_random_thrust(
        self, thrust_random_coeff: float, env_ids: torch.Tensor | None = None
    ):
        """
        Applies random thrust scaling accross environments
        random_thrust_scale = nominal_thrust_scale * (1 + coeff * Uniform(-1, 1))
        """
        if thrust_random_coeff == 0.0:
            return

        if env_ids is None:
            env_ids = torch.arange(len(self.scale_thrust_batch), device=self.device)
        random_factor = 1.0 + thrust_random_coeff * (
            2.0 * torch.rand(env_ids.numel(), device=self.device, dtype=self.dtype)
            - 1.0
        )
        self.scale_thrust_batch[env_ids] = self.nominal_scale_thrust * random_factor

    def apply_random_torque(
        self, torque_random_coeff: float, env_ids: torch.Tensor | None = None
    ):
        """
        Applies random torque scaling accross environments
        random_torque_scale = nominal_torque_scale * (1 + coeff * Uniform(-1, 1))
        """
        if torque_random_coeff == 0.0:
            return

        if env_ids is None:
            env_ids = torch.arange(len(self.scale_torque_batch), device=self.device)
        random_factor = 1.0 + torque_random_coeff * (
            2.0 * torch.rand(env_ids.numel(), device=self.device, dtype=self.dtype)
            - 1.0
        )
        self.scale_torque_batch[env_ids] = self.nominal_scale_torque * random_factor
