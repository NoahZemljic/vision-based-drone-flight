from __future__ import annotations

from collections import OrderedDict

from dataclasses import MISSING
from isaaclab.utils.configclass import configclass
from typing import Union

from rpg_simulator.quadrotor_params import QuadrotorParams
from rpg_simulator.low_level_controller_betaflight import BetaFlightParams
from rpg_simulator.low_level_controller_simple import LowLevelControllerSimpleParams


@configclass
class QuadRandomizationCfg:
    """Domain randomization of the simulator: physical params, command delay
    and initial-state perturbations. Consumed by
    :meth:`QuadrotorSimulator.randomize_params` / :meth:`perturb_init_state`."""

    random_mass: float = 0.0
    random_inertia: float = 0.0
    random_drag: float = 0.0
    random_thrust_map: float = 0.0
    random_torque: float = 0.0
    random_delay: float = 0.0
    # Seed for the command-latency draw ONLY. When > 0, the [0, random_delay)
    # sample uses a dedicated torch.Generator re-seeded to this value at the start
    # of every episode, so every run (every model, every eval seed) draws the SAME
    # delay sequence ("random-but-consistent" canonical eval, PROBE_PLAN §4). When 0
    # (default), the draw uses the global RNG as before (training / ordinary eval).
    delay_seed: int = 0
    random_init_pos: list[float] = [0.0, 0.0, 0.0]  # in m
    random_init_pos_scale: float = 0.0
    random_init_vel: list[float] = [0.0, 0.0, 0.0]  # in m/s
    random_init_vel_scale: float = 0.0
    random_init_att: list[float] = [0.0, 0.0, 0.0]  # in deg
    random_init_att_scale: float = 0.0
    random_init_ome: list[float] = [0.0, 0.0, 0.0]  # in deg/s
    random_init_ome_scale: float = 0.0


@configclass
class QuadrotorSimulatorParams:
    """Parameters for the quadrotor simulator."""

    quad_params: QuadrotorParams = MISSING

    delay: float = 0.0  # Simulation delay [s]
    ctrl_dt: float = 0.02
    controller: str = MISSING
    controller_params: Union[BetaFlightParams, LowLevelControllerSimpleParams] = MISSING

    integrator: str = MISSING
    integrator_dt: float = MISSING  # Integrator time step [s]

    imu_enable: bool = False
    imu_params: object = MISSING

    model_pipeline: OrderedDict = (
        MISSING  # List of model names to include in the simulation
    )
