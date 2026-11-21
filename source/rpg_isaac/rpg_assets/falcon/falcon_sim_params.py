"""Falcon simulator configuration.

Mirrors ``agilib/params/falcon_simulation.yaml``: RK4 at 1 ms, a Betaflight
low-level controller at 50 Hz and a 20 ms command delay.

``ModelBattery`` must run after ``ModelMotor``, which writes the electrical
power into the energy derivative it consumes. It only discharges the pack; the
24.72 V reset voltage comes from ``battery_type`` in :data:`FALCON_PARAMS` and
sits inside the 18-25 V range the Falcon thrust map is tabulated over.
"""

from __future__ import annotations

from collections import OrderedDict

from rpg_assets.falcon.params.falcon_params import *  # noqa: F403
from rpg_simulator.simulator_params import QuadrotorSimulatorParams

FALCON_SIM_PARAMS = QuadrotorSimulatorParams(
    quad_params=FALCON_PARAMS,  # noqa: F405
    delay=0.0,
    ctrl_dt=0.02,
    controller="LowLevelControllerBetaFlight",
    controller_params=FALCON_LLC_BETAFLIGHT_PARAMS,  # noqa: F405
    integrator="IntegratorRK4",
    integrator_dt=0.001,
    imu_enable=False,
    imu_params=None,
    model_pipeline=OrderedDict(
        [
            ("ModelInit", None),  # required
            ("ModelMotor", FALCON_MOTOR_PARAMS),  # noqa: F405 - required
            ("ModelBattery", FALCON_BATTERY_PARAMS),  # noqa: F405 - must follow ModelMotor
            ("ModelThrustTorqueSimple", None),  # required
            ("ModelRotorSimpleAugmentation", FALCON_SIMPLE_AUG_ROTOR_PARAMS),  # noqa: F405
            ("ModelRigidBody", None),  # required for agisim env
            ("ModelScale", None),
        ]
    ),
)
