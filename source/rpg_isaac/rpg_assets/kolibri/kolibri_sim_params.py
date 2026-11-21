from __future__ import annotations

from collections import OrderedDict

from rpg_simulator.simulator_params import QuadrotorSimulatorParams
from rpg_assets.kolibri.params.kolibri_params import *

KOLIBRI_SIM_PARAMS = QuadrotorSimulatorParams(
    quad_params=KOLIBRI_PARAMS,
    delay=0.0,
    ctrl_dt=0.02,
    controller="LowLevelControllerBetaFlight",
    controller_params=KOLIBRI_LLC_BETAFLIGHT_PARAMS,
    # alternative: controller="LowLevelControllerSimple", controller_params=KOLIBRI_LLC_SIMPLE_PARAMS
    integrator="IntegratorRK4",  # "IntegratorRK4",
    integrator_dt=0.001,
    imu_enable=False,
    imu_params=None,
    model_pipeline=OrderedDict(
        [
            ("ModelInit", None),  # required
            ("ModelMotor", KOLIBRI_MOTOR_PARAMS),  # required
            ("ModelBattery", KOLIBRI_BATTERY_PARAMS),
            ("ModelThrustTorqueSimple", None),  # required
            ("ModelRotorSimpleAugmentation", KOLIBRI_SIMPLE_AUG_ROTOR_PARAMS),
            ("ModelRigidBody", None),  # required for agisim env
            ("ModelScale", None),
        ]
    ),
    # model_pipeline=OrderedDict([
    #     ("ModelInit", None),
    #     ("ModelMotor", KOLIBRI_MOTOR_PARAMS),
    #     ("ModelBattery", KOLIBRI_BATTERY_PARAMS),
    #     ("ModelThrustTorqueSimple", None),
    #     ("ModelRotorSimpleAugmentation", KOLIBRI_SIMPLE_AUG_ROTOR_PARAMS),
    # ])
)
