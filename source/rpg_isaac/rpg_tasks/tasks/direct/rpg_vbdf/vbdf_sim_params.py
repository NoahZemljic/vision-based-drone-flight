"""The course's kolibri, on the current (re-identified) rpg_assets model: mass 0.220 kg
and the course thrust derate; LLC/motor/battery/rotor-augmentation are rpg_assets defaults.
(The course's own agilicious pin used a pre-re-ID Betaflight throttle map — dropped 2026-08-19,
so the original flightmare checkpoints no longer reproduce bit-comparably here.)"""

from __future__ import annotations

from rpg_assets.kolibri.kolibri_sim_params import KOLIBRI_SIM_PARAMS
from rpg_assets.kolibri.params.kolibri_params import KOLIBRI_PARAMS

VBDF_KOLIBRI_PARAMS = KOLIBRI_PARAMS.replace(
    mass=0.220,
    derated_thrust_max=2.5,  # train_tracking.py --thrust_max default; overrides config_tracking.yaml
)

VBDF_KOLIBRI_SIM_PARAMS = KOLIBRI_SIM_PARAMS.replace(
    quad_params=VBDF_KOLIBRI_PARAMS,
    delay=0.02,  # kolibri_simulation.yaml
    integrator="IntegratorRK4",
)
