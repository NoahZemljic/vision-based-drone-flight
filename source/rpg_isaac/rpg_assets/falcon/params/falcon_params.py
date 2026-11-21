from __future__ import annotations

from rpg_simulator.low_level_controller_betaflight import BetaFlightParams
from rpg_simulator.low_level_controller_simple import LowLevelControllerSimpleParams
from rpg_simulator.model_battery import BatteryParams
from rpg_simulator.model_motor import MotorParams
from rpg_simulator.model_rotor_simple_augmentation import RotorSimpleAugParams
from rpg_simulator.quadrotor_params import QuadrotorParams

# ``quads/falcon.yaml``
FALCON_PARAMS = QuadrotorParams(
    mass=0.600,  # [kg]
    # Motor positions relative to center of mass [m]
    tbm_fr=[0.072, -0.086, 0.0],  # Front right motor position
    tbm_bl=[-0.072, 0.086, 0.0],  # Back left motor position
    tbm_br=[-0.072, -0.086, 0.0],  # Back right motor position
    tbm_fl=[0.072, 0.086, 0.0],  # Front left motor position
    inertia=[0.002084, 0.001830, 0.003118],  # [kgm^2]
    # Motor specifications
    motor_omega_min=150.0,  # [rad/s]
    motor_omega_max=3000.0,  # [rad/s]
    motor_tau=0.032,  # Motor time constant [s]
    omega_max=[10.0, 10.0, 6.0],  # Max angular velocity [rad/s]
    # Thrust and torque
    thrust_map=[1.802e-6, 0.0, 0.0],  # Thrust coefficients [N/(rad/s)^2, N/(rad/s), N]
    kappa=0.012,  # Torque-to-thrust ratio [Nm/N]
    thrust_min=0.0,  # [N]
    thrust_max=15.0,  # [N] per motor
    rotors_config="cross",  # "cross" or "plus"
    battery_type="6s",
    # Aerodynamics. Kept faithful to quads/falcon.yaml; no model in the
    # rpg_simulator pipeline consumes these today (the rotor-augmentation fit
    # below already carries the identified aerodynamic residual).
    aero_coeff_1=[0.26, 0.28, 0.42],  # [N/(v/m)]
    aero_coeff_3=[0.0, 0.0, 0.0],  # [N/(v/m)]
    aero_coeff_h=0.01,  # [N/(v^2/m^2)]
    # Scaling factors
    scale_thrust=1.0,
    scale_torque=1.0,
    scale_drag=1.0,
)

# ``dynamics_models/falcon_model_motor.yaml``
FALCON_MOTOR_PARAMS = MotorParams(
    time_constant=0.032,
    inertia=2.653760e-07,
    pwr_coeff_drag=2.112e-08,
    pwr_coeff_linear=6.088e-03,
    pwr_coeff_cubic=2.200e-08,
    pwr_coeff_sixth=7.700e-20,
)

# ``dynamics_models/falcon_model_battery_6s.yaml``
FALCON_BATTERY_PARAMS = BatteryParams(
    cell_series=6,
    cell_parallel=1,
    cell_capacity=1.2,
    u_0=4.12,
    time_constant=3.3,
    time_constant_short=0.005,
    power_coefficient=0.001048461870588,
    energy_coefficients_lin=-0.110217859867365,
    energy_coefficients_square=0.010336802983888,
    energy_coefficients_cube=-4.377763980885748e-04,
    r_coeff_power=-7.7608e-05,
    r_coeff_capacity=0.0069498,
    r_min=11e-3,
)

# ``dynamics_models/falcon_model_rotor_simple_augmentation.yaml``.
# Identified for: cl 1.5625e-06, kappa 0.012, mass 0.6, inertia
# (0.002084, 0.001830, 0.003118), offset_x 0.072, offset_y 0.086,
# inertia_motor 2.65376e-07.
FALCON_SIMPLE_AUG_ROTOR_PARAMS = RotorSimpleAugParams(
    scale_force=1.0,
    scale_torque=1.0,
    fx_offset=0.14602002830533453,
    fx_vx=-0.19423049603300707,
    fx_vx_sqr=-0.0042694793988313625,
    fx_vx_m_mean=-5.155559446023972e-08,
    fy_offset=0.0996355742747026,
    fy_vy=-0.23981120204796372,
    fy_vy_sqr=0.0033527634203825026,
    fy_vy_m_mean=-9.051040256169429e-08,
    # Unlike Kolibri (whose yaml comments the term out), the Falcon fit keeps
    # its constant body-z residual active.
    fz_offset=-0.48852463248131783,
    fz_vhor=0.017898195037748303,
    fz_vz=-0.5578937438970561,
    fz_m_mean=-1.75384487600223e-08,
    fz_vhor_m_mean=-1.4268110727199551e-08,
    fz_vhor_sqr=0.012694103875070934,
    fz_vhor_vz_m_mean=-3.139949498326411e-09,
    fz_vz_cub=-0.0011280621887058973,
    tx_offset=3.85939478332828,
    tx_vy=1.3949128607111616,
    tx_vy_m_mean=7.608648926460404e-07,
    ty_offset=4.367360554440793,
    ty_vx=-10.542667684303431,
    ty_vx_m_mean=4.0110444388119504e-07,
    tz_offset=0.8700417418655099,
    tz_vx=0.08384119203117552,
    tz_vy=0.1066207325682483,
)

# ``low_level_controllers/falcon_llc_betaflight.yaml``
FALCON_LLC_BETAFLIGHT_PARAMS = BetaFlightParams(
    kp_roll=40.0,
    ki_roll=80.0,
    kd_roll=20.0,
    kp_pitch=40.0,
    ki_pitch=80.0,
    kd_pitch=20.0,
    kp_yaw=30.0,
    ki_yaw=60.0,
    kd_yaw=0.0,
    min_throttle_pwm=1050,
    max_throttle_pwm=1900,
    dshot_offset=0.055,
    gyro_lowpass_freq=250.0,
    dterm_lowpass_freq=170.0,
    omega_cmd_sqrt=79.79,
    omega_cmd_lin=0.04071,
    omega_volt=61.31,
    omega_offset=-1933.2,
    thrust_map="falcon/config/falcon_thrust_map.csv",
)

# ``low_level_controllers/llc_simple.yaml`` (airframe-independent in Agilicious)
FALCON_LLC_SIMPLE_PARAMS = LowLevelControllerSimpleParams(
    p_roll=20.0, p_pitch=20.0, p_yaw=41.0
)
