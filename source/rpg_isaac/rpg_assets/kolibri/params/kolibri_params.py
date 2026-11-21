from __future__ import annotations

from rpg_simulator.quadrotor_params import QuadrotorParams
from rpg_simulator.model_battery import BatteryParams
from rpg_simulator.model_motor import MotorParams
from rpg_simulator.model_rotor_simple_augmentation import RotorSimpleAugParams
from rpg_simulator.low_level_controller_betaflight import BetaFlightParams
from rpg_simulator.low_level_controller_simple import LowLevelControllerSimpleParams

KOLIBRI_PARAMS = QuadrotorParams(
    mass=0.220,  # [kg]
    # Motor positions relative to center of mass [m]
    tbm_fr=[0.041, -0.041, 0.0],  # Front right motor position
    tbm_bl=[-0.041, 0.041, 0.0],  # Back left motor position
    tbm_br=[-0.041, -0.041, 0.0],  # Back right motor position
    tbm_fl=[0.041, 0.041, 0.0],  # Front left motor position
    inertia=[0.000142, 0.000165, 0.000213],  # [kgm^2]
    # Motor specifications
    motor_omega_min=150.0,  # [rad/s]
    motor_omega_max=4400.0,  # [rad/s]
    motor_tau=0.033,  # Motor time constant [s]  # FIXED: Was 0.033, should be 0.0245
    omega_max=[10.0, 10.0, 4.0],  # Max angular velocity [rad/s]
    # Thrust and torque
    thrust_map=[2.05e-7, 0.0, 0.0],  # Thrust coefficients [N/(rad/s)^2, N/(rad/s), N]
    kappa=0.00804,  # Torque-to-thrust ratio [Nm/N]
    thrust_min=0.0,  # [N]
    thrust_max=3.5,  # [N] per motor
    derated_thrust_max=3.5,  # [N] per motor, racing derate of the 3.97 N the motors reach
    rotors_config="cross",  # "cross" or "plus"
    battery_type="4s",
    # Aerodynamics
    aero_coeff_1=[0.0, 0.0, 0.0],  # [N/(v/m)]
    aero_coeff_3=[0.0, 0.0, 0.0],  # [N/(v/m)]
    aero_coeff_h=0.0,  # [N/(v^2/m^2)]
    # Scaling factors
    scale_thrust=1.0,
    scale_torque=1.0,
    scale_drag=1.0,
)

KOLIBRI_MOTOR_PARAMS = MotorParams(
    time_constant=0.0245,
    inertia=2.653760e-07,
    pwr_coeff_drag=1.648230e-09,
    pwr_coeff_linear=2.797876e-03,
    pwr_coeff_cubic=1.648230e-09,
    pwr_coeff_sixth=3.183752e-21,
)

KOLIBRI_BATTERY_PARAMS = BatteryParams(
    cell_series=4,
    cell_parallel=1,
    cell_capacity=0.85,
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

KOLIBRI_SIMPLE_AUG_ROTOR_PARAMS = RotorSimpleAugParams(
    scale_force=1.0,
    scale_torque=1.0,
    fx_offset=-0.614044334629854527,
    fx_vx=-0.308613450930366739,
    fx_vx_sqr=0.003046437356012082,
    fx_vx_m_mean=-0.000000042079237977,
    fy_offset=-0.300318342783466030,
    fy_vy=-0.205691979607555364,
    fy_vy_sqr=-0.004763007785011045,
    fy_vy_m_mean=-0.000000042456661569,
    fz_offset=0.0,  # -1.066860221644093665,
    fz_vhor=0.109917526105503788,
    fz_vz=-0.429577541454658818,
    fz_m_mean=-0.000000135271290541,
    fz_vhor_m_mean=0.000000043631632424,
    fz_vhor_sqr=0.012471162539952002,
    fz_vhor_vz_m_mean=-0.000000005720545880,
    fz_vz_cub=-0.000280348156492543,
    tx_offset=-9.513463731361623488,
    tx_vy=14.640493029433025995,
    tx_vy_m_mean=0.000002252553268484,
    ty_offset=64.242146390171697590,
    ty_vx=-21.461364703951598898,
    ty_vx_m_mean=-0.000001867999360543,
    tz_offset=-0.541617903890835639,
    tz_vx=-0.038674299391118654,
    tz_vy=0.138291212515108147,
)

KOLIBRI_LLC_BETAFLIGHT_PARAMS = BetaFlightParams(
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
    max_throttle_pwm=2000,
    dshot_offset=0.055,
    gyro_lowpass_freq=400.0,
    dterm_lowpass_freq=400.0,
    omega_cmd_sqrt=38.732,
    omega_cmd_lin=1.0789,
    omega_volt=99.606,
    omega_offset=-1687.8,
    thrust_map="kolibri/config/kolibri_thrust_map.csv",
)

KOLIBRI_LLC_SIMPLE_PARAMS = LowLevelControllerSimpleParams(
    p_roll=20.0, p_pitch=20.0, p_yaw=41.0
)
