"""Base and hover configs of the VBDF course environments (flightmare
``lb/drone_course_ex``), see README.md."""

from __future__ import annotations

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_newton.physics import NewtonCfg

from rpg_simulator.simulator_params import QuadRandomizationCfg
from rpg_tasks.tasks.direct.rpg_base.base_env_cfg import RPGBackendCfg, RPGBaseEnvCfg, RPGPhysicsCfg
from rpg_tasks.tasks.direct.rpg_vbdf.vbdf_sim_params import VBDF_KOLIBRI_SIM_PARAMS


@configclass
class RPGVbdfPhysicsCfg(RPGPhysicsCfg):
    """Kit-less Newton scene host by default (no cameras in these tasks); ``physics=agisim_physx`` still selects the RTX host."""

    default: NewtonCfg = RPGPhysicsCfg().agisim_newton_mjwarp


@configclass
class RPGVbdfBaseEnvCfg(RPGBaseEnvCfg):
    backend: RPGBackendCfg = RPGBackendCfg(default="agisim")
    rpg_sim_params = VBDF_KOLIBRI_SIM_PARAMS
    delay: float = 0.030  # fixed delay; + U(0,20ms) random_delay => TOTAL uniform [30,50] ms
    action_space: int = 4
    state_space: int = 0
    max_t: float = 30.0
    episode_length_s: float = 30.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=0.0, replicate_physics=True)
    # every env randomizes (flightmare keeps a separate test env instance)
    eval_env_idx: int = -1
    test_env: bool = False
    # Seed for the command-latency draw (consistent-random eval). 0 = disabled (global
    # RNG, default training behaviour). Set > 0 (e.g. via eval_seeded.py) to pin the
    # [0, random_delay) sequence so every compared run sees the SAME delay draw.
    delay_seed: int = 0
    # environment.world_bounds [xmin, ymin, zmin, xmax, ymax, zmax]
    # Canonical TRACKING room (coordinator 2026-09-15): 6m x 6m x 2.25m. Sized so a 1.5m
    # standoff circle around the actor ALWAYS fits fully in-bounds — the actor spline is
    # inset by >= standoff (see vbdf_tracking_base/_reset_idx), so no wall-forced crash is
    # geometrically possible. Matches the render stage box (vbdf_render_env.STAGE_BOX).
    # NOTE: this is the TRACKING room only — HOVER overrides its own smaller box below,
    # because hover is a fixed-target task (target at origin) with no standoff circle.
    world_bounds: list[float] = [0.0, -3.0, 0.25, 6.0, 3.0, 2.5]
    derated_thrust_max: float = 2.5  # per motor [N]; train_tracking.py --thrust_max
    collision_coeff: float = -8.0
    # randomization.rand_init_* (full ranges; applyRandomState scales them per call)
    rand_init_pos: list[float] = [0.8, 0.8, 0.6]
    rand_init_vel: list[float] = [0.8, 0.8, 0.8]
    rand_init_att_deg: list[float] = [20.0, 20.0, 20.0]
    rand_init_ome_deg: list[float] = [45.0, 45.0, 45.0]
    log_interval_iters: int = 10
    # Action low-pass filter blend (0 = disabled / original behaviour). When (0, 1), the
    # issued command is â_t = α·a_t + (1−α)·â_{t−1} (a_t = mapped policy command, â = per-env
    # persistent buffer). Structural smoothing: calms abrupt pitch/yaw/thrust without touching
    # the reward or advantage. The policy adapts to the filtered plant. Buffer reset on reset.
    act_lpf_alpha: float = 0.0
    use_applied_cmd_obs: bool = True  # True = obs carries the AGED applied command (current_cmd);
                                      # False = the ISSUED prev_act. Story knob for the "dead-time
                                      # observability" turn (S2/V1); baselines set False.

    def __post_init__(self):
        super().__post_init__()
        self.scene.env_spacing = 0.0
        self.sim.physics = RPGVbdfPhysicsCfg()
        # push the consistent-random delay seed into the randomization cfg so
        # QuadrotorSimulator draws the identical latency sequence every episode.
        if self.delay_seed > 0:
            self.quad_randomization_cfg.delay_seed = self.delay_seed


@configclass
class RPGVbdfHoverEnvCfg(RPGVbdfBaseEnvCfg):
    """Student skeleton (lb/drone_course_ex): 15-d state obs, crude hover reward."""

    # Hover is a FIXED-TARGET task (target at origin), no actor/standoff — so the 6x6
    # tracking room is wrong for it (it would add +50% regression distance and move the
    # crash wall). Override to the original 4x6 box hover was trained+shipped on.
    world_bounds: list[float] = [0.0, -3.0, 0.25, 4.0, 3.0, 2.5]

    def __post_init__(self):
        super().__post_init__()
        self.viewer.eye = (2.2, -2.2, 2.2)  # viewer / --video camera looks at the hover point
        self.viewer.lookat = (0.0, 0.0, 1.2)

    ctrl_dt: float = 0.01  # sim_dt
    integrator_dt: float = 0.001
    observation_space: int = 15
    tracking_coeff: float = 0.1
    target_point: list[float] = [0.0, 0.0, 1.0]
    rate_penalty_coeff: float = 0.0005  # hover smoothness — halved from 0.001 (sum aggregation made it ~3x; 0.001 over-damped; 0.0005 keeps roll valve closed while allowing position correction)
    thrust_penalty_coeff: float = 0.0005  # thrust SMOOTHNESS (per-step delta); 0.001 froze thrust to 0 -> shifted jitter to roll; 0.0005 aims ~1-3 std
    quad_randomization_cfg: QuadRandomizationCfg = QuadRandomizationCfg(
        random_mass=0.05,
        random_thrust_map=0.10,
        random_delay=0.02,
    )  # light DR (matches course config_tracking.yaml); gives the throttle a slow-moving
    # equilibrium to track, so it feathers instead of freezing (fixes the hover-i05/i06 roll kick)


@configclass
class RPGVbdfTrackingEnvCfg(RPGVbdfBaseEnvCfg):
    """Starting point for the actor-tracking task (Task 2), see the exercise sheet."""

    ctrl_dt: float = 0.01  # sim_dt
    integrator_dt: float = 0.001
    observation_space: int = 15
    tracking_coeff: float = 0.1
    target_point: list[float] = [0.0, 0.0, 1.0]
    # rgb_camera: rotational angle (roll, pitch, yaw) of the camera in degree
    camera_r_BC: list[float] = [0.0, 0.0, 0.0]
    camera_config: str = "offboard_small_ds"
    # real-world latency: fixed delay 0.030 (inherited) + random_delay 0.02 => TOTAL [30,50]ms
    quad_randomization_cfg: QuadRandomizationCfg = QuadRandomizationCfg(random_delay=0.02)


