"""Configs for the two fresh Task-2 tracking environments.

`RPGVbdfStateTrackingEnvCfg`  -> state-based (position) tracker  `Rpg-Vbdf-SB-Tracking-v0`
`RPGVbdfBboxTrackingEnvCfg`   -> vision (virtual bbox) tracker   `Rpg-Vbdf-BB-Tracking-v0`

Both extend the handout's tracking cfg and add the reward/task knobs they need,
keeping single inheritance so ``__post_init__`` chains cleanly to the base config
(which computes ``decimation``).
"""

from __future__ import annotations

import math

from isaaclab.utils.configclass import configclass

from rpg_tasks.tasks.direct.rpg_vbdf.vbdf_env_cfg import RPGVbdfTrackingEnvCfg


@configclass
class RPGVbdfStateTrackingEnvCfg(RPGVbdfTrackingEnvCfg):
    """State-based tracker: observation adds actor rel pos/vel in the follower's BODY
    frame (the deployable "position" signal); reward shaped for standoff tracking."""

    # --- task geometry ---
    desired_dist: float = 1.5          # [m] standoff to hold
    # --- command-history depth (K) ---
    cmd_hist_len: int = 8              # K = length of the shared command-history obs block
                                       # (0 disables it; obs dim = 25 + 4*K for state). K=8
                                       # is the shipped value (obs 57).
    actor_vel_hist: int = 3            # final-1 (§16): past steps of ACTOR relative-velocity in obs,
                                       # giving "where the actor is going" so the policy plans a smooth
                                       # maneuver instead of reacting every 20ms. Adds 3*dim to obs.
    # --- observation representation (S1/S2 ablation) ---
    use_body_frame: bool = True        # True = rel pos/vel rotated into ego body frame (S2); False = world-frame (S1)
    # --- view/angle smoothing (STATE only; default 0 = no behaviour change) ---
    view_smooth_coeff: float = 0.0      # penalty |d(view_angle)/dt| — keeps actor centred w/o yaw-jitter. 0.003 suggested for the S-smooth stage.
    # --- reward weights (per-step, additive) ---
    standoff_coeff: float = 0.1        # Gaussian at desired_dist
    standoff_half: float = 0.5         # [m] where the Gaussian drops to half
    far_coeff: float = 0.02            # weak linear term, zero at 4 m
    far_legacy: bool = False           # True = OLD far-term (+coeff*clamp(4-dist,0), crash-0
                                       # SB-i01 era); False = current penalty (-coeff*clamp(dist-1.5,0)).
                                       # Reward-ablation toggle only.
    bearing_coeff: float = 0.25        # symmetric cos-theta term (STATE; §8). Full +coeff at dead-centre
                                       # (theta=0), 0 at 90deg, NEGATIVE behind -> pulls drone to turn
                                       # and return to centre SMOOTHLY. Was 0.05 with max(0,·) clamp.
    alt_coeff: float = 0.05            # altitude match on a floor-aware target
    alt_half: float = 0.25             # [m] altitude Gaussian width
    # --- smoothness / safety costs (subtracted) ---
    act_rate_coeff: float = 0.006  # final-1 (was 0.01 lever-4): pulled BACK from lever-3 0.03 (which chased distance +58%
                                  # & +10 crashes). Mild roll+pitch magnitude cost only.
    act_delta_coeff: float = 0.0   # final-1 (was 0.03 lever-4): re-enabled across ALL 4 channels (thrust normalized
                                  # by quad_max_force). This is the command-JERK term: taxes the "rapid
                                  # alternation" (yaw->pitch->roll->thrust->hop) while leaving a steady
                                  # maneuver (a_t ~ a_{t-1}) cheap. The real fix-for-good.
    near_coeff: float = 0.45           # quadratic penalty inside near_dist
    near_dist: float = 1.2             # [m] within which the near penalty is active
    # --- floor-aware altitude target ---
    cam_tilt_deg: float = 15.0         # the real camera is tilted up; see PDF Sec 3.1.1
    alt_offset: float = 0.0            # computed below

    def __post_init__(self):
        super().__post_init__()
        self.alt_offset = self.desired_dist * math.sin(math.radians(self.cam_tilt_deg))
        # the camera must actually be tilted for the bbox/alt physics to match reality
        self.camera_r_BC = [0.0, -self.cam_tilt_deg, 0.0]


@configclass
class RPGVbdfBboxTrackingEnvCfg(RPGVbdfTrackingEnvCfg):
    """Vision tracker: observation is only the actor's bounding-box (x, y, w, h)
    history, exactly what YOLO delivers in deployment."""

    bbox_hist_len: int = 8             # number of bbox frames kept in the observation
    bbox_vel_hist: int = 0             # REVERTED: BB-VelHist-01 (bbox_vel_hist=3) was a NEGATIVE result —
                                       # Δ of a noisy projected box amplified churn. 0 = BB-Final-01 obs (85).

    # --- command-history depth (K) ---
    cmd_hist_len: int = 8              # K = length of the shared command-history obs block
                                       # (0 disables it; obs dim = 53 + 4*K for vision). K=8
                                       # is the shipped value (obs 85); K=0 recovers the
                                       # pre-f9ed14f 53-d layout (BB-i08/Bb-i08 and earlier).

    desired_dist: float = 1.5
    # --- reward weights: UNIFIED with the state (SB) tracker (coordinator decision
    # 2026-09-15). All prior vision-specific overrides (standoff_half 0.4, far 0.03,
    # bearing 0.15, act_rate 0.02, act_delta 0.005, near 0.8/1.3) were tuned on the
    # WRONG camera and are therefore invalid. The only legitimate state-vs-vision
    # difference is the OBSERVATION (bbox history vs rel pos/vel), NOT the reward.
    # Vision restarts from the state-shared reward; any future vision-specific tweak
    # must be made + justified AFTER the corrected camera, as a documented decision.
    standoff_coeff: float = 0.1
    standoff_half: float = 0.5
    far_coeff: float = 0.02
    bearing_coeff: float = 0.25   # symmetric cos-theta carry (STATE §8 value); was 0.05 pre-carry
    alt_coeff: float = 0.05
    alt_half: float = 0.25
    act_rate_coeff: float = 0.006  # carry from STATE (SB-PredictHist): gentle roll+pitch-only magnitude
                                   # cost that collapsed twitch without over-damping (was 0.001).
    act_delta_coeff: float = 0.0
    near_coeff: float = 0.45
    near_dist: float = 1.2
    cam_tilt_deg: float = 15.0
    alt_offset: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.alt_offset = self.desired_dist * math.sin(math.radians(self.cam_tilt_deg))
        self.camera_r_BC = [0.0, -self.cam_tilt_deg, 0.0]
