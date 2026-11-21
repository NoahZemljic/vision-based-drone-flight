"""Vision (virtual bounding-box) tracking environment -- Task 2 deliverable #2.

Observation = own state (pos, rot, vel, omega) + last command + an N-frame history of
the actor's (x, y, w, h) bounding box in the follower's double-sphere camera. The bbox
is synthesised by projecting the actor's 26 cm x 8 cm body (course sheet 4.1.4) through
the camera -- exactly the signal a real YOLO detector delivers in deployment, so the
policy sees no privileged metric state.

The REWARD still uses metric state (privileged in simulation only) -- the same split the
Task-2 brief expects: vision input, metric reward during training.

Registered as ``Rpg-Vbdf-BB-Tracking-v0``.
"""

from __future__ import annotations

import torch

from rpg_tasks.tasks.direct.rpg_vbdf.vbdf_tracking_base import TrackingMixin
from rpg_tasks.tasks.direct.rpg_vbdf.vbdf_tracking_cfg import RPGVbdfBboxTrackingEnvCfg


class RPGVbdfBboxTrackingEnv(TrackingMixin):
    # observation layout
    #   0:15   own state (pos 3, rot 6, vel 3, omega 3)
    #   15:19  last command (collective thrust + 3 bodyrates)
    #   19:51  bbox history: N_HIST x (x, y, w, h), newest first
    #   51:63  Δbbox VELOCITY history (BB-VelHist-01): V_HIST x (Δx, Δy, Δw, Δh), newest first
    #   63:65  de-rotated bearing (elevation, azimuth) + range-from-size (the depth cue)
    #          (with V_HIST=0 the block is 0 dim and 63 collapses back to 51 — BB-Final-01 layout)
    OBS_CMD = 15
    OBS_BBOX = 19
    OBS_BBVEL = 19 + 4 * 8  # = 51 (N_HIST=8 fixed for the shared prefix) → start of Δbbox velocity block
    OBS_BEARING = 19 + 4 * 8  # placeholder, recomputed in __init__

    reward_names = ["standoff", "far", "bearing", "alt", "near", "act_smooth", "act_delta", "total_rew"]

    def __init__(self, cfg: RPGVbdfBboxTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        self.N_HIST = cfg.bbox_hist_len
        self.V_HIST = int(getattr(cfg, "bbox_vel_hist", 0))  # 0 => pre-BB-VelHist layout (obs 85)
        self.K = int(getattr(cfg, "cmd_hist_len", 8))
        self.OBS_BBVEL = self.OBS_BBOX + 4 * self.N_HIST          # start of Δbbox velocity block
        self.OBS_BEARING = self.OBS_BBVEL + 4 * self.V_HIST        # after the vel block (bearing 2)
        self.OBS_CMD_HIST = self.OBS_BEARING + 2  # start of shared command-history block
        self.N_OBS = self.OBS_BEARING + 2 + 4 * self.K  # V_HIST=0: 51+2+32=85; V_HIST=3: 63+2+32=97
        cfg.observation_space = self.N_OBS
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._init_tracking(cfg)
        self.obs_buf_policy = torch.zeros(self.num_envs, self.N_OBS, device=self.device, dtype=self.dtype)
        self.bb_hist = torch.zeros(self.num_envs, self.N_HIST, 4, device=self.device, dtype=self.dtype)
        # Δbbox velocity ring buffer (V_HIST steps of (Δx,Δy,Δw,Δh)); zeros-warmup.
        self.bb_vel_hist = torch.zeros(self.num_envs, self.V_HIST, 4, device=self.device, dtype=self.dtype)

    def _get_observations(self):
        qs = self.quad_state
        o = self.obs_buf_policy
        o[:, 0:3] = qs.pos
        o[:, 3:9] = qs.R[:, :, 0:2].permute(0, 2, 1).reshape(self.num_envs, 6)
        o[:, 9:12] = qs.vel
        o[:, 12:15] = qs.ome
        o[:, self.OBS_CMD:self.OBS_BBOX] = (
            self.quadsim.cmd_queue.current_cmd
            if getattr(self.cfg, "use_applied_cmd_obs", True)
            else self.prev_act
        )  # applied (aged) command; use_applied_cmd_obs=False -> issued prev_act

        bbox = self._compute_bbox()  # (N,4) normalized (x,y,w,h), newest first; also caches bearing/range
        # shift history: newest at index 0
        self.bb_hist[:, 1:] = self.bb_hist[:, :-1].clone()
        self.bb_hist[:, 0] = bbox
        o[:, self.OBS_BBOX:self.OBS_BBVEL] = self.bb_hist.reshape(self.num_envs, -1)
        # Δbbox velocity block: per-step Δ = current frame minus the previous-newest frame.
        # Vision-native "where the box is going" (detector-only signal, NO privileged rel-vel).
        if self.V_HIST > 0:
            # shift: newest at index 0 (oldest entries remain zero during warmup)
            self.bb_vel_hist[:, 1:] = self.bb_vel_hist[:, :-1].clone()
            # bb_hist[:,1] is the previous-newest absolute box (pre this-frame) → current Δ
            self.bb_vel_hist[:, 0] = bbox - self.bb_hist[:, 1]
            o[:, self.OBS_BBVEL:self.OBS_BEARING] = self.bb_vel_hist.reshape(self.num_envs, -1)
        o[:, self.OBS_BEARING:self.OBS_CMD_HIST] = self._last_bearing_range  # elevation, azimuth (2)
        o[:, self.OBS_CMD_HIST:self.N_OBS] = self.cmd_hist.reshape(self.num_envs, -1)
        self.last_bbox[:] = bbox
        return {"policy": o}
