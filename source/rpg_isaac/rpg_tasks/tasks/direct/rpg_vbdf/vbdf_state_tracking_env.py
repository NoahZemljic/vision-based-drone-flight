"""State-based (position) tracking environment -- Task 2 deliverable #1.

Observation = own state (pos, rot, vel, omega) + last command + the actor's relative
position AND velocity expressed in the follower's body frame. This relative-in-body-
frame form is what a deployment node can actually build from the actor's
``agiros_pilot/state`` (world frame) + the follower's own attitude -- no privileged
absolute pose.

Registered as ``Rpg-Vbdf-SB-Tracking-v0``.
"""

from __future__ import annotations

import torch

from rpg_tasks.tasks.direct.rpg_vbdf.vbdf_tracking_base import TrackingMixin
from rpg_tasks.tasks.direct.rpg_vbdf.vbdf_tracking_cfg import RPGVbdfStateTrackingEnvCfg


class RPGVbdfStateTrackingEnv(TrackingMixin):
    # observation layout
    #   0:3   pos (world)
    #   3:9   rotation (first two columns of R, flatten col-major = 6)
    #   9:12  vel (world)
    #   12:15 omega (body)
    #   15:19 last command (collective thrust + 3 bodyrates)
    #   19:22 actor rel pos in body frame
    #   22:25 actor rel vel in body frame
    OBS_CMD = 15
    OBS_REL_POS = 19
    OBS_REL_VEL = 22
    OBS_CMD_HIST = 25              # start of the shared command-history block (K * 4 dims)

    reward_names = ["standoff", "far", "bearing", "alt", "near", "act_smooth", "act_delta", "total_rew"]

    def __init__(self, cfg: RPGVbdfStateTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        self.K = int(getattr(cfg, "cmd_hist_len", 8))
        self.OBS_CMD_HIST = 25
        self.N_VEL_HIST = int(getattr(cfg, "actor_vel_hist", 3))  # past steps of actor rel-vel
        self.N_OBS = 25 + 4 * self.K + 3 * self.N_VEL_HIST
        cfg.observation_space = self.N_OBS
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._init_tracking(cfg)
        self.obs_buf_policy = torch.zeros(self.num_envs, self.N_OBS, device=self.device, dtype=self.dtype)
        # actor relative-velocity history (newest first): gives the policy "where the actor is
        # GOING" so it can commit to one smooth maneuver instead of reacting every 20ms (the
        # fix-for-good, §16). (N, N_VEL_HIST, 3).
        self.relv_hist = torch.zeros(self.num_envs, self.N_VEL_HIST, 3, device=self.device, dtype=self.dtype)

    def _get_observations(self):
        qs = self.quad_state
        o = self.obs_buf_policy
        o[:, 0:3] = qs.pos
        o[:, 3:9] = qs.R[:, :, 0:2].permute(0, 2, 1).reshape(self.num_envs, 6)
        o[:, 9:12] = qs.vel
        o[:, 12:15] = qs.ome
        # last APPLIED command (the aged, delayed, in-flight command actually acting on the
        # airframe) — NOT the just-issued pi_act. The command queue delays issuance by
        # delay+random_delay (30-50ms = 3-5 ctrl steps), so prev_act (issued) is stale-by-that-
        # much and the true applied command was unobservable -> the collective-thrust dead-time
        # surge could not be compensated (jitter diagnosis §9). current_cmd is (N,4) collective
        # thrust + 3 bodyrates, same units/layout as prev_act.
        # use_applied_cmd_obs=False -> fall back to the ISSUED prev_act (the pre-fix channel),
        # so the story can build a clean "before applied-command observability" baseline.
        o[:, self.OBS_CMD:self.OBS_REL_POS] = (
            self.quadsim.cmd_queue.current_cmd
            if getattr(self.cfg, "use_applied_cmd_obs", True)
            else self.prev_act
        )

        # actor rel state: world-frame (S1) or ego body-frame (S2 = R^T @ v)
        rel_w = self.other_pos - qs.pos
        relv_w = self.other_vel - qs.vel
        if self.cfg.use_body_frame:
            Rt = qs.R.transpose(-2, -1)
            rel_w = torch.bmm(Rt, rel_w.unsqueeze(-1)).squeeze(-1)
            relv_w = torch.bmm(Rt, relv_w.unsqueeze(-1)).squeeze(-1)
        o[:, self.OBS_REL_POS:self.OBS_REL_VEL] = rel_w
        o[:, self.OBS_REL_VEL:self.OBS_CMD_HIST] = relv_w
        o[:, self.OBS_CMD_HIST:self.OBS_CMD_HIST + 4 * self.K] = self.cmd_hist.reshape(self.num_envs, -1)
        # actor rel-vel history (newest first) — the predictive signal for smooth maneuvering.
        # Guarded on N_VEL_HIST>0 so actor_vel_hist=0 is a valid "no predictive obs" baseline
        # (obs drops the 3*N_VEL_HIST block; the obs slice below is empty and writes nothing).
        # Mirrors the bbox env's bbox_vel_hist=0 guard (V_HIST>0).
        if self.N_VEL_HIST > 0:
            self.relv_hist[:, 1:] = self.relv_hist[:, :-1].clone()
            self.relv_hist[:, 0] = relv_w
            o[:, self.OBS_CMD_HIST + 4 * self.K:] = self.relv_hist.reshape(self.num_envs, -1)

        # also refresh the bbox metric buffer so eval CSV rows carry it (informational)
        self.last_bbox[:] = self._compute_bbox()
        return {"policy": o}
