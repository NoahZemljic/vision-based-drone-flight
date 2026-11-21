"""Port of the student skeleton ``TrackingEnv`` (flightmare ``lb/drone_course_ex``):
15-d state observation, reward ``max(c * (1 - 0.1 * |p - (0,0,1)|), 0)``.
Hands students a policy that reaches the point but oscillates around it."""

from __future__ import annotations

import torch

from rpg_tasks.tasks.direct.rpg_vbdf.vbdf_base_env import RPGVbdfBaseEnv
from rpg_tasks.tasks.direct.rpg_vbdf.vbdf_env_cfg import RPGVbdfHoverEnvCfg


class RPGVbdfHoverEnv(RPGVbdfBaseEnv):
    # trackingenv::RacingState observation layout
    OBS_POS, OBS_ROT, OBS_VEL, OBS_OME = 0, 3, 9, 12
    OBS_CMD_HIST = 15                          # start of shared command-history block
    # N_OBS made dynamic in cmd_hist_len (K) so cmd_hist_len=0 is a valid "no history"
    # baseline (obs = 15 + 4*K; K=8 -> 47). Mirrors the state/vision envs' cmd_hist guard:
    # the cmd_hist fix needs cmd_hist_len=0 to disable history without a stale 32-wide block.
    reward_names = ["tracking", "total_rew"]

    def __init__(self, cfg: RPGVbdfHoverEnvCfg, render_mode: str | None = None, **kwargs):
        self.K = int(getattr(cfg, "cmd_hist_len", 8))
        self.N_OBS = 15 + 4 * self.K
        cfg.observation_space = self.N_OBS
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self.tracking_coeff = cfg.tracking_coeff
        self.rate_penalty_coeff = cfg.rate_penalty_coeff
        self.thrust_penalty_coeff = cfg.thrust_penalty_coeff
        # "make it still" levers, all default 0 -> the terms below vanish and the reward is
        # bit-identical to the shipped H0/H1/H2 one (see vbdf_env_cfg for the why).
        self.ome_rate_coeff = float(getattr(cfg, "ome_rate_coeff", 0.0))
        self.act_delta_coeff = float(getattr(cfg, "act_delta_coeff", 0.0))
        self.velocity_damp_weight = float(getattr(cfg, "velocity_damp_weight", 0.0))
        self.precision_coeff = float(getattr(cfg, "precision_coeff", 0.0))
        self.precision_sigma = float(getattr(cfg, "precision_sigma", 0.10))
        self.rate_near_coeff = float(getattr(cfg, "rate_near_coeff", 0.0))
        self.rate_near_sigma = float(getattr(cfg, "rate_near_sigma", 0.5))
        self.target_point = torch.tensor(cfg.target_point, device=self.device, dtype=self.dtype)
        self.obs_buf_policy = torch.zeros(self.num_envs, self.N_OBS, device=self.device, dtype=self.dtype)

    def _get_observations(self):
        qs = self.quad_state
        o = self.obs_buf_policy
        o[:, 0:3] = qs.pos
        o[:, 3:9] = qs.R[:, :, 0:2].permute(0, 2, 1).reshape(self.num_envs, 6)
        o[:, 9:12] = qs.vel
        o[:, 12:15] = qs.ome
        o[:, self.OBS_CMD_HIST:self.N_OBS] = self.cmd_hist.reshape(self.num_envs, -1)
        return {"policy": o}

    def _get_rewards(self):
        dist = torch.linalg.norm(self.quad_state.pos - self.target_point, dim=1)
        tracking = torch.clamp(self.tracking_coeff * (1.0 - 0.1 * dist), min=0.0)
        # bodyrate smoothness penalty (rad/s; bodyrate channels only). SUM (not mean) so the
        # policy cannot dump all residual correction into a single channel (roll) at 1/3 cost —
        # the mean version let roll become a "free escape valve" (wx std ~3-7 vs wy/wz ~0.5).
        rate_coeff = self.rate_penalty_coeff
        if self.rate_near_coeff > 0.0:
            # proximity-gated smoothness: charge for twitchiness where stillness is the task
            # (at the target) and let the drone fly there cheaply. Same inverse-quadratic
            # gate as the precision bonus, so both terms share one notion of "near".
            sg2 = self.rate_near_sigma**2
            rate_coeff = rate_coeff + self.rate_near_coeff * sg2 / (sg2 + dist**2)
        rate_pen = rate_coeff * torch.abs(self.pi_act[:, 1:4]).sum(dim=1)
        # thrust SMOOTHNESS penalty: per-step delta of collective thrust (NOT magnitude, so it
        # cannot over-dominate the ~0..64 m/s^2 range — differencing removes the hover offset).
        thrust_pen = self.thrust_penalty_coeff * torch.abs(self.pi_act[:, 0] - self.prev_act[:, 0])
        shaped = tracking - rate_pen - thrust_pen
        # --- optional levers (no-ops at their 0 defaults) ---
        if self.precision_coeff > 0.0:
            # Sharp near-target bonus: the stock linear term's 0.01/m slope makes the last 5 cm
            # worth less than the rate penalty, so the policy parks on a static bias.
            #
            # Shape is inverse-quadratic (Lorentzian), NOT Gaussian, and that is load-bearing.
            # A Gaussian bonus works at a short training budget and then collapses at a long
            # one: its gradient at 1 m is ~6e-21/m, so the moment training wanders outside the
            # ~3 sigma basin there is nothing left to pull it back and the policy converges to
            # "perfectly still, 2 m away" (measured: exp(-d^2/2s^2), c=0.3 s=0.1, 20 M steps ->
            # 0.012 m, the SAME run at 50 M -> 2.31 m + 96 crashes). The inverse-quadratic form
            # peaks just as sharply (31x the linear slope at 5 cm) but decays as 1/d^2 instead
            # of exp(-d^2), so it still contributes ~0.004/m out at 2.3 m and the basin is the
            # whole room.
            sigma2 = self.precision_sigma**2
            shaped = shaped + self.precision_coeff * sigma2 / (sigma2 + dist**2)
        if self.ome_rate_coeff > 0.0:
            # penalise what the GYRO sees, not what was commanded — H0's commanded wy std is
            # 3.3 while the plant rings at 6.4, so the commanded-only penalty under-charges it.
            shaped = shaped - self.ome_rate_coeff * torch.linalg.norm(self.quad_state.ome, dim=1)
        if self.act_delta_coeff > 0.0:
            # command jerk, all 4 channels (thrust normalised by the collective range so the
            # 0..45 m/s^2 channel cannot dominate the +-10 rad/s ones)
            body_delta = torch.linalg.norm(self.pi_act[:, 1:4] - self.prev_act[:, 1:4], dim=1)
            thrust_delta = torch.abs(self.pi_act[:, 0] - self.prev_act[:, 0]) / self.quad_max_force
            shaped = shaped - self.act_delta_coeff * (body_delta + thrust_delta)
        if self.velocity_damp_weight > 0.0:
            shaped = shaped - self.velocity_damp_weight * (self.quad_state.vel**2).sum(dim=1)
        rew = torch.clamp(shaped, min=0.0)
        # col order matters: _finish_reward optimizes components[:, -1] (the LAST col = total).
        comps = torch.stack([tracking, rew], dim=1)  # col0 = pure distance (CSV), col1 = shaped total
        self._accum_metric("dist_to_target", dist)
        self._accum_metric("speed", torch.linalg.norm(self.quad_state.vel, dim=1))
        return self._finish_reward(comps)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._log_finished_episodes(env_ids)
        super()._reset_idx(env_ids)  # zero state, cmd queue, (no) DR
        # TrackingEnv::reset: middle of the arena, gently perturbed except position
        qs = self.quad_state
        qs.pos[env_ids] = self.world_lo + 0.5 * (self.world_hi - self.world_lo)
        if not self.test_env:
            self.apply_random_state(env_ids, 1.0, 0.25, 0.2, 0.2)
        self._finalize_reset(env_ids)
