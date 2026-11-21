"""Actor-tracking task, student starting point (Task 2 of the exercise sheet).

The actor drone flies a random cubic-spline reference (``utils/random_trajectory.py``,
provided) and its state is kept in ``self.other_pos / other_vel / other_att / other_R``.
As handed out, the environment trains the same crude hover-at-(0,0,1) policy as
``Rpg-Vbdf-Hover-v0``."""

from __future__ import annotations

import torch

from rpg_tasks.tasks.direct.rpg_vbdf.utils.random_trajectory import BatchedRandomTrajectory
from rpg_tasks.tasks.direct.rpg_vbdf.vbdf_base_env import RPGVbdfBaseEnv
from rpg_tasks.tasks.direct.rpg_vbdf.vbdf_env_cfg import RPGVbdfTrackingEnvCfg


class RPGVbdfTrackingEnv(RPGVbdfBaseEnv):
    # observation layout
    # 3 position
    # 6 rotation matrix
    # 3 velocity
    # 3 omega
    OBS_POS, OBS_ROT, OBS_VEL, OBS_OME, N_OBS = 0, 3, 9, 12, 15
    reward_names = ["tracking", "total_rew"]

    def __init__(self, cfg: RPGVbdfTrackingEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.observation_space = self.N_OBS
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        dev, dt = self.device, self.dtype
        self.tracking_coeff = cfg.tracking_coeff
        self.target_point = torch.tensor(cfg.target_point, device=dev, dtype=dt)

        # the actor (provided): a random spline reference per environment
        self.reference = BatchedRandomTrajectory(self.num_envs, dev, dt)
        self.other_pos = torch.zeros(self.num_envs, 3, device=dev, dtype=dt)
        self.other_vel = torch.zeros(self.num_envs, 3, device=dev, dtype=dt)
        self.other_att = torch.zeros(self.num_envs, 4, device=dev, dtype=dt)
        self.other_att[:, 0] = 1.0
        self.other_R = torch.eye(3, device=dev, dtype=dt).repeat(self.num_envs, 1, 1)
        self.ref_time = torch.zeros(self.num_envs, device=dev, dtype=dt)
        self.vel_max = 1.0

        self.obs_buf_policy = torch.zeros(self.num_envs, self.N_OBS, device=dev, dtype=dt)
        # read by run/vbdf/eval_vbdf_policy.py
        self.last_bbox = torch.zeros(self.num_envs, 4, device=dev, dtype=dt)
        self.dist_to_actor = torch.zeros(self.num_envs, device=dev, dtype=dt)
        self.view_angle = torch.zeros(self.num_envs, device=dev, dtype=dt)

    # ------------------------------------------------------------------ actor
    def _advance_actor(self) -> None:
        self.ref_time += self.ctrl_dt
        pos, vel, _acc, q, R = self.reference.setpoint_R(self.ref_time)
        self.other_pos[:] = pos
        self.other_vel[:] = vel
        self.other_att[:] = q
        self.other_R[:] = R

    def _step_dynamics(self, action, timings=None, *, camera_only=False):
        super()._step_dynamics(action, timings, camera_only=camera_only)
        self._advance_actor()
        # R_CW, t = self.camera.pose(self.quad_state.R, self.quad_state.pos)

    # ------------------------------------------------------------------ obs / reward
    def _get_observations(self):
        qs = self.quad_state
        o = self.obs_buf_policy
        o[:, 0:3] = qs.pos
        o[:, 3:9] = qs.R[:, :, 0:2].permute(0, 2, 1).reshape(self.num_envs, 6)
        o[:, 9:12] = qs.vel
        o[:, 12:15] = qs.ome
        return {"policy": o}

    def _get_rewards(self):
        dist = torch.linalg.norm(self.quad_state.pos - self.target_point, dim=1)
        tracking = torch.clamp(self.tracking_coeff * (1.0 - 0.1 * dist), min=0.0)
        comps = torch.stack([tracking, tracking], dim=1)
        self.dist_to_actor[:] = torch.linalg.norm(self.other_pos - self.quad_state.pos, dim=1)
        self._accum_metric("dist_to_actor", self.dist_to_actor)
        return self._finish_reward(comps)

    def _update_curriculum(self) -> None:
        pass

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._log_finished_episodes(env_ids)
        super()._reset_idx(env_ids)  # zero state, cmd queue, DR + random delay
        c = self.cfg
        lo, hi = self.world_lo, self.world_hi
        span = hi - lo

        # the actor: a random spline through the (inset) room. This is the handout's
        # crude hover-at-target baseline (no standoff), so the plain 10% inset is fine here;
        # the standoff-aware inset lives in TrackingMixin (vbdf_tracking_base.py).
        slo = (lo + 0.1 * span).expand(env_ids.numel(), 3)
        shi = (hi - 0.1 * span).expand(env_ids.numel(), 3)
        self.reference.sample_random(env_ids, slo, shi, 10.0 * c.max_t, self.vel_max)
        # the actor's clock starts at the (possibly randomized) transport delay
        self.ref_time[env_ids] = self.quadsim.random_delay[env_ids].to(self.dtype)
        pos, vel, _acc, q, R = self.reference.setpoint_R(torch.zeros_like(self.ref_time))
        self.other_pos[env_ids] = pos[env_ids]
        self.other_vel[env_ids] = vel[env_ids]
        self.other_att[env_ids] = q[env_ids]
        self.other_R[env_ids] = R[env_ids]

        # Place quad in the middle of the arena
        qs = self.quad_state
        qs.pos[env_ids] = lo + 0.5 * span
        if not self.test_env:
            # perturb starting point but "gently" except position
            self.apply_random_state(env_ids, 1.0, 0.25, 0.2, 0.2)

        self.last_bbox[env_ids] = 0.0
        self._finalize_reset(env_ids)
