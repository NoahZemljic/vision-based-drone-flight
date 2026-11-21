"""Shared pieces of the VBDF course envs (flightmare ``EnvBase`` + the common parts of
``TrackingEnv``): flightmare-style action mapping, world-bounds crash detection,
flightmare ``applyRandomState``, per-episode reward-component bookkeeping."""

from __future__ import annotations

import math
import torch

from rpg_simulator.utils.math_utils import quat_from_angle_axis, quat_from_euler_xyz, quat_mul
from rpg_tasks.tasks.direct.rpg_base.base_env import RPGBaseEnv
from rpg_tasks.tasks.direct.rpg_vbdf.vbdf_env_cfg import RPGVbdfBaseEnvCfg


class RPGVbdfBaseEnv(RPGBaseEnv):
    ACT_DIM = 4
    # flight modes (trackingenv::RacingState)
    EPISODE_DONE = -1
    FLYING = 0
    CRASH_WORLD = 1
    TARGET_LOST = 2

    reward_names: list[str] = ["total_rew"]

    def __init__(self, cfg: RPGVbdfBaseEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.action_space = self.ACT_DIM
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self.cfg.episode_length_s = cfg.max_t
        self.test_env = cfg.test_env
        dev, dt = self.device, self.dtype
        wb = torch.tensor(cfg.world_bounds, device=dev, dtype=dt)
        self.world_lo, self.world_hi = wb[:3], wb[3:]

        # EnvBase::loadQuadrotor action mapping ("old" flightmare normalization)
        quad = self.cfg.rpg_sim_params.quad_params
        self.mass = quad.mass
        per_motor = cfg.derated_thrust_max if cfg.derated_thrust_max is not None else quad.thrust_max
        self.quad_max_force = 4.0 * per_motor
        wmax = quad.omega_max
        half = self.quad_max_force / (2.0 * self.mass)
        self.act_mean = torch.tensor([half, 0.0, 0.0, 0.0], device=dev, dtype=dt)
        self.act_std = torch.tensor([half, wmax[0], wmax[1], wmax[2]], device=dev, dtype=dt)
        self.act_min = torch.tensor([0.0, -wmax[0], -wmax[1], -wmax[2]], device=dev, dtype=dt)
        self.act_max = torch.tensor([2 * half, wmax[0], wmax[1], wmax[2]], device=dev, dtype=dt)
        print(f"[RPGVbdf] kolibri m={self.mass} kg, max collective {self.quad_max_force:.1f} N, ctrl {1/self.ctrl_dt:.1f} Hz")

        self.pi_act = torch.zeros(self.num_envs, 4, device=dev, dtype=dt)
        self.prev_act = torch.zeros(self.num_envs, 4, device=dev, dtype=dt)
        # action low-pass filter buffer (per-env persistent issued command â). When
        # cfg.act_lpf_alpha > 0 the issued command is alpha*new + (1-alpha)*prev_state.
        self.act_lpf_alpha = float(getattr(cfg, "act_lpf_alpha", 0.0))
        self.act_lpf = torch.zeros(self.num_envs, 4, device=dev, dtype=dt)
        # K = cmd_hist_len command history (newest first): (N, K, 4). Same layout/units
        # as the mapped command (collective thrust + 3 bodyrates). Advanced each step from
        # the *issued* mapped command (self.pi_act); zeroed on episode reset.
        self.cmd_hist_len = int(getattr(cfg, "cmd_hist_len", 8))
        self.cmd_hist = torch.zeros(self.num_envs, self.cmd_hist_len, 4, device=dev, dtype=dt)
        self.flight_mode = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self.collision_coeff = cfg.collision_coeff
        self.rand_init_pos = torch.tensor(cfg.rand_init_pos, device=dev, dtype=dt)
        self.rand_init_vel = torch.tensor(cfg.rand_init_vel, device=dev, dtype=dt)
        self.rand_init_att = torch.tensor(cfg.rand_init_att_deg, device=dev, dtype=dt) * math.pi / 180.0
        self.rand_init_ome = torch.tensor(cfg.rand_init_ome_deg, device=dev, dtype=dt) * math.pi / 180.0

        # reward components of the current step and their episode sums (for logging)
        self.rew_dim = len(self.reward_names)
        self.reward_components = torch.zeros(self.num_envs, self.rew_dim, device=dev, dtype=dt)
        self._ep_rew_sum = torch.zeros(self.num_envs, self.rew_dim, device=dev, dtype=dt)
        self._ep_len = torch.zeros(self.num_envs, device=dev, dtype=dt)
        self._log_rew = torch.zeros(self.rew_dim, device=dev, dtype=dt)  # finished-episode sums
        self._log_len = torch.zeros((), device=dev, dtype=dt)
        self._log_eps = 0
        self._log_crash = 0
        self._log_extra: dict[str, float] = {}
        self._metric_sum: dict[str, torch.Tensor] = {}
        self._metric_n = 0
        self._iteration = 0

    # ------------------------------------------------------------------ actions
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.flight_mode[:] = self.FLYING
        # SB3 clips the sampled action to the [-1, 1] box; TrackingEnv::stepSimulation
        # then maps act * std + mean (its cwiseMax/cwiseMin result is discarded, but the
        # clipped box already lands inside [act_min, act_max]).
        self.pi_act[:] = torch.clamp(actions, -1.0, 1.0) * self.act_std + self.act_mean
        # structural action low-pass filter (optional): â = α·a + (1−α)·â_prev, issued as the
        # real command. Calms abrupt pitch/yaw/thrust churn; policy adapts to the filtered
        # plant. When act_lpf_alpha==0 this is a no-op (issue the raw command, byte-identical).
        if self.act_lpf_alpha > 0.0:
            self.act_lpf.mul_(1.0 - self.act_lpf_alpha).add_(self.pi_act, alpha=self.act_lpf_alpha)
            self.pi_act[:] = self.act_lpf
        # record THIS step's issued command, newest first
        if self.cmd_hist_len > 1:
            self.cmd_hist[:, 1:] = self.cmd_hist[:, :-1].clone()
        if self.cmd_hist_len > 0:
            self.cmd_hist[:, 0] = self.pi_act
        super()._pre_physics_step(self.pi_act)

    def step(self, action: torch.Tensor):
        obs, rew, term, tout, extras = super().step(action)
        self.prev_act[:] = self.pi_act
        return obs, rew, term, tout, extras

    # ------------------------------------------------------------------ dones / crash
    def _out_of_bounds(self) -> torch.Tensor:
        p = self.quad_state.pos
        return ((p < self.world_lo) | (p > self.world_hi)).any(dim=1)

    def _get_dones(self):
        self._update_curriculum()
        crash = self._out_of_bounds()
        self.flight_mode[crash] = self.CRASH_WORLD
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.test_env:  # flightmare test env never times out
            time_out = torch.zeros_like(time_out)
        terminated = crash & ~time_out
        self._log_crash += int(crash.sum().item())
        return terminated, time_out

    def _update_curriculum(self) -> None:
        pass

    # ------------------------------------------------------------------ bookkeeping
    def _finish_reward(self, components: torch.Tensor) -> torch.Tensor:
        """Store components (N, rew_dim; last = total), add the terminal collision
        reward (EnvBase::isTerminalState) and return the per-step total."""
        self.reward_components[:] = components
        crash = self.flight_mode == self.CRASH_WORLD
        total = components[:, -1] + crash.to(components.dtype) * self.collision_coeff
        self.reward_components[:, -1] = total
        self._ep_rew_sum += self.reward_components
        self._ep_len += 1.0
        self._metric_n += 1
        return total

    def _log_finished_episodes(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._log_rew += self._ep_rew_sum[env_ids].sum(dim=0)
        self._log_len += self._ep_len[env_ids].sum()
        self._log_eps += int(env_ids.numel())
        self._ep_rew_sum[env_ids] = 0.0
        self._ep_len[env_ids] = 0.0

    def _accum_metric(self, name: str, value: torch.Tensor) -> None:
        """Accumulate a per-step batch-mean metric (no host sync until logging)."""
        v = value.detach().mean()
        if name in self._metric_sum:
            self._metric_sum[name] += v
        else:
            self._metric_sum[name] = v.clone()

    def train_log_dict(self) -> dict[str, float]:
        d = {}
        n = max(self._log_eps, 1)
        for i, name in enumerate(self.reward_names):
            d[f"Reward/{name}"] = float(self._log_rew[i].item()) / n
        d["Episode/length"] = float(self._log_len.item()) / n
        d["Episode/count"] = float(self._log_eps)
        d["Termination/crash_world"] = float(self._log_crash)
        for k, v in self._metric_sum.items():
            d[f"Metrics/{k}"] = float(v.item()) / max(self._metric_n, 1)
        d.update(self._log_extra)
        self._log_rew.zero_()
        self._log_len.zero_()
        self._log_eps = 0
        self._log_crash = 0
        self._metric_sum = {}
        self._metric_n = 0
        return d

    def on_train_log(self, iteration: int, logger) -> None:
        self._iteration = iteration
        if iteration % self.cfg.log_interval_iters != 0:
            return
        d = self.train_log_dict()
        # Preferred path: record into the SB3 logger so the metrics land in the
        # tensorboard event files (and, with wandb's sync_tensorboard, in W&B too).
        record = getattr(logger, "record", None)
        if callable(record):
            for k, v in d.items():
                record(k, v, exclude=("stdout",))  # our own print() below covers stdout
        else:
            # fallback if no SB3 logger is available (e.g. a bare eval loop)
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(d, step=int(getattr(logger, "tot_timesteps", iteration)))
            except ImportError:
                pass
        rews = " ".join(f"{k.split('/')[1]}={v:.3f}" for k, v in d.items() if k.startswith("Reward/"))
        print(
            f"[RPGVbdf] it {iteration}: eps={int(d['Episode/count'])} len={d['Episode/length']:.0f} "
            f"crash={int(d['Termination/crash_world'])} {rews}"
            + "".join(f" {k.split('/')[1]}={v:.3f}" for k, v in d.items() if k.startswith("Metrics/"))
        )

    # ------------------------------------------------------------------ resets
    def apply_random_init_state(self, env_ids=None):
        pass  # placement is done by the task in _reset_idx (test/train aware)

    def apply_random_state(self, env_ids: torch.Tensor, rand_pos: float, rand_vel: float, rand_att: float, rand_ome: float):
        """EnvBase::applyRandomState: perturb pos/att/vel/ome of env_ids in place."""
        n = env_ids.numel()
        u = lambda: 2.0 * torch.rand(n, 3, device=self.device, dtype=self.dtype) - 1.0  # noqa: E731
        qs = self.quad_state
        qs.pos[env_ids] += u() * self.rand_init_pos * rand_pos
        rpy = u() * self.rand_init_att * rand_att
        # AngleAxis(rx,X)*AngleAxis(ry,Y)*AngleAxis(rz,Z) * q
        qx = quat_from_angle_axis(rpy[:, 0], torch.tensor([1.0, 0, 0], device=self.device, dtype=self.dtype).expand(n, 3))
        qy = quat_from_angle_axis(rpy[:, 1], torch.tensor([0, 1.0, 0], device=self.device, dtype=self.dtype).expand(n, 3))
        qz = quat_from_angle_axis(rpy[:, 2], torch.tensor([0, 0, 1.0], device=self.device, dtype=self.dtype).expand(n, 3))
        qs.att[env_ids] = quat_mul(quat_mul(quat_mul(qx, qy), qz), qs.att[env_ids])
        qs.vel[env_ids] += u() * self.rand_init_vel * rand_vel
        qs.ome[env_ids] += u() * self.rand_init_ome * rand_ome

    @staticmethod
    def yaw_of(q: torch.Tensor) -> torch.Tensor:
        """agilib QuadState::getYaw(0): heading of the body x axis."""
        # body x in world = first column of R
        w, x, y, z = q.unbind(-1)
        xb_x = 1 - 2 * (y * y + z * z)
        xb_y = 2 * (x * y + w * z)
        return torch.atan2(xb_y, xb_x)

    @staticmethod
    def apply_yaw(q: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """agilib QuadState::applyYaw: q * AngleAxis(angle, body z)."""
        qz = quat_from_angle_axis(angle, torch.tensor([0.0, 0.0, 1.0], device=q.device, dtype=q.dtype).expand(q.shape[0], 3))
        q = quat_mul(q, qz)
        return torch.where(q[:, :1] < 0, -q, q)

    def _finalize_reset(self, env_ids: torch.Tensor) -> None:
        """Common tail: recompute caches, seed the delayed command with hover, zero buffers."""
        qs = self.quad_state
        qs.att[env_ids] = qs.att[env_ids] / torch.linalg.norm(qs.att[env_ids], dim=-1, keepdim=True)
        self.quadsim.refresh_caches()
        # before the first command clears the transport delay the sim holds a hover command
        self.quadsim.cmd_queue.current_cmd[env_ids, 0] = 9.81
        self.quadsim.cmd_queue.current_cmd[env_ids, 1:] = 0.0
        self.pi_act[env_ids] = 0.0
        self.prev_act[env_ids] = 0.0
        self.act_lpf[env_ids] = 0.0
        self.cmd_hist[env_ids] = 0.0
        if hasattr(self, "relv_hist"):      # state env only (actor-vel history obs, §16)
            self.relv_hist[env_ids] = 0.0
        self.flight_mode[env_ids] = self.FLYING
        if not self.using_agisim:
            self.bridge.push_root_state(env_ids)

    @staticmethod
    def rpy_quat(roll, pitch, yaw):
        return quat_from_euler_xyz(roll, pitch, yaw)
