"""Shared base for the two fresh Task-2 tracking environments.

Common behaviours (actor advancement from the provided random spline, the actor's
3D body -> double-sphere bounding box, the shared reward, reset) live here so the
two variants -- state-based and vision-based -- differ ONLY in the observation block
they hand to the policy, which is the cleanest way to isolate "position vs vision".

Everything operates on batched torch tensors (leading dim N = num_envs), matching the
handout's convention.
"""

from __future__ import annotations

import math

import torch

from rpg_tasks.tasks.direct.rpg_vbdf.utils.double_sphere_camera import DoubleSphereCamera, OFFBOARD_SMALL_DS
from rpg_tasks.tasks.direct.rpg_vbdf.utils.random_trajectory import BatchedRandomTrajectory
from rpg_tasks.tasks.direct.rpg_vbdf.vbdf_base_env import RPGVbdfBaseEnv


class TrackingMixin(RPGVbdfBaseEnv):
    """Common tracking plumbing. Not a registered gym env on its own."""

    # actor body geometry (course sheet 4.1.4: 26 cm motor-to-motor diagonal, ~8 cm tall).
    # The diagonal is 26 cm, so the "arm" (half-width along a body axis) follows from the
    # motor pattern; we use a simple box of this extent for the virtual bounding box.
    BODY_HALF_XY = 0.092       # [m] ~ 26 cm / 2 / sqrt(2) per motor arm
    BODY_HALF_H = 0.04         # [m] half the ~8 cm height
    # "no detection" sentinel bbox (x, y, w, h), normalized to [-1, 1], matching what the
    # deployment pilot builds from an empty BoundingBoxes message (0 px -> -1).
    LOST_BBOX = (-1.0, -1.0, 0.0, 0.0)

    def _init_tracking(self, cfg):
        dev, dt = self.device, self.dtype
        self.desired_dist = cfg.desired_dist
        self.target_point = torch.tensor(cfg.target_point, device=dev, dtype=dt)

        # the actor (provided): a random cubic-spline reference per environment
        self.reference = BatchedRandomTrajectory(self.num_envs, dev, dt)
        self.other_pos = torch.zeros(self.num_envs, 3, device=dev, dtype=dt)
        self.other_vel = torch.zeros(self.num_envs, 3, device=dev, dtype=dt)
        self.other_att = torch.zeros(self.num_envs, 4, device=dev, dtype=dt)
        self.other_att[:, 0] = 1.0
        self.other_R = torch.eye(3, device=dev, dtype=dt).repeat(self.num_envs, 1, 1)
        self.ref_time = torch.zeros(self.num_envs, device=dev, dtype=dt)
        self.vel_max = 1.0

        # the analytical double-sphere camera used to synthesise the virtual bbox
        r_BC = tuple(float(v) for v in cfg.camera_r_BC)
        self.camera = DoubleSphereCamera(
            OFFBOARD_SMALL_DS["intrinsics"], OFFBOARD_SMALL_DS["resolution"],
            r_BC_deg=r_BC, device=dev, dtype=torch.float32,
        )

        # eval/metric buffers (read by run/vbdf/eval_vbdf_policy.py)
        self.last_bbox = torch.zeros(self.num_envs, 4, device=dev, dtype=dt)
        self.dist_to_actor = torch.zeros(self.num_envs, device=dev, dtype=dt)
        self.view_angle = torch.zeros(self.num_envs, device=dev, dtype=dt)
        # previous bearing/view angle, for the |d(view_angle)/dt| smoothing term
        self.prev_view_angle = torch.zeros(self.num_envs, device=dev, dtype=dt)
        # de-rotated bearing (elevation, azimuth) + range-from-size, cached by _compute_bbox
        self._last_bearing_range = torch.zeros(self.num_envs, 2, device=dev, dtype=dt)

        # actor body corners in the ACTOR body frame (for the bounding box): a box of
        # 8 corners = 4 horizontal arms repeated at the top (+h) and bottom (-h).
        hz = self.BODY_HALF_XY
        hv = self.BODY_HALF_H
        corners_xy = torch.tensor(
            [[hz, hz], [-hz, hz], [-hz, -hz], [hz, -hz]], device=dev, dtype=dt
        )  # (4,2)
        top = torch.cat([corners_xy, torch.full((4, 1), hv, device=dev, dtype=dt)], dim=1)
        bot = torch.cat([corners_xy, torch.full((4, 1), -hv, device=dev, dtype=dt)], dim=1)
        self._actor_corners = torch.cat([top, bot], dim=0)  # (8,3)

    # ------------------------------------------------------------------ actor
    def _advance_actor(self) -> None:
        self.ref_time += self.ctrl_dt
        pos, vel, _acc, q, R = self.reference.setpoint_R(self.ref_time)
        # Keep the actor EVALUATED position inside the same 0.1*span inset as its control points
        # (robust to natural-cubic-spline overshoot between knots — Lane 1's diagnosis: without
        # this the actor can drift to the wall and force the tracker out). This is a plain
        # correctness clamp, NOT adversarial machinery — the actor is the course's default
        # 10%-inset random spline, unchanged. x/y clamp only; z keeps its own inset.
        span = self.world_hi - self.world_lo
        m = 0.1 * span  # (3,) — matches the _reset_idx margin
        pos = torch.stack([
            torch.clamp(pos[:, 0], self.world_lo[0] + m[0], self.world_hi[0] - m[0]),
            torch.clamp(pos[:, 1], self.world_lo[1] + m[1], self.world_hi[1] - m[1]),
            pos[:, 2],
        ], dim=-1)
        self.other_pos[:] = pos
        self.other_vel[:] = vel
        self.other_att[:] = q
        self.other_R[:] = R

    def _step_dynamics(self, action, timings=None, *, camera_only=False):
        super()._step_dynamics(action, timings, camera_only=camera_only)
        self._advance_actor()

    # ------------------------------------------------------------------ bbox
    def _compute_bbox(self) -> torch.Tensor:
        """Project the actor's 8 body corners through the follower's double-sphere
        camera -> normalized (x, y, w, h) per env, using the LOST sentinel when no
        corner is in frame. x, y = min corner (course convention), w, h = extent."""
        qs = self.quad_state
        R_CW, t = self.camera.pose(qs.R, qs.pos)

        # actor corners in world: other_pos + other_R @ actor_corner
        corners_b = self._actor_corners  # (8,3)
        corners_w = self.other_pos.unsqueeze(1) + torch.einsum(
            "nij,kj->nki", self.other_R, corners_b
        )  # (N,8,3)

        uvv = self.camera.project(corners_w, R_CW, t, homogeneous=True)  # (N,8,3)
        u, v, valid = uvv[..., 0], uvv[..., 1], uvv[..., 2]  # each (N,8)

        any_valid = valid.any(dim=1)  # (N,)
        u_min = u.min(dim=1).values
        u_max = u.max(dim=1).values
        v_min = v.min(dim=1).values
        v_max = v.max(dim=1).values

        x = u_min
        y = v_min
        w = u_max - u_min
        h = v_max - v_min

        bbox = torch.stack([x, y, w, h], dim=1)  # (N,4) float64 from the camera
        bbox = bbox.float()
        lost = torch.tensor(self.LOST_BBOX, device=self.device, dtype=torch.float32).expand_as(bbox)
        bbox = torch.where(any_valid.unsqueeze(1), bbox, lost)
        # also cache the de-rotated bearing + range-from-size for the BB observation.
        # Bearing = elevation/azimuth of the actor in a gravity-aligned frame (a real
        # node recovers this from the bbox centre via the camera model). Range = the
        # known 26 cm actor width / its apparent angular size (metric from known size,
        # the standard monocular depth cue).

        # direction to the actor CENTRE in camera frame (unit), from the projected centre
        centre_px_u = 0.5 * (u_min + u_max)          # normalized, in [-1,1]
        centre_px_v = 0.5 * (v_min + v_max)
        # de-rotate into gravity frame: use the drone's own attitude to level the bearing.
        # The camera-frame bearing is the direction; we recover elevation (angle above
        # horizon) and azimuth (relative to the drone's heading) directly from the world
        # direction to the actor (equivalent information; direction is what a camera
        # measures, so this is not privileged).
        rel_w = self.other_pos - qs.pos                      # (N,3) world vector
        rel_w_n = rel_w / torch.linalg.norm(rel_w, dim=1, keepdim=True).clamp_min(1e-9)
        elevation = torch.asin(rel_w_n[:, 2])                # angle above horizontal, rad
        az = torch.atan2(rel_w_n[:, 1], rel_w_n[:, 0])       # world-frame azimuth
        heading = torch.atan2(qs.R[:, 1, 0], qs.R[:, 0, 0])  # drone yaw (body x in world)
        azimuth = torch.atan2(torch.sin(az - heading), torch.cos(az - heading))  # wrap to [-pi, pi]

        # Normalise both to [-1, 1] to live alongside the other [-1,1] obs channels
        # (the range-from-size idea was dropped: it was ill-conditioned — near-zero bbox
        # width -> huge/NaN range — and the bbox width w is already in the observation).
        elevation_n = elevation / (math.pi / 2.0)
        azimuth_n = azimuth / math.pi
        bearing = torch.stack([elevation_n, azimuth_n], dim=1).float()
        self._last_bearing_range = bearing
        return bbox

    # ------------------------------------------------------------------ reward
    def _get_rewards(self):
        qs = self.quad_state
        p = qs.pos
        rel = self.other_pos - p                       # world-frame actor relative to drone
        dist = torch.linalg.norm(rel, dim=1)           # (N,)

        # bearing: angle between body x axis and the actor direction
        body_x = qs.R[:, :, 0]                         # first column of R = body x in world
        dir_n = rel / dist.clamp_min(1e-9).unsqueeze(1)
        cos_theta = (body_x * dir_n).sum(dim=1)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

        # --- reward terms ---
        cfg = self.cfg
        # standoff: Gaussian centred at desired_dist
        standoff = cfg.standoff_coeff * torch.exp(-((dist - self.desired_dist) ** 2) / (2 * cfg.standoff_half ** 2))
        # far term: PENALTY that grows with overshoot above the standoff (pull back toward
        # 1.5m), zero at/below desired_dist. Symmetric with the near penalty. The earlier
        # versions got the SIGN wrong: a positive "far" reward (clamp(4-dist) or clamp(dist-1.5))
        # rewarded the drone for being far / fighting the standoff, causing drift-out to 3-4m.
        # far_legacy (ablation) restores the OLD positive form to test whether this flip is
        # the crash driver.
        if getattr(cfg, "far_legacy", False):
            far = cfg.far_coeff * torch.clamp(4.0 - dist, min=0.0)
        else:
            far = -cfg.far_coeff * torch.clamp(dist - self.desired_dist, min=0.0)
        # bearing: symmetric, peaks at dead-centre, smooth falloff, NO max(0,·) clamp.
        # reward = +coeff at theta=0, ->0 at 90deg aside, NEGATIVE when the actor is
        # behind the body x axis (pulls the drone to turn and re-centre). Implemented
        # verbatim from the "veer ok, return smoothly to centre" decision (§8).
        bearing = cfg.bearing_coeff * cos_theta
        # altitude: floor-aware target = actor_z - alt_offset, Gaussian
        alt_target = self.other_pos[:, 2] - cfg.alt_offset
        alt = cfg.alt_coeff * torch.exp(-((p[:, 2] - alt_target) ** 2) / (2 * cfg.alt_half ** 2))
        # near-collision: quadratic penalty inside near_dist
        near = torch.clamp(cfg.near_dist - dist, min=0.0)
        near_pen = -cfg.near_coeff * near ** 2

        # smoothness costs: penalise actual body rate (anti-wobble) and the per-step
        # BODYRATE-ONLY command delta (kill oscillation spikes). The delta is taken on
        # the bodyrate channels [1:4] only — NOT the collective-thrust channel [0], whose
        # 0..45 N range would dominate and over-penalise altitude control (the mistake
        # that collapsed the earlier SB-i02 anti-wobble attempt).
        # ATTITUDE-RATE penalty on ROLL+PITCH only (lever 3, §13). The old isotropic ||omega||
        # was blind to the actual twitch: the bearing reward (body_x·dir_n) is INVARIANT under
        # roll-about-body_x, so the policy was never discouraged from banking/swaying on roll+pitch
        # at ~1 rad/s while "pointing". Penalising sqrt(omex^2+omey^2) gives that flat-gradient
        # attitude DOF a real cost, decoupled from yaw (already calm). act_rate_coeff raised so it
        # is non-negligible vs bearing_coeff=0.25.
        # smoothness: (1) a MILD magnitude cost on roll+pitch attitude (yaw free), and (2) the
        # command-JERK cost on ALL 4 channels — the term that actually kills the "rapid
        # alternation" the boss described (yaw->pitch->roll->thrust->hop) while leaving a STEADY
        # maneuver (a_t ~ a_{t-1}) cheap. act_rate_coeff pulled back (lever-3 0.03 chased
        # distance +10 crashes +58%) and act_delta_coeff re-enabled across all channels; thrust is
        # normalized by quad_max_force so the 0..45N range doesn't dominate.
        ome = qs.ome
        omega = torch.sqrt(ome[:, 0] ** 2 + ome[:, 1] ** 2)   # roll+pitch norm (yaw free)
        act_smooth = -cfg.act_rate_coeff * omega
        body_delta = torch.linalg.norm(self.pi_act[:, 1:4] - self.prev_act[:, 1:4], dim=1)
        thrust_delta = torch.abs(self.pi_act[:, 0] - self.prev_act[:, 0]) / self.quad_max_force
        cmd_delta = body_delta + thrust_delta                  # ALL 4 channels, thrust normalized
        act_delta = -cfg.act_delta_coeff * cmd_delta

        # bookkeeping for logging (reward_names defined in the concrete variant)
        self.dist_to_actor[:] = dist
        self.view_angle[:] = torch.acos(cos_theta)
        # STATE view/angle smoothing: penalise per-step change in pointing angle (the
        # "keep actor centred w/o yaw-jitter" term). getattr so shared TrackingMixin never
        # hard-fails for VISION (no field -> 0.0). Added to `total` only (not reward_names).
        view_delta = torch.abs(self.view_angle - self.prev_view_angle)
        view_smooth = -float(getattr(self.cfg, "view_smooth_coeff", 0.0)) * view_delta
        self.prev_view_angle[:] = self.view_angle

        total = standoff + far + bearing + alt + near_pen + act_smooth + act_delta + view_smooth
        if float(getattr(self.cfg, "view_smooth_coeff", 0.0)) != 0.0:
            self._accum_metric("view_delta", view_delta)

        self._accum_metric("dist_to_actor", dist)

        n = len(self.reward_names) - 1  # last slot = total
        comps = torch.zeros(self.num_envs, n, device=self.device, dtype=self.dtype)
        # component order must match reward_names (excluding the final total)
        names = self.reward_names[:-1]
        named = {
            "standoff": standoff, "far": far, "bearing": bearing, "alt": alt,
            "near": near_pen, "act_smooth": act_smooth, "act_delta": act_delta,
        }
        for i, nm in enumerate(names):
            comps[:, i] = named.get(nm, torch.zeros_like(total))
        comps = torch.cat([comps, total.unsqueeze(1)], dim=1)
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

        # actor: random spline through the (inset) room — the course's DEFAULT behaviour.
        # Plain 10% relative inset on every axis. No adversarial "teacher pushes the tracker"
        # machinery was needed: the course's stock spline already brings the actor near the
        # walls, and near-wall corner tracking is the natural difficulty (boss directive:
        # "adversarial training beyond what the course already had is not necessary"). The
        # earlier actor_wall_margin knob was redundant (0.1*span = 0.6m > 0.5m in the 6x6 room)
        # and deleted as cleanup — i03 already used this 0.6m inset, so no recompute.
        margin = 0.1 * span
        slo = (lo + margin).expand(env_ids.numel(), 3)
        shi = (hi - margin).expand(env_ids.numel(), 3)
        self.reference.sample_random(env_ids, slo, shi, 10.0 * c.max_t, self.vel_max)
        self.ref_time[env_ids] = self.quadsim.random_delay[env_ids].to(self.dtype)
        pos, vel, _acc, q, R = self.reference.setpoint_R(torch.zeros_like(self.ref_time))
        self.other_pos[env_ids] = pos[env_ids]
        self.other_vel[env_ids] = vel[env_ids]
        self.other_att[env_ids] = q[env_ids]
        self.other_R[env_ids] = R[env_ids]

        # place the follower mid-room
        qs = self.quad_state
        qs.pos[env_ids] = lo + 0.5 * span
        if not self.test_env:
            self.apply_random_state(env_ids, 1.0, 0.25, 0.2, 0.2)

        self.last_bbox[env_ids] = 0.0
        self.prev_view_angle[env_ids] = 0.0
        self._finalize_reset(env_ids)
