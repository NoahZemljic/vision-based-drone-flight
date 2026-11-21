"""Batched port of agilib ``RandomTrajectory`` / ``HoverReference`` (the actor drone).

agilib samples ``ceil(duration * avg_speed / d)`` uniform points in a box
(``d = |(hi - lo) / 3|``), fits a natural cubic spline per axis over uniformly
spaced knots and returns pos / vel / acc(+g) with the attitude built from
velocity and acceleration (``agilib/src/reference/trajectory_reference/random_trajectory.cpp``,
``agilib/src/utils/cubic_spline.cpp``).
"""

from __future__ import annotations

import math
import torch

from rpg_simulator.utils.math_utils import matrix_from_quat, normalize, quat_from_matrix

G = 9.8066


class BatchedRandomTrajectory:
    """One reference per env: either a random spline or a hover point."""

    def __init__(self, num_envs: int, device, dtype=torch.float32, max_points: int = 64):
        self.n = num_envs
        self.device = device
        self.dtype = dtype
        self.kmax = max_points
        z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)  # noqa: E731
        self.y = z(num_envs, max_points, 3)  # knots
        self.m = z(num_envs, max_points, 3)  # second derivatives at the knots
        self.h = torch.ones(num_envs, device=device, dtype=dtype)  # knot spacing
        self.k = torch.full((num_envs,), 2, device=device, dtype=torch.long)  # knots used
        self.duration = z(num_envs)
        self.hover = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.hover_pos = z(num_envs, 3)

    # ------------------------------------------------------------------ sampling
    def sample_hover(self, env_ids: torch.Tensor, pos: torch.Tensor) -> None:
        self.hover[env_ids] = True
        self.hover_pos[env_ids] = pos

    def sample_random(
        self, env_ids: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, duration: float, avg_speed: float
    ) -> None:
        """Random spline through uniform points in [lo, hi] (per-env bounds (M,3))."""
        n = env_ids.numel()
        if n == 0:
            return
        # agilib: expected distance of two uniform points ~ |(hi-lo)/3| (all envs share
        # the same box size, so one knot count per call)
        dist = torch.linalg.norm((hi[0] - lo[0]) / 3.0).item()
        k = int(math.ceil(duration * avg_speed / dist))
        k = max(2, min(k, self.kmax))
        pts = lo.unsqueeze(1) + torch.rand(n, k, 3, device=self.device, dtype=self.dtype) * (hi - lo).unsqueeze(1)
        h = duration / (k - 1)
        m = self._natural_spline_m(pts, h)
        self.y[env_ids] = 0.0
        self.m[env_ids] = 0.0
        self.y[env_ids, :k] = pts
        self.m[env_ids, :k] = m
        self.h[env_ids] = h
        self.k[env_ids] = k
        self.duration[env_ids] = duration
        self.hover[env_ids] = False

    @staticmethod
    def _natural_spline_m(y: torch.Tensor, h: float) -> torch.Tensor:
        """Second derivatives of the natural cubic spline, uniform spacing h. y: (n,k,3)."""
        n, k, _ = y.shape
        m = torch.zeros_like(y)
        if k < 3:
            return m
        # interior rows i=1..k-2:  h/6 m[i-1] + 2h/3 m[i] + h/6 m[i+1] = (y[i+1]-y[i])/h - (y[i]-y[i-1])/h
        A = torch.zeros(k - 2, k - 2, device=y.device, dtype=y.dtype)
        idx = torch.arange(k - 2, device=y.device)
        A[idx, idx] = 2.0 * h / 3.0
        A[idx[:-1], idx[:-1] + 1] = h / 6.0
        A[idx[1:], idx[1:] - 1] = h / 6.0
        d = (y[:, 2:] - y[:, 1:-1]) / h - (y[:, 1:-1] - y[:, :-2]) / h  # (n,k-2,3)
        m[:, 1:-1] = torch.linalg.solve(A.unsqueeze(0).expand(n, -1, -1), d)
        return m

    # ------------------------------------------------------------------ evaluation
    def setpoint(self, t: torch.Tensor):
        """State of the actor at per-env time t (N,). Returns pos, vel, acc(+g), quat wxyz."""
        t = torch.clamp(t, min=0.0)
        t = torch.minimum(t, self.duration)  # past the end: hold the end setpoint
        # segment index: agilib walks `while x > x_(i+1)` -> i = ceil(t/h)-1
        i = torch.clamp(torch.ceil(t / self.h) - 1, min=0)
        i = torch.minimum(i, (self.k - 2).to(t.dtype)).long()
        h = self.h.unsqueeze(-1)
        x0 = (i.to(t.dtype) * self.h).unsqueeze(-1)
        x1 = x0 + h
        tt = t.unsqueeze(-1)
        ar = torch.arange(self.n, device=self.device)
        y0, y1 = self.y[ar, i], self.y[ar, i + 1]
        m0, m1 = self.m[ar, i], self.m[ar, i + 1]
        p0 = y0 - m0 * h * h / 6.0
        p1 = (y1 - y0) / h - h * (m1 - m0) / 6.0
        pos = (m0 * (x1 - tt) ** 3 + m1 * (tt - x0) ** 3) / (6.0 * h) + p1 * (tt - x0) + p0
        vel = -(m0 * (x1 - tt) ** 2 + m1 * (tt - x0) ** 2) / (2.0 * h) + p1
        acc = (m0 * (x1 - tt) + m1 * (tt - x0)) / h
        acc = acc + torch.tensor([0.0, 0.0, G], device=self.device, dtype=self.dtype)  # acc -= GVEC
        # attitude: x along velocity, z along thrust direction, y = z x x (as agilib; not
        # re-orthogonalised there either, the quaternion conversion absorbs it)
        vn = vel / torch.linalg.norm(vel, dim=-1, keepdim=True).clamp_min(1e-12)
        an = acc / torch.linalg.norm(acc, dim=-1, keepdim=True).clamp_min(1e-12)
        rot = torch.stack([vn, torch.cross(an, vn, dim=-1), an], dim=-1)  # columns
        q = quat_from_matrix(rot)
        bad = ~torch.isfinite(q).all(dim=-1)
        q[bad] = q.new_tensor([1.0, 0.0, 0.0, 0.0])
        q = normalize(q)
        # hover references
        hv = self.hover
        pos = torch.where(hv.unsqueeze(-1), self.hover_pos, pos)
        vel = torch.where(hv.unsqueeze(-1), torch.zeros_like(vel), vel)
        acc = torch.where(hv.unsqueeze(-1), acc.new_tensor([0.0, 0.0, G]).expand_as(acc), acc)
        q = torch.where(hv.unsqueeze(-1), q.new_tensor([1.0, 0.0, 0.0, 0.0]).expand_as(q), q)
        return pos, vel, acc, q

    def setpoint_R(self, t: torch.Tensor):
        pos, vel, acc, q = self.setpoint(t)
        return pos, vel, acc, q, matrix_from_quat(q)
