"""Batched 2nd-order Butterworth low-pass (agilib ``LowPassFilter``)."""

from __future__ import annotations

import math
import torch


class LowPassFilter:
    def __init__(self, cutoff_hz: float, sampling_hz: float, num_envs: int, dim: int, device, dtype=torch.float32):
        K = math.tan(math.pi * cutoff_hz / sampling_hz)
        poly = K * K + math.sqrt(2.0) * K + 1.0
        self.b0 = K * K / poly
        self.b1 = 2.0 * self.b0
        self.a1 = 2.0 * (K * K - 1.0) / poly
        self.a2 = (K * K - math.sqrt(2.0) * K + 1.0) / poly
        self.x1 = torch.zeros(num_envs, dim, device=device, dtype=dtype)
        self.x2 = torch.zeros_like(self.x1)
        self.y1 = torch.zeros_like(self.x1)
        self.y2 = torch.zeros_like(self.x1)

    def reset(self, env_ids, value: float = 0.0):
        for b in (self.x1, self.x2, self.y1, self.y2):
            b[env_ids] = value

    def add(self, x: torch.Tensor) -> torch.Tensor:
        y = self.b0 * x + self.b1 * self.x1 + self.b0 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2
        self.x2, self.x1 = self.x1, x.clone()
        self.y2, self.y1 = self.y1, y
        return y
