# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""One Euro filter for vectors and quaternions."""
from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x: np.ndarray | None = None
        self.dx: np.ndarray | None = None
        self.t: float | None = None

    def reset(self) -> None:
        self.x = self.dx = self.t = None

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.x is None or self.t is None or t <= self.t:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = t - self.t
        dx = (x - self.x) / dt
        self.dx = self.dx + _alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        self.x = self.x + _alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x


class OneEuroQuat:
    """One-Euro on quaternion components with sign continuity, then renormalize."""

    def __init__(self, **kw):
        self.f = OneEuro(**kw)

    def reset(self) -> None:
        self.f.reset()

    def __call__(self, q, t: float) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if self.f.x is not None and float(q @ self.f.x) < 0:
            q = -q
        out = self.f(q, t)
        n = float(np.linalg.norm(out))
        if n < 1e-9:
            self.f.reset()
            return self.f(q, t)
        self.f.x = out / n
        return self.f.x
