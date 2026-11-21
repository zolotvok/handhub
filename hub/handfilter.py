# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""Where a hand is, estimated continuously from cameras that do not expose together.

Combining "the two pictures taken at the same time" does not work with ordinary cameras:
they are never at the same time, one of them often does not see the hand at all, and a
picture 70 ms old of a moving hand puts it centimetres away. So instead of solving the
hand's place afresh from whichever pictures happen to be recent, the palm's position and
velocity are kept as a running estimate (a Kalman filter), and every picture, from any
camera, at its own time, corrects it in the two directions that picture can see:

  * a camera's view of the palm says which line from the camera the palm is on;
  * the hand's size in the picture says roughly how far along that line (weakly);
  * the body skeleton's wrist says roughly where the hand is at all (more weakly still).

Between pictures the estimate moves on with the hand's velocity. A second camera seeing
the hand a moment later then fixes the distance the first could not, without the two
having to be paired up.
"""
from __future__ import annotations

import numpy as np

ACCEL_SIGMA = 2.0         # m/s^2: how hard a hand is taken to change its velocity between pictures
VELOCITY_TAU_S = 0.4      # a hand does not coast: unconfirmed velocity fades over this long
GATE = 16.0               # squared Mahalanobis distance beyond which a measurement is not believed
MAX_BACK_S = 0.25         # a picture older than this against the estimate is ignored
MAX_REJECTS = 8           # this many disbelieved measurements in a row: the estimate itself is wrong


class PalmFilter:
    def __init__(self):
        self.x = np.zeros(6)          # position (3), velocity (3), in the reference frame
        self.p = np.eye(6)
        self.t = -1e9
        self.ok = False
        self.rejects = 0

    def start(self, pos: np.ndarray, t: float, sigma: float = 0.15) -> None:
        self.x = np.r_[np.asarray(pos, float), np.zeros(3)]
        self.p = np.diag([sigma ** 2] * 3 + [1.0] * 3)
        self.t, self.ok, self.rejects = t, True, 0

    def predict(self, t: float) -> None:
        dt = t - self.t
        if dt <= 0.0:
            return
        keep = float(np.exp(-dt / VELOCITY_TAU_S))
        f = np.eye(6)
        f[:3, 3:] = np.eye(3) * dt
        f[3:, 3:] = np.eye(3) * keep
        q = np.zeros((6, 6))
        q[:3, :3] = np.eye(3) * dt ** 4 / 4.0
        q[:3, 3:] = q[3:, :3] = np.eye(3) * dt ** 3 / 2.0
        q[3:, 3:] = np.eye(3) * dt ** 2
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + q * ACCEL_SIGMA ** 2
        self.t = t

    def position(self, t: float | None = None) -> np.ndarray:
        """The palm now, or carried on along its velocity to time t (at most 100 ms)."""
        if t is None:
            return self.x[:3].copy()
        return self.x[:3] + self.x[3:] * float(np.clip(t - self.t, 0.0, 0.1))

    def _apply(self, y: np.ndarray, h: np.ndarray, r: np.ndarray) -> bool:
        s = h @ self.p @ h.T + r
        s_inv = np.linalg.inv(s)
        if float(y @ s_inv @ y) > GATE:
            self.rejects += 1
            return False
        k = self.p @ h.T @ s_inv
        self.x = self.x + k @ y
        self.p = (np.eye(6) - k @ h) @ self.p
        self.p = (self.p + self.p.T) / 2.0
        self.rejects = 0
        return True

    def _at(self, t: float):
        """Bring the estimate to a measurement's time. A measurement newer than the
        estimate moves the estimate forward; an older one (cameras report out of order)
        is compared with where the palm was then. Returns the lag, or None if too old."""
        if t >= self.t:
            self.predict(t)
            return 0.0
        back = self.t - t
        return back if back <= MAX_BACK_S else None

    def see(self, t: float, r_cam: np.ndarray, centre: np.ndarray, xn: np.ndarray, sigma: float) -> bool:
        """A camera saw the palm at normalised image position xn (x/z, y/z).
        r_cam, centre: the camera's rotation (reference <- camera) and position."""
        back = self._at(t)
        if back is None:
            return False
        pc = r_cam.T @ (self.x[:3] - self.x[3:] * back - centre)
        if pc[2] < 0.05:
            self.rejects += 1
            return False
        j = np.array([[1.0 / pc[2], 0.0, -pc[0] / pc[2] ** 2], [0.0, 1.0 / pc[2], -pc[1] / pc[2] ** 2]]) @ r_cam.T
        h = np.hstack([j, -back * j])
        return self._apply(np.asarray(xn, float) - pc[:2] / pc[2], h, np.eye(2) * sigma ** 2)

    def far(self, t: float, centre: np.ndarray, distance: float, sigma: float) -> bool:
        """The palm is about this far from a camera (judged from the hand's size)."""
        back = self._at(t)
        if back is None:
            return False
        d = self.x[:3] - self.x[3:] * back - centre
        rng = float(np.linalg.norm(d))
        if rng < 1e-6:
            return False
        u = d / rng
        h = np.hstack([u, -back * u])[None, :]
        return self._apply(np.array([distance - rng]), h, np.array([[sigma ** 2]]))

    def near(self, t: float, pos: np.ndarray, sigma: float) -> bool:
        """The palm is somewhere around this point (the skeleton's wrist)."""
        back = self._at(t)
        if back is None:
            return False
        h = np.hstack([np.eye(3), -back * np.eye(3)])
        return self._apply(np.asarray(pos, float) - (self.x[:3] - self.x[3:] * back), h, np.eye(3) * sigma ** 2)
