# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""Gestures from MediaPipe world landmarks: pinch and curl values, button clicks with
hysteresis, input profiles, and holding the last state when confidence is low."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BTN_TRIGGER, BTN_GRIP, BTN_A, BTN_B, BTN_SYSTEM, BTN_STICK = 1, 2, 4, 8, 16, 32

# (mcp, tip) for index, middle, ring, pinky
FINGERS = [(5, 8), (9, 12), (13, 16), (17, 20)]


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else float(x)


def _dist(w: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(w[a] - w[b]))


@dataclass
class GestureOutput:
    trigger: float = 0.0
    grip: float = 0.0
    curl: list = field(default_factory=lambda: [0.0] * 5)  # thumb, index, middle, ring, pinky
    buttons: int = 0
    pinch: float = 1.0  # normalized thumb-index distance, for the UI


class GestureDetector:
    """Per-hand detector. Keeps click state for hysteresis."""

    def __init__(self):
        self.out = GestureOutput()

    def reset(self) -> None:
        self.out = GestureOutput()

    def update(self, w: np.ndarray, cfg: dict, confident: bool = True) -> GestureOutput:
        if not confident:
            return self.out  # hold previous states instead of guessing
        prev = self.out.buttons
        s = _dist(w, 0, 9)
        if s < 1e-6:
            return self.out

        span = cfg["curl_open"] - cfg["curl_closed"]
        curls = []
        for mcp, tip in FINGERS:
            r = _dist(w, tip, 0) / max(_dist(w, mcp, 0), 1e-6)
            curls.append(clamp01((cfg["curl_open"] - r) / span))
        # Thumb: tip distance to the pinky knuckle, relative to hand size.
        thumb = clamp01((1.3 - _dist(w, 4, 17) / s) / (1.3 - 0.6))
        grip = float(np.mean(curls[1:]))

        pinch = _dist(w, 4, 8) / s
        if cfg["profile"] == "finger":
            trigger = curls[0]
        else:
            trigger = clamp01((cfg["pinch_far"] - pinch) / (cfg["pinch_far"] - cfg["pinch_near"]))

        buttons = 0
        if trigger > cfg["trigger_click"] or (prev & BTN_TRIGGER and trigger >= cfg["trigger_release"]):
            buttons |= BTN_TRIGGER
        if grip > cfg["grip_click"] or (prev & BTN_GRIP and grip >= cfg["grip_release"]):
            buttons |= BTN_GRIP
        if cfg["profile"] != "finger" and grip < cfg["ab_max_grip"]:
            for bit, tip in ((BTN_A, 12), (BTN_B, 16)):
                d = _dist(w, 4, tip) / s
                if d < cfg["ab_pinch"] or (prev & bit and d < cfg["ab_pinch"] * 1.4):
                    buttons |= bit

        self.out = GestureOutput(trigger=trigger, grip=grip, curl=[thumb, *curls], buttons=buttons, pinch=pinch)
        return self.out
