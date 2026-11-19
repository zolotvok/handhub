# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""Tracker backend interface, so MediaPipe can be swapped (e.g. for ONNX Runtime)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class HandDetection:
    handedness: str  # "Left" | "Right", as reported by the model (selfie convention)
    score: float
    image: np.ndarray  # (21, 2) pixels in the raw frame
    world: np.ndarray  # (21, 3) metres, hand-centred, camera-like axes


class TrackerBackend(Protocol):
    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> list[HandDetection]: ...

    def close(self) -> None: ...
