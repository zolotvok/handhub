# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""MediaPipe Tasks HandLandmarker backend (VIDEO mode, CPU)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from ..config import CACHE_DIR
from .backend import HandDetection

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")


POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
# Indices into MediaPipe's 33 pose landmarks used to locate the head: nose, left eye,
# right eye, left ear, right ear, mouth left, mouth right.
HEAD_LANDMARKS = (0, 2, 5, 7, 8, 9, 10)


def ensure_model(url: str = MODEL_URL) -> Path:
    path = CACHE_DIR / url.rsplit("/", 1)[1]
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        print(f"[tracking] downloading {url}")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


class HeadBackend:
    """Head landmarks from MediaPipe Pose (works from the side and partly from behind)."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(POSE_MODEL_URL))),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
        )
        self._lm = vision.PoseLandmarker.create_from_options(opts)
        self._last_ts = -1

    def detect(self, rgb: np.ndarray, timestamp_ms: int):
        """Returns (pixels (33, 2), visibility (33,), world (33, 3)) for the whole body, or None.

        `world` is MediaPipe's metric 3D estimate in camera-like axes (x right, y down,
        z away from the camera), centred on the hips.
        """
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)
        self._last_ts = timestamp_ms
        h, w = rgb.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._lm.detect_for_video(image, timestamp_ms)
        if not res.pose_landmarks:
            return None
        lms, wl = res.pose_landmarks[0], res.pose_world_landmarks[0]
        px = np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float32)
        vis = np.array([p.visibility or 0.0 for p in lms], dtype=np.float32)
        world = np.array([[p.x, p.y, p.z] for p in wl], dtype=np.float32)
        return px, vis, world

    def close(self) -> None:
        self._lm.close()


class RoiHandBackend:
    """Hand landmarks in a cropped region (one hand, no tracking between calls).

    The hand detector shrinks its input to a small square, so a hand that is small in
    the full picture (a person standing far from the camera) disappears. Cropping
    around the wrist that the body tracker found gives it a hand it can see.
    """

    def __init__(self, num_hands: int = 1):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.IMAGE, num_hands=num_hands,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)

    def detect(self, rgb: np.ndarray, x0: int, y0: int, x1: int, y1: int, size: int = 256):
        """Detect one hand inside the box; landmarks are returned in full-image pixels."""
        found = self.detect_all(rgb, x0, y0, x1, y1, size)
        return found[0] if found else None

    def detect_all(self, rgb: np.ndarray, x0: int, y0: int, x1: int, y1: int, size: int = 256) -> list:
        """All hands inside the box (up to num_hands), in full-image pixels."""
        import cv2

        crop = rgb[y0:y1, x0:x1]
        if crop.shape[0] < 16 or crop.shape[1] < 16:
            return []
        up = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(up))
        res = self._lm.detect(image)
        return [HandDetection(
            handedness=handed[0].category_name, score=float(handed[0].score),
            image=np.array([[x0 + p.x * (x1 - x0), y0 + p.y * (y1 - y0)] for p in lms], dtype=np.float32),
            world=np.array([[p.x, p.y, p.z] for p in wlms], dtype=np.float32),
        ) for lms, wlms, handed in zip(res.hand_landmarks, res.hand_world_landmarks, res.handedness)]

    def close(self) -> None:
        self._lm.close()


class MediaPipeBackend:
    def __init__(self, num_hands: int = 2):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._last_ts = -1

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> list[HandDetection]:
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)  # must be strictly increasing
        self._last_ts = timestamp_ms
        h, w = rgb.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._lm.detect_for_video(image, timestamp_ms)
        out = []
        for lms, wlms, handed in zip(res.hand_landmarks, res.hand_world_landmarks, res.handedness):
            out.append(HandDetection(
                handedness=handed[0].category_name,
                score=float(handed[0].score),
                image=np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float32),
                world=np.array([[p.x, p.y, p.z] for p in wlms], dtype=np.float32),
            ))
        return out

    def close(self) -> None:
        self._lm.close()
