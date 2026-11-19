# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""A camera that moves with the head (the headset phone's own camera).

Its pose at any moment comes from two things the hub already has:
  * where the head is: the eyes, tracked by the fixed cameras, in the webcam's frame;
  * which way the head is turned: the headset's orientation from its motion sensors,
    reported by the VR runtime in the runtime's own world frame.

Two constant rotations tie these together and are found by calibration:
    R_ref<-cam(t) = A @ M(t) @ B
where M(t) is the headset orientation (runtime world <- headset), B turns camera axes
into headset axes (how the camera sits in the headset) and A turns the runtime's world
into the webcam's frame (the runtime's "north" is arbitrary). The camera's position is
the head position plus a small constant offset expressed in headset axes.

Calibration data are moments when a hand is known in 3D from the fixed cameras and is
also seen by the head camera: solving that hand's pose in the head camera's picture
gives the camera's true orientation and position at that moment.
"""
from __future__ import annotations

import cv2
import numpy as np

# Camera axes (x right, y down, z forward) in headset axes (x right, y up, -z forward),
# for the four ways a phone can sit in a viewer; calibration starts from the best of these.
MOUNT_GUESSES = [
    np.diag([1.0, -1.0, -1.0]),                                   # landscape, camera upright
    np.diag([-1.0, 1.0, -1.0]),                                   # landscape, upside down
    np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),   # portrait
    np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),  # portrait, other way up
]


def _nearest_rotation(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m)
    return u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0))))


def camera_pose_from_hand(pts_ref: np.ndarray, px: np.ndarray, k: np.ndarray):
    """Pose of a camera from a hand's known 3D landmarks (21, 3) and their pixels (21, 2).

    Returns (R ref<-cam, camera centre in ref, reprojection RMS px) or None.
    """
    obj = np.ascontiguousarray(pts_ref, dtype=np.float64)
    img = np.ascontiguousarray(px, dtype=np.float64)
    dist = np.zeros(5)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, k, dist, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        return None
    if not ok or float(tvec[2, 0]) <= 0.05:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    r_cr, _ = cv2.Rodrigues(rvec)
    return r_cr.T, (-r_cr.T @ tvec).reshape(3), rms


def solve_alignment(r_refcam: list, m_hmd: list, iterations: int = 40):
    """Find A and B with r_refcam[i] ~= A @ m_hmd[i] @ B.

    Needs the head turned in different directions among the samples; with the head
    held still the two rotations cannot be separated. Returns (A, B, mean error in
    degrees, worst error in degrees) or None.
    """
    if len(r_refcam) < 6:
        return None
    best = None
    for b0 in MOUNT_GUESSES:
        b = b0
        for _ in range(iterations):
            a = _nearest_rotation(sum(r @ (m @ b).T for r, m in zip(r_refcam, m_hmd)))
            b = _nearest_rotation(sum((a @ m).T @ r for r, m in zip(r_refcam, m_hmd)))
        errs = [rotation_angle_deg(a @ m @ b, r) for r, m in zip(r_refcam, m_hmd)]
        if best is None or np.mean(errs) < best[2]:
            best = (a, b, float(np.mean(errs)), float(np.max(errs)))
    return best


def head_spread_deg(m_hmd: list) -> float:
    """How much the head turned across the samples (largest angle from the first)."""
    return max((rotation_angle_deg(m_hmd[0], m) for m in m_hmd), default=0.0)


def camera_extrinsics(a: np.ndarray, b: np.ndarray, offset_hmd: np.ndarray, m_hmd: np.ndarray,
                      head_ref: np.ndarray):
    """The moving camera's (R ref<-cam, centre in ref) for one moment."""
    return a @ m_hmd @ b, np.asarray(head_ref, float) + a @ m_hmd @ np.asarray(offset_hmd, float)
