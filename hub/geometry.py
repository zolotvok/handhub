# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""Frames, quaternions, PnP.

Camera (OpenCV): x right, y down, z forward.
Body: x right, y up, -z forward (the direction the user faces).
Quaternions are (w, x, y, z).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

# Camera facing the user with no tilt: 180 deg about z.
BASE_CAM_TO_BODY = np.diag([-1.0, -1.0, 1.0])

WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP = 0, 4, 5, 8
MIDDLE_MCP, MIDDLE_TIP, RING_MCP, RING_TIP, PINKY_MCP, PINKY_TIP = 9, 12, 13, 16, 17, 20
PALM_IDS = [0, 5, 9, 13, 17]


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def cam_to_body_rotation(yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """R_cam->body: base rotation plus the camera's yaw (about body y), pitch (x), roll (z)."""
    r = rot_y(math.radians(yaw_deg)) @ rot_x(math.radians(pitch_deg)) @ rot_z(math.radians(roll_deg))
    return r @ BASE_CAM_TO_BODY


def cam_to_body(p_cam: np.ndarray, r_cam_body: np.ndarray, c: np.ndarray) -> np.ndarray:
    """p_body = C + R . p_cam, for one point or an (N, 3) array."""
    return np.asarray(c, float) + np.asarray(p_cam, float) @ r_cam_body.T


def intrinsics_from_hfov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Pinhole K from the field of view along the image's long side, so a camera
    rotated to portrait keeps the same setting."""
    f = (max(width, height) / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], dtype=float)


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return normalize(np.array(q, dtype=float))


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize(np.asarray(q, float))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def yaw_offset_from_hmd(q_hmd: np.ndarray) -> float:
    """Yaw such that R_y(yaw).(0,0,-1) equals the flattened HMD forward vector."""
    f = matrix_from_quat(q_hmd) @ np.array([0.0, 0.0, -1.0])
    return math.atan2(-f[0], -f[2])


def solve_hand_pnp(world_lm: np.ndarray, image_px: np.ndarray, k: np.ndarray, scale: float = 1.0):
    """Place MediaPipe's hand-centred metric landmarks in the camera frame.

    Returns (points_cam (21, 3), reprojection_error_px) or None.
    """
    obj = np.ascontiguousarray(world_lm, dtype=np.float64) * scale
    img = np.ascontiguousarray(image_px, dtype=np.float64)
    dist = np.zeros(5)
    ok, rvec, tvec = False, None, None
    for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, k, dist, flags=flag)
        except cv2.error:
            ok = False
        if ok and float(tvec[2, 0]) > 0.05:
            break
        ok = False
    if not ok:
        return None
    r, _ = cv2.Rodrigues(rvec)
    pts = obj @ r.T + tvec.reshape(1, 3)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    return pts, err


def controller_pose(pts: np.ndarray, roll_offset_deg: float = 0.0, pitch_offset_deg: float = 0.0,
                    grip_offset=(0.0, 0.0, 0.0)):
    """Hand landmarks (21, 3) in the body frame -> (position, rotation matrix).

    z = -(along the hand), y = thumb side ("handshake" grip), x = y cross z.
    """
    f = normalize(pts[MIDDLE_MCP] - pts[WRIST])
    a = pts[INDEX_MCP] - pts[PINKY_MCP]
    a = normalize(a - f * float(a @ f))
    z = -f
    y = a
    x = np.cross(y, z)
    r = np.column_stack([x, y, z])
    r = r @ rot_z(math.radians(roll_offset_deg)) @ rot_x(math.radians(pitch_offset_deg))
    pos = pts[PALM_IDS].mean(axis=0) + r @ np.asarray(grip_offset, float)
    return pos, r


def head_rotation(world: np.ndarray, vis: np.ndarray, min_vis: float = 0.6):
    """Head orientation in the camera's frame from pose landmarks (nose, left eye,
    right eye, left ear, right ear) given in camera-like axes.

    Returns a rotation matrix whose columns are the head's right, up and backward
    axes (so -z is where the face points), or None when the face is not visible enough.
    """
    nose, l_eye, r_eye, l_ear, r_ear = np.asarray(world, float)[:5]
    if vis[0] < min_vis:
        return None
    if min(vis[3], vis[4]) >= min_vis:
        left, right = l_ear, r_ear
    elif min(vis[1], vis[2]) >= min_vis:
        left, right = l_eye, r_eye
    else:
        return None
    x = normalize(right - left)  # the person's right
    fwd = nose - (l_ear + r_ear) / 2.0
    fwd = normalize(fwd - x * float(fwd @ x))
    if float(np.linalg.norm(fwd)) < 0.5:
        return None
    y = np.cross(x, fwd)  # up (x right, -z forward => y = x cross forward)
    return np.column_stack([x, y, -fwd])


def fit_rigid(model: np.ndarray, target: np.ndarray):
    """Rotation and translation that best move `model` onto `target` (both (N, 3)),
    without scaling or bending. Returns (R, t, rms distance) with target ~= model @ R.T + t."""
    mc, tc = model.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((target - tc).T @ (model - mc))
    r = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    t = tc - r @ mc
    rms = float(np.sqrt(np.mean(np.sum((model @ r.T + t - target) ** 2, axis=1))))
    return r, t, rms


def hand_side(pts: np.ndarray, min_m: float = 0.01):
    """Left or right from a hand's 3D landmarks alone: True = right, False = left, None =
    cannot tell.

    The normal of the palm triangle (wrist, index knuckle, pinky knuckle) points out of
    the palm on a right hand and out of the back on a left hand. Fingers curl toward the
    palm and the thumb sits on the palm side, so the sign of their offset along that
    normal gives the hand. Both must agree, and the curl must be clear (a flat hand
    cannot be told from its mirror image). This needs true 3D points (two or more
    cameras); it does not depend on any camera's opinion of left and right.
    """
    n = np.cross(pts[INDEX_MCP] - pts[WRIST], pts[PINKY_MCP] - pts[WRIST])
    norm = float(np.linalg.norm(n))
    if norm < 1e-9:
        return None
    n = n / norm
    curl = float(n @ np.mean([pts[t] - pts[m] for m, t in ((5, 8), (9, 12), (13, 16), (17, 20))], axis=0))
    thumb = float(n @ (pts[THUMB_TIP] - pts[WRIST]))
    if abs(curl) < min_m or (curl > 0) != (thumb > 0):
        return None
    return curl > 0
