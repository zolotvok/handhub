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

Calibration data are moments when a hand's palm is known in 3D from the fixed cameras
and is also seen by the head camera. Each gives one direction: where the head camera saw
the palm, against where the palm really was. A, B and the offset are fitted to many such
directions at once (solve_mount). The headset's heading drifts and starts anew with every
VR session, so the hub keeps fitting while it runs; one hand's pose in one picture
(camera_pose_from_hand) proved far too coarse for this.
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


def _rot_vec(v: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    kx, ky, kz = v / theta
    k = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def _wahba(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The rotation R that best gives dst ~= R @ src, for rows of unit vectors."""
    return _nearest_rotation(dst.T @ src)


def bearing_errors(a, b, off, d, x, m, h):
    """Angle (degrees) between where the head camera saw each palm and where it was.

    d: (n, 3) unit directions in the camera's axes; x: (n, 3) palm centres and h: (n, 3)
    head positions in the reference frame; m: (n, 3, 3) headset orientations.
    Returns (errors, unit directions from the camera to the palms in the reference frame).
    """
    centre = h + np.einsum("ij,njk,k->ni", a, m, off)
    u = x - centre
    u = u / np.maximum(np.linalg.norm(u, axis=1), 1e-9)[:, None]
    seen = np.einsum("ij,njk,kl,nl->ni", a, m, b, d)
    return np.degrees(np.arccos(np.clip((seen * u).sum(axis=1), -1.0, 1.0))), u


def solve_mount(d, x, m, h, group, inits: list, fix_mount: bool = False, keep_share: float = 75.0,
                off_weight: float = 0.5, off_centre=None):
    """Fit A, B and the camera's offset to palm sightings.

    group: (n,) sighting number. One sighting of a palm may be listed against both
    tracked hands when it is not known which one it was; the better-fitting one counts.
    inits: starting (A, B, offset) triples; the best result is returned.
    fix_mount: keep B and the offset as given and fit only A (enough to follow the
    headset's heading drifting, and all that can be told apart while the head is still).
    Returns {"a", "b", "off", "err_deg" (median over the sightings used), "used"} or None.
    """
    d, x, m, h = (np.asarray(v, float) for v in (d, x, m, h))
    order = np.argsort(np.asarray(group), kind="stable")
    d, x, m, h, grp = d[order], x[order], m[order], h[order], np.asarray(group)[order]
    starts = np.flatnonzero(np.r_[True, grp[1:] != grp[:-1]])
    if len(starts) < 12:
        return None
    which = np.cumsum(np.r_[True, grp[1:] != grp[:-1]]) - 1  # sighting index of each row

    def choose(a, b, off):
        """Rows to use: per sighting the better candidate, less the worst sightings."""
        e, u = bearing_errors(a, b, off, d, x, m, h)
        best = np.minimum.reduceat(e, starts)
        rows = np.flatnonzero((e == best[which]) & (best[which] <= np.percentile(best, keep_share)))
        return rows, e, u, best

    result = None
    for a, b, off in inits:
        a, b, off = np.array(a, float), np.array(b, float), np.array(off, float)
        for _ in range(12):
            rows, _, u, _ = choose(a, b, off)
            a = _wahba(np.einsum("njk,kl,nl->nj", m[rows], b, d[rows]), u[rows])
            if not fix_mount:
                b = _wahba(d[rows], np.einsum("nkj,lk,nl->nj", m[rows], a, u[rows]))
        rows, _, _, best = choose(a, b, off)
        score = float(np.percentile(best, keep_share * 0.8))
        if result is None or score < result[0]:
            result = (score, a, b, off, rows)
    _, a, b, off, rows = result

    # Refine everything together, now with the offset, on the rows chosen.
    n_par = 3 if fix_mount else 9

    def residuals(p):
        ra = _rot_vec(p[0:3]) @ a
        rb, ro = (b, off) if fix_mount else (b @ _rot_vec(p[3:6]), off + p[6:9])
        centre = h[rows] + np.einsum("ij,njk,k->ni", ra, m[rows], ro)
        u = x[rows] - centre
        u = u / np.maximum(np.linalg.norm(u, axis=1), 1e-9)[:, None]
        r = (np.einsum("ij,njk,kl,nl->ni", ra, m[rows], rb, d[rows]) - u).reshape(-1)
        # the camera sits within a hand's width of the eyes: hold the offset loosely there
        return r if fix_mount else np.r_[r, off_weight * (ro - (0.0 if off_centre is None else off_centre))]

    p = np.zeros(n_par)
    for _ in range(8):
        r0 = residuals(p)
        jac = np.empty((len(r0), n_par))
        for j in range(n_par):
            dp = np.zeros(n_par)
            dp[j] = 1e-5
            jac[:, j] = (residuals(p + dp) - r0) / 1e-5
        step = np.linalg.lstsq(jac, -r0, rcond=None)[0]
        p = p + step
        if float(np.abs(step).max()) < 1e-6:
            break
    a = _rot_vec(p[0:3]) @ a
    if not fix_mount:
        b, off = b @ _rot_vec(p[3:6]), off + p[6:9]
    rows, _, _, best = choose(a, b, off)
    kept = best[best <= np.percentile(best, keep_share)]
    return {"a": a, "b": b, "off": off, "err_deg": float(np.median(kept)), "used": int(len(kept))}
