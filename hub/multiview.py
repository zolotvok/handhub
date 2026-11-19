# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""Multi-camera geometry: camera-to-camera calibration from hand observations, and
triangulation of hand landmarks.

Conventions: a camera's extrinsics (R, t) map its coordinates into the reference
(primary) camera's frame: X_ref = R @ X_cam + t. Pixels use a pinhole K, no distortion.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
_EA = np.array([a for a, _ in HAND_EDGES])
_EB = np.array([b for _, b in HAND_EDGES])

MIN_PAIR_SAMPLES = 25


def normalize_px(px: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Pixels (..., 2) -> normalized image coordinates."""
    return (np.asarray(px, float) - k[:2, 2]) / np.array([k[0, 0], k[1, 1]])


def bone_length(pts: np.ndarray) -> np.ndarray:
    """Sum of hand bone lengths for (..., 21, 3) points."""
    return np.linalg.norm(pts[..., _EA, :] - pts[..., _EB, :], axis=-1).sum(axis=-1)


def triangulate(views: list) -> tuple[np.ndarray, float]:
    """DLT triangulation of landmarks seen by two or more cameras.

    views: (k, r, t, px (N, 2), weight) per camera, with (r, t) the camera's extrinsics.
    Returns (points (N, 3) in the reference frame, RMS reprojection error in pixels).
    """
    rows = []
    for k, r, t, px, w in views:
        p = np.hstack([r.T, (-r.T @ t).reshape(3, 1)])  # reference -> camera
        xy = normalize_px(px, k)
        rows.append(w * (xy[:, 0, None] * p[2] - p[0]))
        rows.append(w * (xy[:, 1, None] * p[2] - p[1]))
    a = np.stack(rows, axis=1)  # (N, 2 * views, 4)
    _, _, vt = np.linalg.svd(a)
    x = vt[:, -1, :]
    pts = x[:, :3] / np.where(np.abs(x[:, 3:]) < 1e-12, 1e-12, x[:, 3:])

    sq = []
    for k, r, t, px, _ in views:
        cam = (pts - t) @ r  # r.T @ (X - t)
        z = np.where(np.abs(cam[:, 2:]) < 1e-9, 1e-9, cam[:, 2:])
        proj = cam[:, :2] / z * np.array([k[0, 0], k[1, 1]]) + k[:2, 2]
        sq.append(np.sum((proj - px) ** 2, axis=1))
    return pts, float(np.sqrt(np.mean(sq)))


def triangulate_robust(views: list, max_rms: float):
    """Triangulate, dropping a camera that disagrees with the others.

    With three or more views whose joint solution is poor (one camera was moved since
    calibration, or matched the wrong thing), the best-agreeing pair is used instead.
    Returns (points, rms, indices of the views used) or None.
    """
    if len(views) < 2:
        return None
    pts, rms = triangulate(views)
    if rms <= max_rms:
        return pts, rms, list(range(len(views)))
    best = None
    if len(views) > 2:
        for i in range(len(views)):
            for j in range(i + 1, len(views)):
                p, r = triangulate([views[i], views[j]])
                if r <= max_rms and (best is None or r < best[1]):
                    best = (p, r, [i, j])
    return best


BODY_SEGMENTS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
                 (23, 25), (25, 27), (24, 26), (26, 28)]
MIN_PAIR_POINTS = 250
MIN_PAIR_SAMPLES_ANY = 10


def solve_samples(samples: list, k_a: np.ndarray, k_b: np.ndarray) -> dict:
    """Relative pose of camera B in camera A's frame from simultaneous observations.

    samples: (a_px (n, 2), b_px (n, 2), segments) per moment, where the two pixel arrays
    are the same n landmarks (of a hand or of the body) and segments is a list of
    (i, j, metres): known distances between landmarks i and j, which fix the scale
    (an essential matrix has none).
    Returns {"ok", "r", "t" (X_a = r @ X_b + t), "rms_px", "inliers", "samples"} or
    {"ok": False, "error", "samples"}.
    """
    s = len(samples)
    n_pts = sum(len(x[0]) for x in samples)
    # Body samples carry many landmarks each, so fewer moments are needed than with a hand.
    if s < MIN_PAIR_SAMPLES_ANY or (s < MIN_PAIR_SAMPLES and n_pts < MIN_PAIR_POINTS):
        return {"ok": False, "error": f"only {s} shared samples (need {MIN_PAIR_SAMPLES})", "samples": s}
    a_all = np.concatenate([x[0] for x in samples]).astype(float)
    b_all = np.concatenate([x[1] for x in samples]).astype(float)
    pa, pb = normalize_px(a_all, k_a), normalize_px(b_all, k_b)
    focal = float(np.mean([k_a[0, 0], k_b[0, 0]]))
    e, mask = cv2.findEssentialMat(pb, pa, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=5.0 / focal)
    if e is None or e.shape[0] < 3:
        return {"ok": False, "error": "no consistent geometry found", "samples": s}
    _, r, t, mask = cv2.recoverPose(e[:3], pb, pa, np.eye(3), mask=mask)
    t = t.reshape(3)
    inlier = mask.reshape(-1) > 0
    inlier_ratio = float(inlier.mean())
    if inlier_ratio < 0.4:
        return {"ok": False, "error": f"cameras disagree ({inlier_ratio:.0%} consistent points)", "samples": s}

    eye, zero = np.eye(3), np.zeros(3)
    pts, _ = triangulate([(k_a, eye, zero, a_all, 1.0), (k_b, r, t, b_all, 1.0)])
    in_front = inlier & (pts[:, 2] > 0)
    ratios, start = [], 0
    for a_px, _, segments in samples:
        for i, j, metres in segments:
            if in_front[start + i] and in_front[start + j]:
                d = float(np.linalg.norm(pts[start + i] - pts[start + j]))
                if d > 1e-6 and metres > 1e-3:
                    ratios.append(metres / d)
        start += len(a_px)
    if len(ratios) < 30:
        return {"ok": False, "error": "too few clean samples to set the scale", "samples": s}
    t = t * float(np.median(ratios))

    # Error on the consistent points, at metric scale.
    _, rms = triangulate([(k_a, eye, zero, a_all[in_front], 1.0), (k_b, r, t, b_all[in_front], 1.0)])
    return {"ok": True, "r": r, "t": t, "rms_px": rms, "inliers": inlier_ratio, "samples": s}


def hand_segments(world: np.ndarray, hand_scale: float = 1.0) -> list:
    w = np.asarray(world, float) * hand_scale
    return [(a, b, float(np.linalg.norm(w[a] - w[b]))) for a, b in HAND_EDGES]


def solve_pair(a_px: np.ndarray, b_px: np.ndarray, k_a: np.ndarray, k_b: np.ndarray,
               world: np.ndarray, hand_scale: float = 1.0) -> dict:
    """solve_samples for whole-hand observations: (S, 21, 2) pixels and (S, 21, 3) world."""
    return solve_samples([(a, b, hand_segments(w, hand_scale)) for a, b, w in zip(a_px, b_px, world)], k_a, k_b)


def chain_extrinsics(pairs: dict, reference: str) -> dict:
    """Combine pair solutions {(a, b): (r, t)} with X_a = r @ X_b + t into extrinsics
    {name: (R, t)} relative to `reference`, following whichever pairs connect."""
    out = {reference: (np.eye(3), np.zeros(3))}
    todo = [reference]
    while todo:
        cur = todo.pop()
        r_cur, t_cur = out[cur]
        for (a, b), (r, t) in pairs.items():
            if a == cur and b not in out:
                out[b] = (r_cur @ r, r_cur @ t + t_cur)
                todo.append(b)
            elif b == cur and a not in out:
                out[a] = (r_cur @ r.T, t_cur - r_cur @ r.T @ t)  # X_b = r.T @ (X_a - t)
                todo.append(a)
    return out


class PairCollector:
    """Gathers simultaneous observations of the same landmarks by pairs of cameras.

    Each camera reports "tracks": a hand (21 landmarks) or the body (33 landmarks),
    with a per-landmark validity mask and known segment lengths. The cameras are not
    synchronized, so only slow movement is used: the track must be nearly still in
    both views, which keeps the timing error small.
    """

    def __init__(self, max_speed: float = 0.25, min_move: float = 0.03, max_samples: int = 500):
        self.max_speed, self.min_move, self.max_samples = max_speed, min_move, max_samples
        self.samples: dict = {}  # (a, b) -> list of (a_px, b_px, segments)
        self._last: dict = {}  # (cam, track) -> deque of (t_s, px, valid, segments, width)
        self._last_kept: dict = {}  # (a, b, track) -> (a_px, valid) of the last stored sample

    def _speed(self, cam: str, track: str) -> float:
        """Recent landmark speed in image widths per second (inf if unknown)."""
        h = self._last.get((cam, track))
        if not h or len(h) < 2:
            return float("inf")
        (t0, p0, v0, _, w), (t1, p1, v1, _, _) = h[0], h[-1]
        both = v0 & v1
        if both.sum() < 4:
            return float("inf")
        return float(np.mean(np.linalg.norm(p1[both] - p0[both], axis=1))) / w / max(t1 - t0, 1e-3)

    def add(self, cam: str, track: str, t_s: float, px: np.ndarray, valid: np.ndarray, segments: list,
            width: int, order: list, max_dt_s: float = 0.05) -> None:
        """Record cam's observation of a track and pair it with other cameras' fresh ones.

        segments: (i, j, metres) known distances between landmarks of this track.
        """
        px, valid = np.asarray(px, float), np.asarray(valid, bool)
        hist = self._last.setdefault((cam, track), deque(maxlen=3))
        hist.append((t_s, px, valid, segments, width))
        if self._speed(cam, track) > self.max_speed:
            return
        for (other, tr), ohist in list(self._last.items()):
            if tr != track or other == cam or not ohist:
                continue
            t_o, px_o, valid_o, _, w_o = ohist[-1]
            if abs(t_s - t_o) > max_dt_s or self._speed(other, track) > self.max_speed:
                continue
            common = valid & valid_o
            if common.sum() < 6:
                continue
            a, b = sorted((cam, other), key=order.index)
            a_px, b_px, a_w = (px, px_o, width) if a == cam else (px_o, px, w_o)
            prev = self._last_kept.get((a, b, track))
            if prev is not None:
                both = common & prev[1]
                if both.sum() >= 4 and np.mean(np.linalg.norm(a_px[both] - prev[0][both], axis=1)) / a_w < self.min_move:
                    continue  # nothing has moved since the last sample: no new information
            index = {int(i): n for n, i in enumerate(np.flatnonzero(common))}
            segs = [(index[i], index[j], m) for i, j, m in segments if i in index and j in index]
            lst = self.samples.setdefault((a, b), [])
            if len(lst) < self.max_samples:
                lst.append((a_px[common].copy(), b_px[common].copy(), segs))
                self._last_kept[(a, b, track)] = (a_px.copy(), common.copy())

    def counts(self) -> dict:
        return {f"{a}+{b}": len(v) for (a, b), v in self.samples.items()}


# ---- joint refinement ------------------------------------------------------------

def _rodrigues(v: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(v, float).reshape(3, 1))[0]


def _pair_residuals(a_px, b_px, k_a, k_b, ext_a, ext_b) -> np.ndarray:
    """Reprojection residuals (N, 4) of points triangulated from two views, in pixels
    scaled to a 640-pixel-wide image so cameras of different resolution weigh alike."""
    views = [(k_a, *ext_a, a_px, 1.0), (k_b, *ext_b, b_px, 1.0)]
    rows = []
    for k, r, t, px, _ in views:
        p = np.hstack([r.T, (-r.T @ t).reshape(3, 1)])
        xy = normalize_px(px, k)
        rows.append(xy[:, 0, None] * p[2] - p[0])
        rows.append(xy[:, 1, None] * p[2] - p[1])
    _, _, vt = np.linalg.svd(np.stack(rows, axis=1))
    x = vt[:, -1, :]
    pts = x[:, :3] / np.where(np.abs(x[:, 3:]) < 1e-12, 1e-12, x[:, 3:])
    out = []
    for k, r, t, px, _ in views:
        cam = (pts - t) @ r
        z = np.where(np.abs(cam[:, 2:]) < 1e-6, 1e-6, cam[:, 2:])
        proj = cam[:, :2] / z * np.array([k[0, 0], k[1, 1]]) + k[:2, 2]
        out.append((proj - px) * (640.0 / (2.0 * max(k[0, 2], k[1, 2]))))
    return np.hstack(out)


def refine_joint(samples: dict, intrinsics, hfov: dict, ext: dict, reference: str,
                 optimize_fov: bool = True, iterations: int = 25) -> tuple:
    """Refine all cameras' extrinsics (and fields of view) together.

    The pairwise solutions are each a little off and do not have to be consistent with
    one another. This minimizes the reprojection error of every shared observation
    over all cameras at once (a bundle adjustment with the 3D points re-triangulated
    at each step), with a robust loss so mismatched points do not pull it.

    samples: {(a, b): [(a_px, b_px, segments), ...]}; intrinsics(name, hfov_deg) -> K;
    ext: {name: (R, t)} initial, including the reference camera.
    Returns (ext, hfov, rms_before, rms_after) with rms in 640-wide pixels.
    """
    names = [n for n in ext if n != reference]
    pairs = {k: (np.concatenate([s[0] for s in v]).astype(float), np.concatenate([s[1] for s in v]).astype(float))
             for k, v in samples.items() if k[0] in ext and k[1] in ext and v}
    if not names or not pairs:
        return ext, hfov, None, None
    fov_names = list(ext) if optimize_fov else []
    r0 = {n: ext[n][0] for n in names}

    def unpack(p):
        e = {reference: ext[reference]}
        for i, n in enumerate(names):
            e[n] = (r0[n] @ _rodrigues(p[6 * i:6 * i + 3]), ext[n][1] + p[6 * i + 3:6 * i + 6])
        f = dict(hfov)
        for j, n in enumerate(fov_names):
            f[n] = float(np.clip(hfov[n] * np.exp(p[6 * len(names) + j]), 30.0, 150.0))
        return e, f

    def residuals(p):
        e, f = unpack(p)
        ks = {n: intrinsics(n, f[n]) for n in e}
        return np.concatenate([_pair_residuals(a_px, b_px, ks[a], ks[b], e[a], e[b]).ravel()
                               for (a, b), (a_px, b_px) in pairs.items()])

    def robust(r):  # Huber weights, 6 px
        a = np.abs(r)
        return np.where(a <= 6.0, 1.0, 6.0 / np.maximum(a, 1e-9))

    def rms(r):
        return float(np.sqrt(np.mean(np.minimum(r * r, 36.0 * 4))))  # outliers capped, not ignored

    p = np.zeros(6 * len(names) + len(fov_names))
    # The overall scale cannot be found from reprojection: hold the first camera's
    # distance from the reference by not optimizing along its baseline direction.
    r = residuals(p)
    start = rms(r)
    lam = 1e-2
    steps = np.full(len(p), 1e-4)
    for _ in range(iterations):
        w = np.sqrt(robust(r))
        jac = np.empty((len(r), len(p)))
        for i in range(len(p)):
            d = np.zeros(len(p))
            d[i] = steps[i]
            jac[:, i] = (residuals(p + d) - r) / steps[i]
        jw, rw = jac * w[:, None], r * w
        h, gvec = jw.T @ jw, jw.T @ rw
        improved = False
        for _ in range(6):
            try:
                dp = np.linalg.solve(h + lam * np.diag(np.diag(h) + 1e-9), -gvec)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            # keep the scale: remove motion of the first camera along its own baseline
            base = ext[names[0]][1] + p[3:6]
            n_base = float(np.linalg.norm(base))
            if n_base > 1e-6:
                u = base / n_base
                dp[3:6] -= u * float(dp[3:6] @ u)
            r_new = residuals(p + dp)
            if rms(r_new * np.sqrt(robust(r_new))) < rms(r * w):
                p, r, lam, improved = p + dp, r_new, lam * 0.4, True
                break
            lam *= 6
        if not improved:
            break
    e, f = unpack(p)
    return e, f, start, rms(r)


def metric_scale(samples: dict, intrinsics, hfov: dict, ext: dict) -> float | None:
    """Factor that brings a refined solution to metres: the median, over every known
    segment in every pair's samples, of its real length divided by its triangulated one."""
    ratios = []
    for (a, b), lst in samples.items():
        if a not in ext or b not in ext:
            continue
        k_a, k_b = intrinsics(a, hfov[a]), intrinsics(b, hfov[b])
        for a_px, b_px, segments in lst:
            if not segments:
                continue
            pts, rms = triangulate([(k_a, *ext[a], np.asarray(a_px, float), 1.0),
                                    (k_b, *ext[b], np.asarray(b_px, float), 1.0)])
            if rms > 12.0:
                continue
            for i, j, metres in segments:
                d = float(np.linalg.norm(pts[i] - pts[j]))
                if d > 1e-6 and metres > 0.05:
                    ratios.append(metres / d)
    return float(np.median(ratios)) if len(ratios) >= 30 else None
