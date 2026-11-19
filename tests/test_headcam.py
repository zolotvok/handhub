# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""The head-mounted camera's alignment on synthetic data."""
import numpy as np

from hub import geometry as g
from hub import headcam as hc


def _rand_rot(rng, scale=1.0):
    return g.rot_y(rng.uniform(-1, 1) * scale) @ g.rot_x(rng.uniform(-0.6, 0.6) * scale) @ g.rot_z(rng.uniform(-0.4, 0.4) * scale)


def test_solve_alignment_recovers_mount_and_world_rotation():
    rng = np.random.default_rng(5)
    a_true = g.rot_y(1.1) @ g.rot_x(-0.7) @ g.rot_z(0.05)          # runtime world -> webcam frame
    b_true = hc.MOUNT_GUESSES[1] @ g.rot_x(0.08) @ g.rot_y(-0.05)   # upside-down phone, slightly askew
    m = [_rand_rot(rng) for _ in range(30)]
    noise = [_rand_rot(rng, 0.02) for _ in m]
    r = [a_true @ mi @ b_true @ ni for mi, ni in zip(m, noise)]
    a, b, mean_err, worst = hc.solve_alignment(r, m)
    assert hc.rotation_angle_deg(a, a_true) < 1.5
    assert hc.rotation_angle_deg(b, b_true) < 1.5
    assert mean_err < 2.0


def test_solve_alignment_needs_samples():
    assert hc.solve_alignment([np.eye(3)] * 3, [np.eye(3)] * 3) is None


def test_camera_pose_from_hand_and_extrinsics_round_trip():
    rng = np.random.default_rng(2)
    k = g.intrinsics_from_hfov(1280, 720, 100)
    hand = rng.normal(0, 0.04, (21, 3)) + np.array([0.1, -0.2, 1.4])     # in the webcam frame
    r_true = g.rot_y(2.6) @ g.rot_x(0.3)                                  # camera looking back at the hand
    c_true = np.array([0.15, -0.35, 1.9])
    cam = (hand - c_true) @ r_true
    assert cam[:, 2].min() > 0.1
    px = cam[:, :2] / cam[:, 2:] * k[0, 0] + k[:2, 2]
    r, c, rms = hc.camera_pose_from_hand(hand, px, k)
    assert hc.rotation_angle_deg(r, r_true) < 0.5
    assert np.allclose(c, c_true, atol=0.01)
    assert rms < 0.5

    # camera_extrinsics composes the same pose from its parts
    a, b = g.rot_y(0.4), hc.MOUNT_GUESSES[0]
    m = a.T @ r_true @ b.T
    offset = (a @ m).T @ (c_true - np.array([0.1, -0.3, 1.95]))
    r2, c2 = hc.camera_extrinsics(a, b, offset, m, np.array([0.1, -0.3, 1.95]))
    assert np.allclose(r2, r_true, atol=1e-9) and np.allclose(c2, c_true, atol=1e-9)
