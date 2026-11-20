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


def _sightings(rng, a, b, off, n=200, wrong_share=0.0, noise_deg=0.5, turn=1.0):
    """Palms in front of a head that turns, as the head camera sees them."""
    d, x, m, h, grp = [], [], [], [], []
    for i in range(n):
        mi = _rand_rot(rng, turn)
        head = np.array([0.0, -0.6, 1.2]) + rng.normal(0, 0.05, 3)
        r, c = hc.camera_extrinsics(a, b, off, mi, head)
        palm_cam = np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.2, 0.25), rng.uniform(0.3, 0.6)])
        palm = r @ palm_cam + c
        seen = palm_cam + rng.normal(0, np.radians(noise_deg), 3) * np.linalg.norm(palm_cam)
        other = palm + np.array([0.3, 0.05, 0.0])  # the other hand, 30 cm away
        for cand in ((palm, other) if rng.random() >= wrong_share else (other,)):
            d.append(seen / np.linalg.norm(seen)), x.append(cand), m.append(mi), h.append(head), grp.append(i)
    return np.array(d), np.array(x), np.array(m), np.array(h), np.array(grp)


def _guesses():
    return [(g.rot_x(np.pi) @ g.rot_y(y), b0, np.zeros(3)) for b0 in hc.MOUNT_GUESSES[:2]
            for y in np.radians(np.arange(0, 360, 30))]


def test_solve_mount_recovers_everything_with_both_hands_listed():
    rng = np.random.default_rng(11)
    a_true = g.rot_x(np.pi) @ g.rot_y(0.9) @ g.rot_x(0.25)
    b_true = hc.MOUNT_GUESSES[0] @ g.rot_x(0.3) @ g.rot_y(-0.1)
    off_true = np.array([0.07, -0.03, -0.05])
    fit = hc.solve_mount(*_sightings(rng, a_true, b_true, off_true, wrong_share=0.1), _guesses())
    assert fit["err_deg"] < 1.0
    assert hc.rotation_angle_deg(fit["a"], a_true) < 2.0
    assert hc.rotation_angle_deg(fit["b"], b_true) < 2.0
    assert np.linalg.norm(fit["off"] - off_true) < 0.03


def test_solve_mount_follows_heading_drift_with_the_head_still():
    rng = np.random.default_rng(12)
    a_true = g.rot_x(np.pi) @ g.rot_y(0.9)
    b_true, off_true = hc.MOUNT_GUESSES[0] @ g.rot_x(0.3), np.array([0.07, -0.03, -0.05])
    drifted = a_true @ g.rot_y(np.radians(12))          # the runtime's heading wandered 12 degrees
    data = _sightings(rng, drifted, b_true, off_true, n=80, turn=0.05)
    fit = hc.solve_mount(*data, [(a_true, b_true, off_true)], fix_mount=True)
    assert fit["err_deg"] < 1.0
    assert hc.rotation_angle_deg(fit["a"], drifted) < 2.0
    assert hc.rotation_angle_deg(fit["b"], b_true) < 1e-6


def test_solve_mount_needs_sightings():
    rng = np.random.default_rng(13)
    data = _sightings(rng, np.eye(3), hc.MOUNT_GUESSES[0], np.zeros(3), n=5)
    assert hc.solve_mount(*data, _guesses()) is None
