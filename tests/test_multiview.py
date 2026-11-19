# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""Multi-camera geometry on synthetic cameras."""
import numpy as np
import pytest

from hub import geometry as g
from hub import multiview as mv


def _hand(rng):
    """A plausible non-planar 21-point hand, metres, centred."""
    pts = np.zeros((21, 3))
    for f, x in enumerate((-0.04, -0.02, 0.0, 0.02, 0.04)):
        for j in range(4):
            pts[1 + f * 4 + j] = [x + rng.normal(0, 0.003), -0.03 - 0.025 * j, rng.normal(0, 0.01) + 0.01 * j]
    return pts


def _project(pts_ref, k, r, t):
    cam = (pts_ref - t) @ r
    return cam[:, :2] / cam[:, 2:] * np.array([k[0, 0], k[1, 1]]) + k[:2, 2]


@pytest.fixture
def scene():
    rng = np.random.default_rng(7)
    k_a = g.intrinsics_from_hfov(640, 480, 70)
    k_b = g.intrinsics_from_hfov(1280, 720, 100)
    # Camera B: 1.2 m to the right of A, turned 60 degrees toward the scene.
    r_b = g.rot_y(np.radians(-60)) @ g.rot_x(np.radians(5))
    t_b = np.array([1.2, 0.1, 0.3])
    a_px, b_px, world, truth = [], [], [], []
    for _ in range(80):
        hand = _hand(rng)
        rot = g.rot_y(rng.uniform(-0.6, 0.6)) @ g.rot_x(rng.uniform(-0.6, 0.6)) @ g.rot_z(rng.uniform(-1, 1))
        centre = np.array([rng.uniform(-0.3, 0.5), rng.uniform(-0.3, 0.3), rng.uniform(0.7, 1.4)])
        pts = hand @ rot.T + centre
        truth.append(pts)
        a_px.append(_project(pts, k_a, np.eye(3), np.zeros(3)))
        b_px.append(_project(pts, k_b, r_b, t_b))
        world.append(hand)
    return dict(k_a=k_a, k_b=k_b, r_b=r_b, t_b=t_b, a_px=np.array(a_px), b_px=np.array(b_px),
                world=np.array(world), truth=np.array(truth), rng=rng)


def test_triangulate_recovers_points(scene):
    s = scene
    pts, rms = mv.triangulate([(s["k_a"], np.eye(3), np.zeros(3), s["a_px"][0], 1.0),
                               (s["k_b"], s["r_b"], s["t_b"], s["b_px"][0], 1.0)])
    assert np.allclose(pts, s["truth"][0], atol=1e-6)
    assert rms < 1e-6


def test_solve_pair_recovers_extrinsics_with_noise(scene):
    s = scene
    a = s["a_px"] + s["rng"].normal(0, 0.7, s["a_px"].shape)
    b = s["b_px"] + s["rng"].normal(0, 0.7, s["b_px"].shape)
    out = mv.solve_pair(a, b, s["k_a"], s["k_b"], s["world"])
    assert out["ok"], out
    angle = np.degrees(np.arccos(np.clip((np.trace(out["r"].T @ s["r_b"]) - 1) / 2, -1, 1)))
    assert angle < 3.0  # linear solution, no refinement yet
    assert np.linalg.norm(out["t"] - s["t_b"]) < 0.10  # scale comes from the hand's size
    assert out["rms_px"] < 3


def test_solve_pair_needs_enough_samples(scene):
    s = scene
    out = mv.solve_pair(s["a_px"][:5], s["b_px"][:5], s["k_a"], s["k_b"], s["world"][:5])
    assert not out["ok"] and "samples" in out["error"]


def test_chain_extrinsics_follows_pairs_in_both_directions():
    r_ab, t_ab = g.rot_y(0.5), np.array([1.0, 0.0, 0.2])      # X_a = r_ab X_b + t_ab
    r_cb, t_cb = g.rot_x(-0.3), np.array([0.0, 0.5, 0.1])     # X_c = r_cb X_b + t_cb
    ext = mv.chain_extrinsics({("a", "b"): (r_ab, t_ab), ("c", "b"): (r_cb, t_cb)}, "a")
    assert set(ext) == {"a", "b", "c"}
    x_c = np.array([0.3, -0.2, 1.1])
    x_b = r_cb.T @ (x_c - t_cb)
    x_a = r_ab @ x_b + t_ab
    r_c, t_c = ext["c"]
    assert np.allclose(r_c @ x_c + t_c, x_a)


def test_pair_collector_keeps_only_slow_distinct_samples():
    c = mv.PairCollector()
    order = ["webcam", "phone"]
    valid = np.ones(21, bool)
    segs = mv.hand_segments(np.random.default_rng(1).normal(0, 0.03, (21, 3)))
    base = np.tile([[100.0, 100.0]], (21, 1))

    def both(t, a_px):
        c.add("webcam", "hand", t, a_px, valid, segs, 640, order)
        c.add("phone", "hand", t + 0.01, a_px * 2, valid, segs, 1280, order)

    t = 0.0
    for _ in range(6):             # still, both cameras: one sample, then no new information
        t += 0.04
        both(t, base)
    assert c.counts() == {"webcam+phone": 1}
    for step in range(6):          # fast movement: rejected
        t += 0.04
        both(t, base + 60 * (step + 1))
    assert c.counts() == {"webcam+phone": 1}
    for _ in range(4):             # still again at a new place: one more sample
        t += 0.04
        both(t, base + 400)
    assert c.counts() == {"webcam+phone": 2}


def test_pair_collector_uses_only_landmarks_valid_in_both_views():
    c = mv.PairCollector()
    order = ["webcam", "phone"]
    px = np.arange(66, dtype=float).reshape(33, 2) * 5
    va = np.ones(33, bool)
    vb = np.ones(33, bool)
    vb[20:] = False                # the phone does not see the legs
    segs = [(11, 12, 0.4), (23, 24, 0.3)]
    for i in range(3):
        c.add("webcam", "body", 1.0 + i * 0.1, px, va, segs, 640, order, max_dt_s=0.12)
        c.add("phone", "body", 1.02 + i * 0.1, px * 2, vb, segs, 1280, order, max_dt_s=0.12)
    (a_px, b_px, kept), = c.samples[("webcam", "phone")]
    assert len(a_px) == 20 and len(b_px) == 20
    assert kept == [(11, 12, 0.4)]   # the hip segment needs landmarks the phone lacks
