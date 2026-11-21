# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""The palm filter on synthetic cameras that never expose together."""
import numpy as np

from hub import geometry as g
from hub.handfilter import PalmFilter

CAMS = {"front": (np.eye(3), np.zeros(3)),
        "side": (g.rot_y(-1.3), np.array([1.3, 0.0, 0.5]))}


def _xn(cam, p):
    r, c = CAMS[cam]
    pc = r.T @ (p - c)
    return pc[:2] / pc[2]


def _path(t):
    return np.array([0.3 * np.sin(2.0 * t), -0.1 + 0.2 * np.sin(1.3 * t), 1.0 + 0.25 * np.cos(1.7 * t)])


def _run(views, noise=0.004, seconds=6.0, seed=1):
    """views(t) -> cameras seeing the hand at that moment. Returns errors (m) after settling."""
    rng = np.random.default_rng(seed)
    f = PalmFilter()
    f.start(_path(0.0) + np.array([0.0, 0.0, 0.25]), 0.0)   # 25 cm wrong in depth to begin with
    errs = []
    # front camera at 30 Hz, side camera at 30 Hz but 17 ms later and reported 40 ms late
    events = [(t, "front", t) for t in np.arange(0.0, seconds, 1 / 30)] + \
             [(t + 0.017 + 0.040, "side", t + 0.017) for t in np.arange(0.0, seconds, 1 / 30)]
    for arrive, cam, t in sorted(events):
        if cam in views(t):
            f.see(t, *CAMS[cam], _xn(cam, _path(t)) + rng.normal(0, noise, 2), noise)
            # every view also judges the distance from the hand's size, to some 15 %
            rng_true = float(np.linalg.norm(_path(t) - CAMS[cam][1]))
            f.far(t, CAMS[cam][1], rng_true * (1 + rng.normal(0, 0.15)), 0.15 * rng_true)
        if arrive > 1.0:
            errs.append(float(np.linalg.norm(f.position() - _path(f.t))))
    return np.array(errs)


def test_two_unsynchronised_cameras_give_depth():
    errs = _run(lambda t: ("front", "side"))
    assert np.median(errs) < 0.015 and errs.max() < 0.05


def test_one_camera_dropping_out_does_not_make_the_hand_leap():
    # the side camera loses the hand for 0.6 s out of every 2 s
    errs = _run(lambda t: ("front", "side") if (t % 2.0) > 0.6 else ("front",))
    assert np.median(errs) < 0.03 and errs.max() < 0.20


def test_a_wild_measurement_is_not_believed():
    f = PalmFilter()
    f.start(np.array([0.0, 0.0, 1.0]), 0.0, sigma=0.03)
    assert not f.see(0.01, *CAMS["front"], np.array([0.4, 0.0]), 0.004)   # 40 cm to the side
    assert np.linalg.norm(f.position() - [0.0, 0.0, 1.0]) < 0.02
    assert f.rejects == 1


def test_size_and_skeleton_hold_the_distance_with_one_camera():
    rng = np.random.default_rng(3)
    f = PalmFilter()
    true = np.array([0.1, 0.0, 1.2])
    f.start(true + np.array([0.0, 0.0, 0.3]), 0.0)
    for i in range(1, 60):
        t = i / 30
        f.see(t, *CAMS["front"], _xn("front", true) + rng.normal(0, 0.004, 2), 0.004)
        f.far(t, CAMS["front"][1], float(np.linalg.norm(true)) * (1 + rng.normal(0, 0.1)), 0.15 * 1.2)
        f.near(t, true + rng.normal(0, 0.05, 3), 0.12)
    assert np.linalg.norm(f.position() - true) < 0.06
