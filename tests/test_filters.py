# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
import numpy as np

from hub.filters import OneEuro, OneEuroQuat


def test_first_sample_passes_through():
    f = OneEuro()
    x = np.array([0.1, -2.0, 3.0])
    assert np.allclose(f(x, 0.0), x)


def test_constant_input_stays_constant():
    f = OneEuro()
    out = None
    for i in range(120):
        out = f(np.array([1.0, 2.0, 3.0]), i / 60)
    assert np.allclose(out, [1.0, 2.0, 3.0])


def test_noisy_constant_input_is_smoothed():
    rng = np.random.default_rng(0)
    f = OneEuro()
    raw = 0.5 + rng.normal(0, 0.05, 120)
    out = [float(f(x, i / 60)) for i, x in enumerate(raw)]
    assert np.std(out[30:]) < np.std(raw[30:])


def test_step_converges():
    f = OneEuro()
    f(0.0, 0.0)
    out = None
    for i in range(1, 181):  # 3 s at 60 Hz
        out = f(1.0, i / 60)
    assert abs(float(out) - 1.0) < 1e-2


def test_non_increasing_timestamp_resets():
    f = OneEuro()
    f(1.0, 1.0)
    assert np.allclose(f(2.0, 1.0), 2.0)  # same timestamp: reset, no division by zero
    assert np.allclose(f(3.0, 0.5), 3.0)  # going backwards: reset again
    assert np.all(np.isfinite(f.dx))


def test_quat_output_is_unit_length():
    rng = np.random.default_rng(2)
    f = OneEuroQuat()
    for i in range(100):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        out = f(q, i / 60)
        assert np.isclose(np.linalg.norm(out), 1.0)


def test_quat_sign_flip_does_not_jump():
    q = np.array([0.9, 0.1, 0.3, 0.2])
    q /= np.linalg.norm(q)
    f = OneEuroQuat()
    f(q, 0.0)
    out = f(-q, 1 / 60)
    assert np.allclose(out, q, atol=1e-6)
