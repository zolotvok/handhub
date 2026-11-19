# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
import copy

import numpy as np
import pytest

from hub.config import DEFAULTS
from hub.gestures import BTN_A, BTN_B, BTN_GRIP, BTN_TRIGGER, GestureDetector

CFG = copy.deepcopy(DEFAULTS["gestures"])
OPEN_THUMB_TIP = np.array([0.07, -0.05, 0.0])


def make_hand(fist=False, thumb_tip=OPEN_THUMB_TIP):
    """MediaPipe-like world landmarks (metres). Wrist at origin, fingers point along -y."""
    w = np.zeros((21, 3))
    w[1:4] = np.linspace(np.zeros(3), thumb_tip, 5)[1:4]
    w[4] = thumb_tip
    for mcp, x in zip((5, 9, 13, 17), (0.03, 0.01, -0.01, -0.03)):
        base = np.array([x, -0.09, 0.0])
        tip = np.array([0.5 * x, -0.04, 0.0]) if fist else np.array([x, -0.18, 0.0])
        for j in range(4):
            w[mcp + j] = base + (tip - base) * j / 3
    return w


def hand_size(w):
    return float(np.linalg.norm(w[9] - w[0]))


def thumb_at_pinch(w, pinch):
    """Place the thumb tip at `pinch` hand-sizes from the index tip (along +x)."""
    w = w.copy()
    w[4] = w[8] + np.array([pinch * hand_size(w), 0.0, 0.0])
    return w


def detector_out(w, cfg=CFG, prev=None, confident=True):
    d = GestureDetector()
    if prev is not None:
        d.update(prev, cfg)
    return d.update(w, cfg, confident)


def test_open_hand_has_no_clicks():
    out = detector_out(make_hand())
    assert out.trigger == pytest.approx(0.0, abs=1e-6)
    assert out.grip == pytest.approx(0.0, abs=1e-6)
    assert out.buttons == 0


def test_pinch_triggers_in_gesture_profile():
    out = detector_out(thumb_at_pinch(make_hand(), 0.0))
    assert out.trigger == pytest.approx(1.0)
    assert out.buttons & BTN_TRIGGER
    assert not out.buttons & BTN_GRIP


def test_fist_sets_grip_not_ab():
    out = detector_out(make_hand(fist=True))
    assert out.grip > 0.9
    assert out.buttons & BTN_GRIP
    assert not out.buttons & (BTN_A | BTN_B)


def test_trigger_hysteresis():
    cfg = CFG
    click = thumb_at_pinch(make_hand(), 0.0)
    hold = thumb_at_pinch(make_hand(), 0.22)  # trigger ~0.8: between release (0.7) and click (0.9)
    clear = thumb_at_pinch(make_hand(), 0.30)  # trigger ~0.57: below release

    d = GestureDetector()
    d.update(click, cfg)
    assert d.update(hold, cfg).buttons & BTN_TRIGGER
    assert not d.update(clear, cfg).buttons & BTN_TRIGGER

    assert not detector_out(hold).buttons & BTN_TRIGGER  # fresh detector: no hysteresis to hold


def test_thumb_on_middle_tip_sets_a_in_gesture_profile_only():
    w = make_hand(thumb_tip=make_hand()[12])
    assert detector_out(w).buttons & BTN_A

    finger = copy.deepcopy(CFG)
    finger["profile"] = "finger"
    assert not detector_out(w, finger).buttons & BTN_A


def test_finger_profile_trigger_is_index_curl():
    finger = copy.deepcopy(CFG)
    finger["profile"] = "finger"

    half = make_hand()
    half[5 + 3] = [0.0, -0.1329, 0.0]  # index tip, about 1.4x the MCP distance from the wrist
    out = detector_out(half, finger)
    assert out.trigger == pytest.approx(out.curl[1])
    assert out.trigger == pytest.approx(0.5, abs=0.02)


def test_finger_profile_pinch_with_extended_index_does_not_click():
    finger = copy.deepcopy(CFG)
    finger["profile"] = "finger"
    out = detector_out(thumb_at_pinch(make_hand(), 0.0), finger)
    assert not out.buttons & BTN_TRIGGER


def test_unconfident_frame_holds_previous_output():
    d = GestureDetector()
    before = d.update(make_hand(), CFG)
    grip_before = before.grip
    after = d.update(make_hand(fist=True), CFG, confident=False)
    assert after is before
    assert after.grip == grip_before
