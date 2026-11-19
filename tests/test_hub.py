# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
import json
import socket

import numpy as np
import pytest

from hub import geometry as g
from hub.config import Config
from hub.fusion import REST_TARGET, Hub


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def make_hub(tmp_path):
    hubs = []

    def _make(swap_hands=False):
        config = Config(tmp_path / "c.json")
        config.update({"driver": {"port": free_port(), "status_port": free_port()}})
        if swap_hands:
            config.update({"camera": {"swap_hands": True}})
        hub = Hub(config=config, start_camera=False)
        hubs.append(hub)
        return hub

    yield _make
    for hub in hubs:
        hub.link.close()


def synthetic_detection():
    """Non-planar hand in front of the camera; returns (image_px, world)."""
    rng = np.random.default_rng(0)
    world = rng.uniform(-0.04, 0.04, (21, 3))
    k = g.intrinsics_from_hfov(640, 480, 70)
    r = g.rot_z(0.1) @ g.rot_x(-0.2) @ g.rot_y(0.3)
    cam = world @ r.T + np.array([0.02, -0.01, 0.5])
    image = (k @ (cam / cam[:, 2:3]).T).T[:, :2]
    return image, world


def test_process_frame_puts_right_label_into_right_hand(make_hub):
    hub = make_hub()
    image, world = synthetic_detection()
    hub.process_frame(1_000_000_000, (640, 480), [("Right", 0.95, image, world)], hub.config.get())

    assert not hub.hands[0].state.tracked
    right = hub.hands[1]
    assert right.state.tracked
    assert np.all(np.isfinite(right.state.pos))
    assert right.state.lm.shape == (21, 3)


def test_swap_hands_puts_label_in_left_hand(make_hub):
    hub = make_hub(swap_hands=True)
    image, world = synthetic_detection()
    hub.process_frame(1_000_000_000, (640, 480), [("Right", 0.95, image, world)], hub.config.get())
    assert hub.hands[0].state.tracked
    assert not hub.hands[1].state.tracked


def test_snapshot_is_json_serializable(make_hub):
    hub = make_hub()
    image, world = synthetic_detection()
    hub.process_frame(1_000_000_000, (640, 480), [("Right", 0.95, image, world)], hub.config.get())
    snap = hub.snapshot()
    json.dumps(snap)
    assert {"camera", "hands", "driver", "hmd"} <= snap.keys()
    assert len(snap["hands"]) == 2


def test_calibrate_rest_moves_hand_to_rest_target(make_hub):
    hub = make_hub()
    assert hub.calibrate_rest()["ok"] is False

    image, world = synthetic_detection()
    cfg = hub.config.get()
    hub.process_frame(1_000_000_000, (640, 480), [("Right", 0.95, image, world)], cfg)
    result = hub.calibrate_rest()
    assert result["ok"] is True

    hub.process_frame(2_000_000_000, (640, 480), [("Right", 0.95, image, world)], hub.config.get())
    assert np.allclose(hub.hands[1].raw_pos, [0.0, -0.35, -0.35], atol=1e-3)
    assert np.allclose(REST_TARGET, [0.0, -0.35, -0.35])


def test_calibrate_align_without_hmd_fails(make_hub):
    hub = make_hub()
    result = hub.calibrate_align()
    assert result["ok"] is False


def test_calibrate_head_puts_thumb_tips_at_temples(tmp_path):
    """After the head calibration, the mean thumb tip sits at the body origin (slightly back)."""
    import socket

    import numpy as np

    from hub import geometry as g
    from hub.config import Config
    from hub.fusion import HEAD_HALF_WIDTH, TEMPLE_Z, Hub

    def free_port():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    cfg = Config(tmp_path / "c.json")
    cfg.update({"driver": {"port": free_port(), "status_port": free_port()}})
    hub = Hub(config=cfg, start_camera=False)
    try:
        assert hub.calibrate_head()["ok"] is False

        rng = np.random.default_rng(3)
        world = rng.uniform(-0.08, 0.08, size=(21, 3))
        k = g.intrinsics_from_hfov(640, 480, 70)

        def detection(label, tx):
            pts = world @ g.rot_x(0.2).T + np.array([tx, 0.05, 0.7])
            px = pts @ k.T
            return (label, 0.95, px[:, :2] / px[:, 2:3], world)

        both = [detection("Right", -0.1), detection("Left", 0.1)]
        hub.process_frame(1_000_000_000, (640, 480), both, hub.config.get())
        out = hub.calibrate_head()
        assert out["ok"] and out["hands_used"] == 2
        hub.process_frame(2_000_000_000, (640, 480), both, hub.config.get())
        mean_tip = np.mean([h.lm_body[g.THUMB_TIP] for h in hub.hands], axis=0)
        assert np.allclose(mean_tip, [0.0, 0.0, TEMPLE_Z], atol=1e-3)

        # One hand only: its thumb tip goes to that side's temple.
        for h in hub.hands:
            h.lose()
        hub.process_frame(3_000_000_000, (640, 480), [detection("Right", 0.1)], hub.config.get())
        out = hub.calibrate_head()
        assert out["ok"] and out["hands_used"] == 1
        hub.process_frame(4_000_000_000, (640, 480), [detection("Right", 0.1)], hub.config.get())
        assert np.allclose(hub.hands[1].lm_body[g.THUMB_TIP], [HEAD_HALF_WIDTH, 0.0, TEMPLE_Z], atol=1e-3)
    finally:
        hub.link.close()


def test_oversized_hand_detection_is_rejected(make_hub):
    """A "hand" as long as the torso (the model firing on a shirt) is not taken as a hand."""
    hub = make_hub()
    cfg = hub.config.get()
    body_px = np.zeros((33, 2))
    body_vis = np.zeros(33)
    for i, xy in {11: (260, 150), 12: (380, 150), 23: (270, 330), 24: (370, 330), 13: (240, 230),
                  14: (400, 230), 15: (230, 300), 16: (410, 300)}.items():
        body_px[i], body_vis[i] = xy, 0.95
    hub.body_views["webcam"] = (1.0, g.intrinsics_from_hfov(640, 480, 70), body_px, body_vis)

    def hand(wrist, length):
        pts = np.tile(np.array(wrist, float), (21, 1))
        pts[9] = pts[0] + [0, -0.5 * length]
        pts[12] = pts[0] + [0, -length]
        return ("Right", 0.9, pts, np.zeros((21, 3)))

    real, fake = hand((410, 300), 60), hand((400, 310), 200)   # torso is 180 px long
    out = hub._which_hands("webcam", 1.1, (640, 480), [real, fake], cfg["camera"])
    assert out[0] is True       # at the right wrist, hand-sized
    assert out[1] is None       # same place, but as long as the torso
