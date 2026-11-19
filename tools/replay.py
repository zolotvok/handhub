#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""Re-run the hub's fusion on a raw recording (POST /api/record/raw) and score it.

    .venv/bin/python tools/replay.py ~/.local/share/handhub/recordings/raw-HHMMSS.pkl [key=value ...]

key=value pairs override constants in hub.fusion (e.g. VIEW_MAX_AGE_S=0.1) so settings
can be compared without anyone standing in front of the cameras.

There is no ground truth, so the score is self-consistency:
  * "miss": how far the fused hand, projected into a camera, lands from the nearest hand
    that camera actually detected in that frame, as a share of the image width. A correct
    hand lands on a detection in every camera that sees it. Counted only for cameras that
    saw some hand in that frame.
  * steps: how far the wrist moves between consecutive updates; big steps are leaps.
  * modes: how each hand was computed.
"""
from __future__ import annotations

import json
import pickle
import socket
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hub import fusion, geometry as g  # noqa: E402
from hub.cameras import runner  # noqa: E402
from hub.config import Config  # noqa: E402


SIDE_STATS: dict = {}
DETAIL: list = []  # (time, hand, camera, mode, views used, miss, detections in frame)


class FakeTime:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def monotonic_ns(self):
        return int(self.now * 1e9)

    def strftime(self, *a):
        import time
        return time.strftime(*a)


def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def replay(path: str, overrides: dict, config_patch: dict | None = None) -> dict:
    data = pickle.load(open(path, "rb"))
    for k, v in overrides.items():
        setattr(fusion, k, type(getattr(fusion, k))(v))
    clock = FakeTime()
    fusion.time = clock
    runner.time = clock

    cfg_data = data["config"]
    cfg_data["driver"]["port"], cfg_data["driver"]["status_port"] = free_port(), free_port()
    tmp = Path(tempfile.mkdtemp()) / "config.json"
    tmp.write_text(json.dumps(cfg_data))
    config = Config(tmp)
    if config_patch:
        config.update(config_patch)
    hub = fusion.Hub(config=config, start_camera=False)
    cfg = config.get()
    cam_cfgs = hub.camera_configs(cfg)

    miss = {i: [] for i in range(2)}
    side = {i: [] for i in range(2)}
    steps = {i: [] for i in range(2)}
    modes = {i: {} for i in range(2)}
    last = {i: None for i in range(2)}
    tracked_time = {i: 0.0 for i in range(2)}
    t_prev = None
    events = data["events"]
    for now, name, msg in events:
        clock.now = now
        if name == "_hmd":
            hub.link.hmd_history.append(msg)
            hub.link.hmd_valid, hub.link.last_status_t = True, now
            continue
        cam = hub.cams[name]
        frame = cam.on_message(msg)
        if frame is None:
            continue
        if cam.last_head is not None:
            hub.process_head(name, frame[1], cam.last_head, cfg)
        hub.process_frame(*frame, cfg, cam=name)
        hub._expire(cfg)

        dt = 0.0 if t_prev is None else min(now - t_prev, 0.2)
        t_prev = now
        ext = hub.extrinsics(cfg)
        dets = msg["hands"]
        size = msg["size"]
        k = g.intrinsics_from_hfov(size[0], size[1], cam_cfgs[name]["hfov_deg"])
        for i, h in enumerate(hub.hands):
            if not h.state.tracked:
                modes[i]["lost"] = modes[i].get("lost", 0.0) + dt
                last[i] = None
                continue
            tracked_time[i] += dt
            m = (h.mode or "?").replace("+snapped", "")
            modes[i][m] = modes[i].get(m, 0.0) + dt
            if h.last_t == frame[0] / 1e9:  # this event updated the hand
                if last[i] is not None:
                    steps[i].append(float(np.linalg.norm(h.lm_ref[0] - last[i])))
                last[i] = h.lm_ref[0].copy()
            # which side: is this hand nearer the skeleton's other wrist than its own?
            own, other = (16, 15) if i == 1 else (15, 16)
            if hub.body_ok and now - hub.body_seen[own] < 0.3 and now - hub.body_seen[other] < 0.3:
                w_own, w_other = hub.r_room.T @ hub.body_pts[own], hub.r_room.T @ hub.body_pts[other]
                if float(np.linalg.norm(w_own - w_other)) > 0.25:  # wrists apart: the question is meaningful
                    side[i].append(float(np.linalg.norm(h.lm_ref[0] - w_other)) < float(np.linalg.norm(h.lm_ref[0] - w_own)))
            if dets and name in ext and now - h.last_seen < 0.08:
                r, t = ext[name]
                c = r.T @ (h.lm_ref[0] - t)
                if c[2] > 0.05:
                    px = c[:2] / c[2] * k[0, 0] + k[:2, 2]
                    d = min(float(np.linalg.norm(np.asarray(det[2])[0] - px)) for det in dets) / size[0]
                    miss[i].append(d)
                    DETAIL.append((now, i, name, h.mode, tuple(h.views_used), d, len(dets)))
    hub.link.close()
    global SIDE_STATS
    SIDE_STATS = dict(hub.side_stats)

    out = {}
    total = events[-1][0] - events[0][0]
    for i, nm in enumerate(("left", "right")):
        s = np.array(steps[i]) if steps[i] else np.zeros(1)
        ms = np.array(miss[i]) if miss[i] else np.zeros(1)
        out[nm] = {
            "tracked": round(tracked_time[i] / total, 2),
            "modes": {k: round(v / total, 2) for k, v in sorted(modes[i].items(), key=lambda x: -x[1])},
            "miss_median": round(float(np.median(ms)), 3), "miss_p90": round(float(np.percentile(ms, 90)), 3),
            "miss_over_10pct": round(float(np.mean(ms > 0.10)), 3),
            "wrong_side": round(float(np.mean(side[i])), 3) if side[i] else None, "side_samples": len(side[i]),
            "steps_over_10cm": int((s > 0.10).sum()), "steps_over_20cm": int((s > 0.20).sum()), "updates": len(s),
        }
    return out


def main() -> None:
    path = sys.argv[1]
    overrides = dict(a.split("=", 1) for a in sys.argv[2:])
    for nm, r in replay(path, overrides).items():
        print(nm, json.dumps(r))


if __name__ == "__main__":
    main()
