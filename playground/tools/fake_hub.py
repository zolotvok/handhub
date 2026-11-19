#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""Fake HandHub for testing the playground without a camera.

Serves ws://127.0.0.1:8798/ws with snapshots in the real hub format (hub/fusion.py snapshot())
at 30 Hz, holding two synthetic hands, and listens on UDP 39579 for the playground's 40-byte
"HHD1" head-pose packets, printing how many valid ones arrive per second.

Run:  python fake_hub.py [--yaw RAD] [--dropout T,DUR] [--calm]
Then: HANDHUB_DIRECT=1 HANDHUB_URL=ws://127.0.0.1:8798/ws HANDHUB_STATUS_PORT=39579 ./run.sh --direct

Body frame: metres, x right, y up, -z forward, origin at the eyes. The desktop camera is at
y = 1.6 and the table top at y = 0.85, i.e. body y = -0.75; the objects sit at z = -0.45.

Motion (period 4 s, t counted from each client connection):
  right hand: vertical circle in the x/y plane centred (0.30, -0.53, -0.42), r = 0.18. It is at the
              bottom (y = -0.71, table-object height) at t = 0 mod 4, directly at the blue cube
              (x = 0.30), closes the grip at 0.2 s and opens it at 2.2 s (carries the cube over the
              right side of the circle and releases it near the top).
  left hand : vertical circle in the y/z plane centred (-0.03, -0.53, -0.67), r = 0.18, its lowest
              point (index fingertip over the middle table button) at t = 1 mod 4; grip closed
              during t = 2..4 mod 4.
--dropout T,DUR drops the right hand (tracked = false, lm = null) for DUR s starting T s after connect.
"""
import argparse
import asyncio
import json
import math
import socket
import struct
import threading
import time

import numpy as np
import websockets

STATUS = struct.Struct("<4sII3f4f")
PERIOD = 4.0
RATE = 30.0


# ------------------------------------------------------------------ hand model

FINGERS = {  # name: (mcp x for a right hand, mcp z, segment lengths)
    "index": (-0.030, -0.090, (0.040, 0.025, 0.022)),
    "middle": (-0.010, -0.095, (0.045, 0.028, 0.024)),
    "ring": (0.010, -0.088, (0.041, 0.026, 0.023)),
    "pinky": (0.028, -0.078, (0.032, 0.020, 0.019)),
}


def hand_local(curl: float, is_right: bool) -> np.ndarray:
    """21 MediaPipe landmarks, hand-local: wrist at the origin, fingers along -z, palm facing -y,
    thumb on -x for a right hand (+x for a left hand). curl in 0..1."""
    sx = 1.0 if is_right else -1.0
    lm = np.zeros((21, 3))
    bend = [math.radians(a) * curl for a in (85.0, 100.0, 65.0)]
    for fi, (name, (mx, mz, seg)) in enumerate(FINGERS.items()):
        p = np.array([mx * sx, 0.0, mz])
        base = 5 + 4 * fi
        lm[base] = p
        cum = 0.0
        for k in range(3):
            cum += bend[k]
            p = p + seg[k] * np.array([0.0, -math.sin(cum), -math.cos(cum)])
            lm[base + 1 + k] = p
    # thumb: CMC, MCP, IP, TIP; open = splayed to the thumb side, curled = across the palm
    side = -sx
    p = np.array([side * 0.020, 0.0, -0.020])
    lm[1] = p
    d_open = np.array([side * math.sin(math.radians(45)), 0.0, -math.cos(math.radians(45))])
    d_closed = np.array([-side * 0.6, -0.5, -0.6])
    for k, ln in enumerate((0.045, 0.032, 0.027)):
        d = d_open * (1 - curl) + d_closed * curl
        d = d / np.linalg.norm(d)
        p = p + ln * d
        lm[2 + k] = p
    return lm


def controller_pose(pts: np.ndarray):
    """Same convention as hub/geometry.controller_pose: z = -(wrist->middle MCP), y = thumb side."""
    f = pts[9] - pts[0]
    f /= np.linalg.norm(f)
    a = pts[5] - pts[17]
    a = a - f * float(a @ f)
    a /= np.linalg.norm(a)
    z, y = -f, a
    x = np.cross(y, z)
    r = np.column_stack([x, y, z])
    pos = pts[[0, 5, 9, 13, 17]].mean(axis=0)
    return pos, r


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> (w, x, y, z)."""
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def hand_snapshot(is_right: bool, palm_target: np.ndarray, closed: bool, curl: float, tracked: bool,
                  confidence: float) -> dict:
    if not tracked:
        return {"tracked": False, "buttons": 0, "pos": [0, 0, 0], "rot": [1, 0, 0, 0], "trigger": 0.0,
                "grip": 0.0, "curl": [0.0] * 5, "pinch": 1.0, "confidence": 0.0, "reproj_px": 0.0,
                "height_px": 0, "lm": None}
    local = hand_local(curl, is_right)
    palm_local = local[[0, 5, 9, 13, 17]].mean(axis=0)
    lm = local - palm_local + palm_target          # hand axes == body axes (fingers along -z)
    pos, r = controller_pose(lm)
    buttons = 2 if closed else 0
    return {
        "tracked": True, "buttons": buttons,
        "pos": np.round(pos, 4).tolist(), "rot": np.round(quat_from_matrix(r), 4).tolist(),
        "trigger": round(curl, 3), "grip": round(curl, 3),
        "curl": [round(curl, 3)] * 5, "pinch": 1.0, "confidence": confidence,
        "reproj_px": 2.0, "height_px": 300, "lm": np.round(lm, 4).tolist(),
    }


def smooth_curl(t: float, closed_from: float, closed_to: float, ramp: float = 0.25) -> float:
    """0 when open, 1 when closed; ramps over `ramp` seconds at each edge (phase wraps at PERIOD)."""
    for tt in (t % PERIOD, t % PERIOD + PERIOD):
        if closed_from <= tt < closed_to:
            return min(1.0, (tt - closed_from) / ramp)
        if closed_to <= tt < closed_to + ramp:
            return 1.0 - (tt - closed_to) / ramp
    return 0.0


def make_snapshot(t: float, yaw: float, dropout, calm: bool = False) -> dict:
    w = 2 * math.pi / PERIOD
    if calm:   # both hands 35 cm in front of and 35 cm below the eyes, small horizontal circles
        right = np.array([0.20 + 0.05 * math.sin(w * t), -0.35, -0.35 + 0.05 * math.cos(w * t)])
        left = np.array([-0.20 - 0.05 * math.sin(w * t), -0.35, -0.35 + 0.05 * math.cos(w * t)])
    else:
        # right hand: bottom of the circle at t = 0 mod 4
        right = np.array([0.30 + 0.18 * math.sin(w * t), -0.53 - 0.18 * math.cos(w * t), -0.42])
        # left hand: lowest point at t = 1 mod 4
        left = np.array([-0.03, -0.53 - 0.18 * math.cos(w * (t - 1.0)), -0.67 + 0.18 * math.sin(w * (t - 1.0))])
    r_closed = 0.2 <= (t % PERIOD) < 2.2        # grip bit follows the closed phase
    l_closed = 2.0 <= (t % PERIOD) < 4.0
    r_curl = smooth_curl(t, 0.2, 2.2)
    l_curl = smooth_curl(t, 2.0, 4.0)
    r_tracked = not (dropout and dropout[0] <= t < dropout[0] + dropout[1])
    hands = [
        hand_snapshot(False, left, l_closed, l_curl, True, 0.90),
        hand_snapshot(True, right, r_closed, r_curl, r_tracked, 0.25 + 0.70 * math.cos(w * t * 0.5) ** 2),
    ]
    return {
        "camera": {"online": True, "fps": 30.0, "error": None, "latency_ms": 20.0},
        "camera_pos": [0.0, -0.35, -0.5],
        "hands": hands,
        "driver": {"connected": True, "rx_error": None},
        "hmd": {"valid": True, "pos": [0.0, 1.6, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]},
        "yaw_offset": yaw,
    }


# ------------------------------------------------------------------ servers

async def handler(ws, args):
    t0 = time.monotonic()
    print("[fake_hub] client connected (t=0 now)", flush=True)
    nxt = t0
    try:
        while True:
            t = time.monotonic() - t0
            await ws.send(json.dumps(make_snapshot(t, args.yaw, args.dropout, args.calm)))
            nxt += 1.0 / RATE
            await asyncio.sleep(max(0.0, nxt - time.monotonic()))
    except websockets.ConnectionClosed:
        print("[fake_hub] client disconnected", flush=True)


def udp_listener(port: int, stats: dict):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.settimeout(0.2)
    while True:
        try:
            data, _ = sock.recvfrom(256)
        except socket.timeout:
            continue
        if len(data) == STATUS.size and data[:4] == b"HHD1":
            v = STATUS.unpack(data)
            stats["valid"] += 1
            stats["last"] = v
        else:
            stats["bad"] += 1


def udp_reporter(stats: dict):
    while True:
        time.sleep(1.0)
        v = stats["last"]
        pose = "" if v is None else " last: seq=%d valid=%d pos=(%.3f, %.3f, %.3f) rot(wxyz)=(%.3f, %.3f, %.3f, %.3f)" % (
            v[1], v[2], *v[3:6], *v[6:10])
        print(f"[fake_hub] status packets/s: {stats['valid']} (bad: {stats['bad']}){pose}", flush=True)
        stats["valid"] = 0
        stats["bad"] = 0


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8798)
    ap.add_argument("--status-port", type=int, default=39579)
    ap.add_argument("--yaw", type=float, default=0.0, help="yaw_offset to publish (radians)")
    ap.add_argument("--dropout", default=None, help="T,DUR: right hand lost for DUR s, T s after connect")
    ap.add_argument("--calm", action="store_true",
                    help="hands hover 35 cm in front of / 35 cm below the eyes (no reach to the table)")
    args = ap.parse_args()
    args.dropout = tuple(float(x) for x in args.dropout.split(",")) if args.dropout else None

    stats = {"valid": 0, "bad": 0, "last": None}
    threading.Thread(target=udp_listener, args=(args.status_port, stats), daemon=True).start()
    threading.Thread(target=udp_reporter, args=(stats,), daemon=True).start()

    async def h(ws):
        await handler(ws, args)

    async with websockets.serve(h, "127.0.0.1", args.port):
        print(f"[fake_hub] ws://127.0.0.1:{args.port}/ws  udp:{args.status_port}  yaw={args.yaw}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
