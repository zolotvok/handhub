#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""Record the hub's camera previews together with its 3D state, and render a video
that shows both, with the 3D model projected back onto every camera image.

    .venv/bin/python tools/record_session.py [seconds] [output_dir] [start_delay_seconds]

Output: frames + state in the directory, `composite.mp4`, and `sheet_*.jpg` stills.
In the video the hub's own overlay (drawn by the camera worker) is green/orange;
the 3D model projected back into each camera is magenta (body) and cyan/yellow
(left/right hand). Where magenta does not sit on the person, the 3D model is wrong
for that camera.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import websockets

HUB = "127.0.0.1:8765"
CAMS = ["webcam", "phone", "pico"]
BODY_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
              (23, 25), (25, 27), (24, 26), (26, 28)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
MAGENTA, CYAN, YELLOW, WHITE = (255, 0, 255), (255, 255, 0), (0, 255, 255), (255, 255, 255)


def get_json(path: str):
    return json.load(urllib.request.urlopen(f"http://{HUB}{path}", timeout=5))


def record(seconds: float, out: Path) -> None:
    stop = time.monotonic() + seconds
    frames = {c: [] for c in CAMS}  # (t, jpeg)

    def grab(cam: str) -> None:
        try:
            r = urllib.request.urlopen(f"http://{HUB}/stream/{cam}.mjpg", timeout=5)
            buf = b""
            while time.monotonic() < stop:
                buf += r.read(16384)
                while True:
                    a = buf.find(b"\xff\xd8")
                    e = buf.find(b"\xff\xd9", a + 2) if a >= 0 else -1
                    if a < 0 or e < 0:
                        break
                    frames[cam].append((time.monotonic(), buf[a:e + 2]))
                    buf = buf[e + 2:]
        except OSError as exc:
            print(f"[{cam}] stream ended: {exc}")

    threads = [threading.Thread(target=grab, args=(c,), daemon=True) for c in CAMS]
    for t in threads:
        t.start()

    snaps, views = [], []

    async def ws() -> None:
        async with websockets.connect(f"ws://{HUB}/ws") as w:
            last_view = 0.0
            while time.monotonic() < stop:
                snaps.append((time.monotonic(), json.loads(await w.recv())))
                if time.monotonic() - last_view > 0.2:
                    last_view = time.monotonic()
                    views.append((last_view, await asyncio.to_thread(get_json, "/api/debug/views")))

    asyncio.run(ws())
    for t in threads:
        t.join(timeout=3)

    out.mkdir(parents=True, exist_ok=True)
    # wall clock <-> the monotonic clock used for the frames, to line up other recordings
    (out / "clock.json").write_text(json.dumps({"unix": time.time(), "monotonic": time.monotonic()}))
    for cam, lst in frames.items():
        d = out / cam
        d.mkdir(exist_ok=True)
        for i, (t, jpg) in enumerate(lst):
            (d / f"{i:05d}.jpg").write_bytes(jpg)
        (out / f"{cam}_times.json").write_text(json.dumps([t for t, _ in lst]))
    (out / "snapshots.json").write_text(json.dumps(snaps))
    (out / "views.json").write_text(json.dumps(views))
    (out / "config.json").write_text(json.dumps(get_json("/api/config")))
    print("recorded:", {c: len(v) for c, v in frames.items()}, "snapshots", len(snaps), "views", len(views))


def nearest(times: list, t: float) -> int:
    return int(np.argmin(np.abs(np.asarray(times) - t))) if times else -1


def rot_base(cfg: dict) -> np.ndarray:
    import math
    c = cfg["camera"]

    def rx(a):
        return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])

    def ry(a):
        return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])

    def rz(a):
        return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])

    r = ry(math.radians(c["yaw_deg"])) @ rx(math.radians(c["pitch_deg"])) @ rz(math.radians(c["roll_deg"]))
    return r @ np.diag([-1.0, -1.0, 1.0])


def project(p_room, r_room, ext, k, width, mirrored):
    """Room point -> pixel in a camera's preview image (None if behind the camera)."""
    p_ref = r_room.T @ np.asarray(p_room, float)
    cam = np.asarray(ext["r"]).T @ (p_ref - np.asarray(ext["t"]))
    if cam[2] < 0.05:
        return None
    x = cam[0] / cam[2] * k[0][0] + k[0][2]
    y = cam[1] / cam[2] * k[1][1] + k[1][2]
    if mirrored:
        x = width - x
    if not (-2000 < x < 4000 and -2000 < y < 4000):
        return None
    return int(round(x)), int(round(y))


def draw_model_on(img, snap, view, cam, cfg, r_room):
    ext = view["extrinsics"].get(cam)
    kinfo = view["body"].get(cam)
    if ext is None or kinfo is None:
        cv2.putText(img, "not calibrated / no data", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        return
    k = kinfo["k"]
    cam_cfg = cfg["camera"] if cam == "webcam" else cfg["extra_cameras"][cam]
    mirrored = bool(cam_cfg["mirror_preview"])
    w = img.shape[1]
    pts = snap["body"]["points"]
    px = [project(p, r_room, ext, k, w, mirrored) if p else None for p in pts]
    for a, b in BODY_EDGES:
        if px[a] and px[b]:
            cv2.line(img, px[a], px[b], MAGENTA, 2, cv2.LINE_AA)
    for i in (0, 7, 8):
        if px[i]:
            cv2.circle(img, px[i], 6, MAGENTA, 2, cv2.LINE_AA)
    for i, col in ((0, CYAN), (1, YELLOW)):
        h = snap["hands"][i]
        if not h.get("lm_room"):
            continue
        hp = [project(p, r_room, ext, k, w, mirrored) for p in h["lm_room"]]
        for a, b in HAND_EDGES:
            if hp[a] and hp[b]:
                cv2.line(img, hp[a], hp[b], col, 2, cv2.LINE_AA)
        if hp[0]:
            cv2.putText(img, "LR"[i] + ":" + (h.get("mode") or "?"), (hp[0][0] + 6, hp[0][1] + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)


def draw_3d(snap, size=(480, 480), axes=(0, 1), title=""):
    """Orthographic view of the model. axes: which room axes go right and up."""
    img = np.full((size[1], size[0], 3), 24, np.uint8)
    scale, cx, cy = 150.0, size[0] / 2, size[1] / 2
    centre = np.array([0.0, -0.3, 1.2])
    head = snap.get("head") or {}
    if head.get("tracked"):
        centre = np.array(head["pos_room"]) + np.array([0.0, -0.6, 0.0])

    def to_px(p):
        d = np.asarray(p, float) - centre
        return int(cx + d[axes[0]] * scale), int(cy - d[axes[1]] * scale)

    for name, c in snap["cameras"].items():
        if c.get("pos_room"):
            a = to_px(c["pos_room"])
            b = to_px(np.array(c["pos_room"]) + 0.35 * np.array(c["dir_room"]))
            cv2.circle(img, a, 6, (160, 160, 160), -1)
            cv2.line(img, a, b, (160, 160, 160), 2)
            cv2.putText(img, name, (a[0] + 6, a[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    pts = snap["body"]["points"]
    for a, b in BODY_EDGES:
        if pts[a] and pts[b]:
            cv2.line(img, to_px(pts[a]), to_px(pts[b]), MAGENTA, 2, cv2.LINE_AA)
    if pts[7] and pts[8]:
        cv2.circle(img, to_px((np.array(pts[7]) + np.array(pts[8])) / 2), 14, MAGENTA, 2, cv2.LINE_AA)
    for i, col in ((0, CYAN), (1, YELLOW)):
        lm = snap["hands"][i].get("lm_room")
        if lm:
            for a, b in HAND_EDGES:
                cv2.line(img, to_px(lm[a]), to_px(lm[b]), col, 2, cv2.LINE_AA)
            cv2.putText(img, "LR"[i], to_px(lm[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    return img


def render(out: Path, fps: float = 15.0) -> None:
    snaps = json.loads((out / "snapshots.json").read_text())
    views = json.loads((out / "views.json").read_text())
    cfg = json.loads((out / "config.json").read_text())
    times = {c: json.loads((out / f"{c}_times.json").read_text()) for c in CAMS}
    r_room = rot_base(cfg)
    shots = []  # (monotonic time, path) of the game's own screenshots, if it saved any
    if (out / "clock.json").exists() and (out / "game").is_dir():
        clock = json.loads((out / "clock.json").read_text())
        for f in sorted((out / "game").glob("*.jpg")):
            try:
                shots.append((float(f.stem) - clock["unix"] + clock["monotonic"], str(f)))
            except ValueError:
                pass
    snap_t = [t for t, _ in snaps]
    view_t = [t for t, _ in views]
    t0, t1 = snap_t[0], snap_t[-1]
    row_h = 480
    writer = None
    stills = []
    n = int((t1 - t0) * fps)
    for f in range(n):
        t = t0 + f / fps
        snap = snaps[nearest(snap_t, t)][1]
        view = views[nearest(view_t, t)][1]
        tiles = []
        for cam in CAMS:
            i = nearest(times[cam], t)
            img = cv2.imread(str(out / cam / f"{i:05d}.jpg")) if i >= 0 else None
            if img is None:
                img = np.zeros((row_h, 640, 3), np.uint8)
            draw_model_on(img, snap, view, cam, cfg, r_room)
            c = snap["cameras"][cam]
            label = f"{cam}  agree {c.get('agreement_px')} px  sees L{int(c['sees'][0])} R{int(c['sees'][1])}"
            cv2.putText(img, label, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)
            tiles.append(cv2.resize(img, (int(img.shape[1] * row_h / img.shape[0]), row_h)))
        top = np.hstack(tiles)
        b = snap["body"]
        info = f"t={t - t0:5.2f}s  body: {b['source']} {'+'.join(b['views'])}"
        for i, name in enumerate(("L", "R")):
            h = snap["hands"][i]
            info += f"   {name}: {h['mode']} {'+'.join(h.get('views', []))}" if h["tracked"] else f"   {name}: -"
        three = [draw_3d(snap, axes=(0, 1), title="front (x right, y up)"),
                 draw_3d(snap, axes=(0, 2), title="top (x right, z away from webcam)")]
        shot = None
        if shots:
            j = nearest([x[0] for x in shots], t)
            if abs(shots[j][0] - t) < 1.0:
                shot = cv2.imread(shots[j][1])
        if shot is not None:  # what the player saw at this moment
            shot = cv2.resize(shot, (int(shot.shape[1] * 480 / shot.shape[0]), 480))
            cv2.putText(shot, "game view", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
            three.append(shot)
        else:
            three.append(draw_3d(snap, axes=(2, 1), title="side (z right, y up)"))
        bottom = np.hstack(three)
        width = max(top.shape[1], bottom.shape[1])

        def pad(im):
            return cv2.copyMakeBorder(im, 0, 0, 0, width - im.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))

        bar = np.zeros((34, width, 3), np.uint8)
        cv2.putText(bar, info, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        frame = np.vstack([pad(top), bar, pad(bottom)])
        if writer is None:
            writer = cv2.VideoWriter(str(out / "composite.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        if f % max(n // 8, 1) == 0:
            stills.append(frame)
    if writer is not None:
        writer.release()
    for i, s in enumerate(stills):
        cv2.imwrite(str(out / f"sheet_{i:02d}.jpg"), s, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("rendered", n, "frames ->", out / "composite.mp4", "and", len(stills), "stills")


def main() -> None:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    data = Path(os.environ.get("HANDHUB_DATA", Path.home() / ".local/share/handhub"))
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else data / "recordings" / time.strftime("rec-%Y%m%d-%H%M%S")
    if not (out / "snapshots.json").exists():
        delay = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
        for left in range(int(delay), 0, -1):  # time to walk from the laptop into position
            print(f"recording starts in {left} s", flush=True)
            time.sleep(1)
        record(seconds, out)
    render(out)


if __name__ == "__main__":
    main()
