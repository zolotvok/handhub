# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""V4L2 camera worker process: capture -> hand landmarks -> results over a pipe.

One process per camera. The worker only detects; all geometry runs in the hub.
"""
from __future__ import annotations

import glob
import os
import re
import threading
import time
from collections import deque

import numpy as np

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


# MediaPipe pose: shoulders, arms, torso, legs (the face is drawn as points).
BODY_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
              (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32)]


def list_devices() -> list[dict]:
    """V4L2 capture devices. Each camera usually exposes two nodes; the second is metadata."""
    out = []
    for path in sorted(glob.glob("/sys/class/video4linux/video*"), key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)))):
        idx = int(re.sub(r"\D", "", os.path.basename(path)))
        try:
            name = open(os.path.join(path, "name")).read().strip()
            node_index = int(open(os.path.join(path, "index")).read().strip())
        except (OSError, ValueError):
            continue
        if node_index == 0:
            out.append({"device": idx, "name": name})
    return out


def open_source(cv2, cfg: dict):
    """Open the camera described by cfg. Returns an object with read()/release(), or
    raises RuntimeError with a message for the UI."""
    kind = cfg.get("type", "v4l2")
    if kind == "scrcpy":
        from .scrcpy_cam import ScrcpyCamera, list_adb_devices

        serial = cfg.get("serial") or ""
        if not serial:
            raise RuntimeError("no phone selected")
        if serial not in [d["serial"] for d in list_adb_devices()]:
            raise RuntimeError(f"phone {serial} is not on adb")
        return ScrcpyCamera(serial, str(cfg.get("camera_id", "0")), cfg["width"], cfg["height"], cfg["fps"],
                            torch=bool(cfg.get("torch", False)), zoom=float(cfg.get("zoom", 0.0)))
    if kind == "pico":
        from .pico_serial import PicoCamera

        return PicoCamera(cfg.get("port", "/dev/ttyACM0"), cfg["width"], cfg["height"], cfg.get("xclk_div", 12),
                          cfg.get("ae_target", 48), cfg.get("jpeg_scale", 8))
    cap = _open(cv2, cfg)
    if cap is None:
        raise RuntimeError(f"cannot open /dev/video{cfg['device']}")
    return cap


def _open(cv2, cfg: dict):
    cap = cv2.VideoCapture(int(cfg["device"]), cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
    cap.set(cv2.CAP_PROP_FPS, cfg["fps"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    _set_dynamic_framerate(int(cfg["device"]), not cfg.get("fixed_framerate", True))
    return cap


def _set_dynamic_framerate(device: int, enabled: bool) -> None:
    """UVC "exposure, dynamic framerate": in low light the camera lengthens exposure
    and halves its frame rate (30 -> 15 fps). Tracking wants the frame rate, so the
    hub turns it off by default; the image gets darker instead."""
    import fcntl
    import struct

    VIDIOC_S_CTRL, V4L2_CID_EXPOSURE_AUTO_PRIORITY = 0xC008561C, 0x009A0903
    try:
        fd = os.open(f"/dev/video{device}", os.O_RDWR)
    except OSError:
        return
    try:
        fcntl.ioctl(fd, VIDIOC_S_CTRL, struct.pack("Ii", V4L2_CID_EXPOSURE_AUTO_PRIORITY, int(enabled)))
    except OSError:
        pass  # control not supported by this camera
    finally:
        os.close(fd)


class _Grabber:
    """Reads frames on its own thread and keeps only the newest one.

    Without this, a tracker that takes slightly longer than one frame period
    makes the blocking read skip to every second frame (30 -> 15 fps), and
    frames wait in the driver queue while the tracker is busy.
    """

    def __init__(self, cap):
        self.cap = cap
        self.cond = threading.Condition()
        self.frame = None
        self.t_ns = 0
        self.failed = False
        self.count = 0  # frames captured, for the capture-rate stat
        self.offsets = deque(maxlen=900)  # arrival minus device timestamp, last ~30 s
        self.running = True
        self.thread = threading.Thread(target=self._run, name="grab", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            t_ns = time.monotonic_ns()
            pts_us = getattr(self.cap, "pts_us", 0)
            if ok and pts_us:
                # A phone stamps each frame with its own clock at capture. Arrival time
                # wobbles with the network (Wi-Fi especially); the smallest gap between
                # arrival and the phone's stamp seen lately is the link's fixed delay, so
                # stamp + that gap is when the frame was really taken, without the wobble.
                self.offsets.append(t_ns - pts_us * 1000)
                t_ns = pts_us * 1000 + min(self.offsets)
            with self.cond:
                if ok:
                    self.frame, self.t_ns = frame, t_ns
                    self.count += 1
                else:
                    self.failed = True
                self.cond.notify()
            if not ok:
                return

    def next(self, timeout: float):
        """Newest frame not yet returned, as (frame, t_ns); None on timeout or failure."""
        with self.cond:
            if self.frame is None and not self.failed:
                self.cond.wait(timeout)
            if self.frame is None:
                return None
            frame, self.frame = self.frame, None
            return frame, self.t_ns

    def close(self) -> None:
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


def _roi_hands(backend, rgb, dets: list, px, vis, boxes: list) -> list:
    """Search for a hand around each wrist the body tracker sees, where the full-picture
    hand detector found none. Returns extra detections; `boxes` receives the search boxes."""
    h, w = rgb.shape[:2]
    out = []
    shoulders = float(np.linalg.norm(px[11] - px[12])) if min(vis[11], vis[12]) > 0.5 else 0.0
    for wrist, elbow in ((15, 13), (16, 14)):
        if vis[wrist] < 0.5:
            continue
        fore = float(np.linalg.norm(px[wrist] - px[elbow])) if vis[elbow] > 0.5 else 0.0
        reach = max(fore, 0.7 * shoulders, 24.0)  # about a forearm's length in pixels
        if any(float(np.linalg.norm(d.image[0] - px[wrist])) < 0.8 * reach for d in dets):
            continue  # already found by the full-picture detector
        if reach > 0.28 * max(w, h):
            continue  # the hand is large in the picture: the full-picture detector's job
        along = (px[wrist] - px[elbow]) / fore if fore > 1.0 else np.zeros(2)
        cx, cy = px[wrist] + along * 0.45 * reach  # the hand lies beyond the wrist
        half = 1.25 * reach
        x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
        x1, y1 = int(min(cx + half, w)), int(min(cy + half, h))
        boxes.append((x0, y0, x1, y1))
        det = backend.detect(rgb, x0, y0, x1, y1)
        if det is not None and float(np.linalg.norm(det.image[0] - px[wrist])) < 1.2 * reach:
            out.append(det)
    return out


def _area_hands(backend, rgb, dets: list, px, vis, boxes: list) -> list:
    """Search the whole area the arms can reach, around the shoulders and above the head.

    The wrist search trusts the body tracker's wrists, which are sometimes wrong (a raised
    arm drawn as resting on the desk). This coarser search finds such a hand anyway, as
    long as the person is not too small in the picture."""
    if min(vis[11], vis[12]) < 0.5:
        return []
    h, w = rgb.shape[:2]
    shoulders = float(np.linalg.norm(px[11] - px[12]))
    torso = max((float(np.linalg.norm(px[a] - px[b])) for a, b in ((11, 23), (12, 24)) if min(vis[a], vis[b]) > 0.5),
                default=0.0)
    unit = max(shoulders, 0.6 * torso, 20.0)  # about a shoulder width in pixels
    if unit > 0.22 * max(w, h):
        return []  # the person is large in the picture: the full-picture detector's job
    cx, cy = (px[11] + px[12]) / 2.0 - np.array([0.0, 0.4 * unit])
    half = 2.4 * unit
    x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
    x1, y1 = int(min(cx + half, w)), int(min(cy + half, h))
    boxes.append((x0, y0, x1, y1))
    out = []
    for det in backend.detect_all(rgb, x0, y0, x1, y1, size=320):
        if all(float(np.linalg.norm(det.image[0] - d.image[0])) > 0.5 * unit for d in dets + out):
            out.append(det)
    return out


class _HeadThread:
    """Runs head detection beside the hand tracker, on the newest frame, at its own pace.

    The head moves slowly and its detector is the slower model, so it must not hold
    up hand frames.
    """

    def __init__(self, backend, min_interval_s: float = 0.2):
        self.backend = backend
        self.min_interval_s = min_interval_s
        self.cond = threading.Condition()
        self.job = None  # (rgb, t_ns)
        self.result = None  # (t_ns, detection or None)
        self.running = True
        self.thread = threading.Thread(target=self._run, name="head", daemon=True)
        self.thread.start()

    def offer(self, rgb, t_ns: int) -> None:
        with self.cond:
            self.job = (rgb, t_ns)
            self.cond.notify()

    def take(self):
        with self.cond:
            out, self.result = self.result, None
            return out

    def _run(self) -> None:
        while self.running:
            with self.cond:
                while self.job is None and self.running:
                    self.cond.wait(0.5)
                if not self.running:
                    return
                (rgb, t_ns), self.job = self.job, None
            started = time.monotonic()
            try:
                det = self.backend.detect(rgb, t_ns // 1_000_000)
            except Exception:
                det = None
            with self.cond:
                self.result = (t_ns, det)
            time.sleep(max(0.0, self.min_interval_s - (time.monotonic() - started)))

    def close(self) -> None:
        self.running = False
        with self.cond:
            self.cond.notify()
        self.thread.join(timeout=2)
        self.backend.close()


def camera_worker(cfg: dict, conn, stop) -> None:
    import cv2
    import numpy as np

    from ..tracking.mediapipe_backend import HeadBackend, MediaPipeBackend, RoiHandBackend

    try:
        backend = MediaPipeBackend()
        head_thread = (_HeadThread(HeadBackend(), 1.0 / max(float(cfg.get("head_hz", 5)), 0.5))
                       if cfg.get("track_head", True) and float(cfg.get("head_hz", 5)) > 0 else None)
        roi_backend = RoiHandBackend() if cfg.get("roi_hands", True) and head_thread is not None else None
        area_backend = RoiHandBackend(num_hands=2) if roi_backend is not None else None
    except Exception as e:  # model download or init failure
        conn.send({"error": f"tracker init failed: {e}"})
        return
    head = None
    body_t_ns = 0  # capture time of the body result in `head`

    mirror = bool(cfg.get("mirror_preview", True))
    # Camera mounted sideways or upside down: turn the image upright before tracking.
    rotate = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
              270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(int(cfg.get("rotate_deg", 0)) % 360)
    # A sensor that delivers a mirrored image must be un-mirrored first: a mirror image
    # swaps left and right hands and cannot be combined with other cameras.
    flip_input = bool(cfg.get("flip_input", False))
    ir_strobe = bool(cfg.get("ir_strobe", False))
    levels = deque(maxlen=8)
    light = str(cfg.get("light_correction", "local"))
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(6, 6))  # gentle: strong settings amplify noise
    tone, lut = None, None  # (black level, span, gamma) of the current tone curve
    grabber = None
    frame_no = 0
    rate_t, rate_n, capture_fps = time.monotonic(), 0, 0.0
    try:
        while not stop.is_set():
            while conn.poll():
                update = conn.recv()
                mirror = bool(update.get("mirror", mirror))
                light = str(update.get("light", light))
            if grabber is None:
                try:
                    grabber = _Grabber(open_source(cv2, cfg))
                except Exception as e:  # device missing, adb or serial failure
                    conn.send({"error": str(e)})
                    stop.wait(3.0)
                    continue
            got = grabber.next(timeout=1.0)
            if got is None:
                if grabber.failed:
                    conn.send({"error": "camera read failed"})
                    grabber.close()
                    grabber = None
                    stop.wait(1.0)
                continue
            frame, t_ns = got
            if ir_strobe:
                # An infrared camera whose emitter lights alternate frames: the unlit ones
                # are black. Keep a frame only if it is in the brighter half of recent ones.
                level = float(frame[::8, ::8].mean())
                levels.append(level)
                if len(levels) >= 4 and max(levels) - min(levels) > 6.0 and level < (max(levels) + min(levels)) / 2.0:
                    continue
            if time.monotonic() - rate_t >= 1.0:
                now = time.monotonic()
                capture_fps = (grabber.count - rate_n) / (now - rate_t)
                rate_t, rate_n = now, grabber.count
            if flip_input:
                frame = cv2.flip(frame, 1)
            if rotate is not None:
                frame = cv2.rotate(frame, rotate)
            if light != "off":
                # Light correction, so the trackers see the person in dim or uneven light.
                # "gamma": stretch the contrast and lift the mid-tones (one tone curve).
                # "local": additionally even out the lighting region by region (CLAHE on
                # lightness), which brings out a person standing in shadow beside a lamp.
                if frame_no % 15 == 0:  # re-measured twice a second
                    grey = cv2.cvtColor(frame[::6, ::6], cv2.COLOR_BGR2GRAY)
                    lo, med, hi = (float(v) for v in np.percentile(grey, (1.5, 50, 98.5)))
                    # Stretch the tones the picture actually uses to the full range (a hazy
                    # or flat picture uses only a narrow band, and a hand then barely differs
                    # from the wall behind it), then lift the mid-tones toward mid-grey.
                    span = max(hi - lo, 40.0)
                    med_s = min(max((med - lo) / span, 0.02), 0.98)
                    g_new = 1.0 if med_s >= 0.40 else float(np.clip(np.log(0.42) / np.log(med_s), 0.45, 1.0))
                    target = (lo, span, g_new)
                    if tone is None or max(abs(a - b) / c for a, b, c in zip(target, tone, (255.0, 255.0, 1.0))) > 0.03:
                        # ease toward the new curve so the picture does not pump
                        tone = target if tone is None else tuple(0.5 * a + 0.5 * b for a, b in zip(target, tone))
                        x = np.clip((np.arange(256) - tone[0]) / tone[1], 0.0, 1.0)
                        lut = np.clip((x ** tone[2]) * 255.0, 0, 255).astype(np.uint8)
                if lut is not None:
                    frame = cv2.LUT(frame, lut)
                if light == "local":
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            t0 = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = backend.detect(rgb, t_ns // 1_000_000)
            frame_no += 1
            head_msg = None
            if head_thread is not None:
                head_thread.offer(rgb, t_ns)
                got_head = head_thread.take()
                if got_head is not None:
                    head_t_ns, head = got_head
                    body_t_ns = head_t_ns
                    head_msg = (head_t_ns, *head) if head is not None else None
            roi_boxes = []
            if roi_backend is not None and head is not None and t_ns - body_t_ns < 500_000_000:
                dets = dets + _roi_hands(roi_backend, rgb, dets, head[0], head[1], roi_boxes)
                # A hand is still missing: the body tracker may have the arm wrong. This wider
                # search is the slowest step, so it runs on every third frame only.
                if len(dets) < 2 and frame_no % 3 == 0:
                    dets = dets + _area_hands(area_backend, rgb, dets, head[0], head[1], roi_boxes)
            proc_ms = (time.perf_counter() - t0) * 1000.0

            jpeg = None
            if frame_no % 2 == 0:  # preview at half rate
                for d in dets:
                    pts = d.image.astype(int)
                    for a, b in HAND_EDGES:
                        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (80, 220, 120), 2, cv2.LINE_AA)
                    for p in pts:
                        cv2.circle(frame, tuple(p), 3, (255, 255, 255), -1, cv2.LINE_AA)
                for x0, y0, x1, y1 in roi_boxes:  # where a hand was searched for around a wrist
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (200, 120, 40), 1)
                if head is not None:
                    bpts = head[0].astype(int)
                    for a, b in BODY_EDGES:
                        if head[1][a] > 0.5 and head[1][b] > 0.5:
                            cv2.line(frame, tuple(bpts[a]), tuple(bpts[b]), (60, 170, 255), 2, cv2.LINE_AA)
                    for i in (0, 2, 5, 7, 8):
                        if head[1][i] > 0.5:
                            cv2.circle(frame, tuple(bpts[i]), 4, (60, 170, 255), 2, cv2.LINE_AA)
                if mirror:
                    frame = cv2.flip(frame, 1)
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                jpeg = buf.tobytes() if ok else None

            conn.send({
                "t_ns": t_ns,
                "size": (frame.shape[1], frame.shape[0]),
                "proc_ms": proc_ms,
                "capture_fps": capture_fps,
                "frame_kb": getattr(grabber.cap, "frame_bytes", 0) / 1000.0,
                "brightness": float(frame[::8, ::8].mean()),
                "hands": [(d.handedness, d.score, d.image, d.world) for d in dets],
                "head": head_msg,  # (t_ns, pixels (5, 2), visibility (5,), world (5, 3)) when new
                "jpeg": jpeg,
            })
    except (BrokenPipeError, EOFError, KeyboardInterrupt):
        pass
    finally:
        if grabber is not None:
            grabber.close()
        backend.close()
        if roi_backend is not None:
            roi_backend.close()
            area_backend.close()
        if head_thread is not None:
            head_thread.close()
