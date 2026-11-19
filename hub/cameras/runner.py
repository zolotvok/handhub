# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""One camera's worker process, as seen from the hub: lifecycle, preview and statistics."""
from __future__ import annotations

import multiprocessing as mp
import time

from .v4l2 import camera_worker

# A change to any of these restarts the camera process.
RESTART_KEYS = ("type", "enabled", "device", "width", "height", "fps", "fixed_framerate",
                "serial", "camera_id", "port", "xclk_div", "rotate_deg", "head_hz", "ae_target", "flip_input", "jpeg_scale", "torch", "ir_strobe", "zoom")


LIVE_KEYS = ("mirror_preview", "light_correction")


class CameraRunner:
    def __init__(self, name: str):
        self.name = name
        self.proc = None
        self.conn = None
        self._stop_evt = None
        self._worker_cfg: dict | None = None
        self.jpeg: bytes | None = None
        self.jpeg_seq = 0
        self.stats = {"fps": 0.0, "latency_ms": 0.0, "proc_ms": 0.0, "error": None, "size": None,
                      "brightness": None, "capture_fps": 0.0, "frame_kb": 0.0, "last_frame_t": 0.0, "hands": 0}
        self._frame_times: list[float] = []
        self.last_frame: tuple | None = None  # (t_ns, size, detections)
        self.last_head = None  # (t_ns, pixels, visibility) from the newest message, if any

    @staticmethod
    def _key(cam_cfg: dict) -> dict:
        return {k: cam_cfg.get(k) for k in RESTART_KEYS}

    def sync(self, cam_cfg: dict) -> None:
        """Start, stop or restart the worker so it matches the config."""
        want = self._key(cam_cfg)
        if self.proc is not None and (want != self._key(self._worker_cfg) or not self.proc.is_alive()):
            died = want == self._key(self._worker_cfg)
            self.stop()
            if died:
                self.stats["error"] = "camera process exited, restarting"
        if self.proc is None and cam_cfg["enabled"]:
            ctx = mp.get_context("spawn")
            self.conn, child = ctx.Pipe(duplex=True)
            self._stop_evt = ctx.Event()
            self.proc = ctx.Process(target=camera_worker, args=(cam_cfg, child, self._stop_evt), daemon=True)
            self.proc.start()
            child.close()
            self._worker_cfg = dict(cam_cfg)
        elif self.proc is not None and any(self._worker_cfg.get(k) != cam_cfg.get(k) for k in LIVE_KEYS):
            for k in LIVE_KEYS:  # settings the worker takes over without restarting
                self._worker_cfg[k] = cam_cfg.get(k)
            self.conn.send({"mirror": cam_cfg["mirror_preview"], "light": cam_cfg.get("light_correction", "local")})
        if not cam_cfg["enabled"]:
            self.stats["error"] = None

    def stop(self) -> None:
        if self.proc is None:
            return
        self._stop_evt.set()
        self.proc.join(timeout=3)
        if self.proc.is_alive():
            self.proc.terminate()
        self.conn.close()
        self.proc = self.conn = self._worker_cfg = None

    def on_message(self, msg: dict) -> tuple | None:
        """Update preview and statistics. Returns (t_ns, size, detections) for a frame."""
        if "error" in msg:
            self.stats["error"] = msg["error"]
            return None
        now = time.monotonic()
        self._frame_times = [t for t in self._frame_times if now - t < 1.0] + [now]
        self.stats.update(
            error=None, size=msg["size"], proc_ms=msg["proc_ms"], brightness=msg["brightness"],
            capture_fps=msg.get("capture_fps", 0.0), fps=float(len(self._frame_times)),
            frame_kb=msg.get("frame_kb", 0.0),
            latency_ms=(time.monotonic_ns() - msg["t_ns"]) / 1e6, last_frame_t=now, hands=len(msg["hands"]),
        )
        if msg.get("jpeg"):
            self.jpeg = msg["jpeg"]
            self.jpeg_seq += 1
        self.last_head = msg.get("head")
        self.last_frame = (msg["t_ns"], msg["size"], msg["hands"])
        return self.last_frame

    def public_stats(self) -> dict:
        s = dict(self.stats)
        s["online"] = time.monotonic() - s.pop("last_frame_t") < 1.5 and not s["error"]
        if not s["online"]:
            s["fps"] = 0.0
        return s
