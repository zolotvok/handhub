# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""Android phone camera over adb, using the scrcpy server as the streamer.

The scrcpy server (pushed to the phone, no app install) captures a camera with
Camera2, encodes H.264 and serves it on a local socket that adb forwards over USB.
This module starts it and decodes the stream into BGR frames.

Each frame's PTS is the phone's own clock in microseconds (kept in `pts_us`).
Frames are still stamped on arrival for now; mapping the device clock to the hub
clock (time sync) will replace that.
"""
from __future__ import annotations

import random
import socket
import struct
import subprocess
import time
from pathlib import Path

SERVER_VERSION = "5.0.1"
SERVER_URL = (f"https://github.com/Genymobile/scrcpy/releases/download/v{SERVER_VERSION}/"
              f"scrcpy-server-v{SERVER_VERSION}")


def ensure_server() -> Path:
    """The scrcpy server (Apache-2.0, by Genymobile), downloaded once into the cache.
    HANDHUB_SCRCPY_SERVER points at an existing copy instead."""
    import os
    import urllib.request

    from ..config import CACHE_DIR

    override = os.environ.get("HANDHUB_SCRCPY_SERVER")
    if override:
        return Path(override)
    path = CACHE_DIR / f"scrcpy-server-v{SERVER_VERSION}"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        print(f"[cameras] downloading {SERVER_URL}")
        urllib.request.urlretrieve(SERVER_URL, tmp)
        tmp.replace(path)
    return path
DEVICE_JAR = "/data/local/tmp/scrcpy-server.jar"
# scrcpy 5 stream: a 4-byte codec id, then packets with a 12-byte header
# (u64 flags|pts, u32 size). A session packet carries width/height in the header
# itself and has no payload.
PACKET_FLAG_SESSION = 1 << 63
PACKET_FLAG_CONFIG = 1 << 62
PACKET_FLAG_KEY = 1 << 61
PACKET_PTS_MASK = PACKET_FLAG_KEY - 1


def adb(serial: str, *args: str, timeout: float = 15.0) -> subprocess.CompletedProcess:
    return subprocess.run(["adb", "-s", serial, *args], capture_output=True, text=True, timeout=timeout)


def list_adb_devices() -> list[dict]:
    """Phones on adb: [{serial, model}]."""
    try:
        out = subprocess.run(["adb", "devices", "-l"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    devs = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            model = next((p.split(":", 1)[1] for p in parts if p.startswith("model:")), "")
            devs.append({"serial": parts[0], "model": model})
    return devs


class ScrcpyCamera:
    """cv2.VideoCapture-like reader: read() -> (ok, bgr_frame)."""

    def __init__(self, serial: str, camera_id: str = "0", width: int = 1280, height: int = 720,
                 fps: int = 30, bitrate: int = 6_000_000, torch: bool = False, zoom: float = 0.0):
        import av

        self.serial = serial
        self._scid = ""
        self.proc: subprocess.Popen | None = None
        self.sock: socket.socket | None = None
        self.port = 0
        self._codec = av.CodecContext.create("h264", "r")
        self._codec.thread_count = 2
        self._av = av
        self._config = b""
        self._pending: list = []

        try:
            server = ensure_server()
        except OSError as e:
            raise RuntimeError(f"cannot get the scrcpy server: {e}") from e
        r = adb(serial, "push", str(server), DEVICE_JAR)
        if r.returncode != 0:
            raise RuntimeError(f"adb push failed: {r.stderr.strip() or r.stdout.strip()}")

        scid = f"{random.getrandbits(31):08x}"
        self._scid = scid
        r = adb(serial, "forward", "tcp:0", f"localabstract:scrcpy_{scid}")
        if r.returncode != 0:
            raise RuntimeError(f"adb forward failed: {r.stderr.strip()}")
        self.port = int(r.stdout.strip())

        cmd = ["adb", "-s", serial, "shell", f"CLASSPATH={DEVICE_JAR}", "app_process", "/",
               "com.genymobile.scrcpy.Server", SERVER_VERSION, f"scid={scid}", "log_level=info",
               "tunnel_forward=true", "video_source=camera", f"camera_id={camera_id}",
               f"camera_size={width}x{height}", f"camera_fps={fps}", f"max_fps={fps}",
               f"video_bit_rate={bitrate}", "video_codec=h264", "audio=false", "control=false",
               *(["camera_torch=true"] if torch else []),  # the phone's flash LED as a lamp
               *([f"camera_zoom={zoom:g}"] if zoom > 0 else []),  # below 1 = the ultrawide lens
               "cleanup=false", "send_device_meta=false", "send_dummy_byte=true",
               "send_codec_meta=false", "send_frame_meta=true"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # The server needs a moment before it listens; the dummy byte confirms the connection.
        deadline = time.monotonic() + 8.0
        while True:
            try:
                s = socket.create_connection(("127.0.0.1", self.port), timeout=2.0)
                s.settimeout(3.0)
                if s.recv(1):
                    self.sock = s
                    break
                s.close()
            except OSError:
                pass
            if self.proc.poll() is not None or time.monotonic() > deadline:
                log = self.proc.stdout.read() if self.proc.poll() is not None else ""
                self.release()
                raise RuntimeError(f"scrcpy server did not start: {log.strip()[-300:]}")
            time.sleep(0.2)
        self.sock.settimeout(5.0)
        codec = self._recv_exact(4)
        if codec != b"h264":
            self.release()
            raise RuntimeError(f"unexpected stream codec {codec!r}")
        self.pts_us = 0  # device clock of the last decoded frame, microseconds

    def _recv_exact(self, n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = self.sock.recv(n - len(buf))
            except OSError:
                return None
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)

    def read(self):
        """Block until the next decoded frame. Returns (ok, bgr ndarray)."""
        while not self._pending:
            head = self._recv_exact(12)
            if head is None:
                return False, None
            pts_flags, size = struct.unpack(">QI", head)
            if pts_flags & PACKET_FLAG_SESSION:
                continue
            data = self._recv_exact(size)
            if data is None:
                return False, None
            if pts_flags & PACKET_FLAG_CONFIG:
                self._config = data  # SPS/PPS, prepended to the next packet
                continue
            packet = self._av.Packet(self._config + data)
            self._config = b""
            self.pts_us = pts_flags & PACKET_PTS_MASK
            try:
                self._pending = list(self._codec.decode(packet))
            except self._av.error.FFmpegError:
                continue
        frame = self._pending.pop(0)
        return True, frame.to_ndarray(format="bgr24")

    def release(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        if self.port:
            try:
                # The server can outlive the adb shell; it would keep the camera busy.
                adb(self.serial, "shell", "pkill", "-f", f"scid={self._scid}", timeout=5)
                adb(self.serial, "forward", "--remove", f"tcp:{self.port}", timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
            self.port = 0
