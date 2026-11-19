# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""Pico 2 + OV5640 camera over USB serial, using the Pico's existing firmware protocol.

Protocol (see rpi_camera/video.py and camhelp.py):
  'y' + 'j'|'r' + WWWW + HHHH   set the video mode (JPEG or raw RGB565), answers OK/ERR
  'v'                            start streaming; each frame is the text line
                                 "VFRAME <w> <h> <fmt> <bytes>" followed by the data
  'q'                            stop streaming

Frames carry no timestamp, so they are stamped on arrival (like the phone source).
"""
from __future__ import annotations

import time

import numpy as np


class PicoCamera:
    """cv2.VideoCapture-like reader: read() -> (ok, bgr_frame)."""

    def __init__(self, port: str = "/dev/ttyACM0", width: int = 640, height: int = 480, xclk_div: int = 12,
                 ae_target: int = 48, jpeg_scale: int = 8):
        import cv2
        import serial

        self._cv2 = cv2
        try:
            self.ser = serial.Serial(port, 115200, timeout=3)
        except serial.SerialException as e:
            raise RuntimeError(f"cannot open {port}: {e}") from e
        time.sleep(0.3)
        self._stop_stream()
        self.ser.write(f"yj{width:04d}{height:04d}".encode())
        self.ser.flush()
        answer = self._wait_line(("OK", "ERR"))
        if not answer.startswith("OK"):
            self.release()
            raise RuntimeError(f"Pico rejected video mode jpeg {width}x{height}: {answer or 'no answer'}")
        # Sensor clock XCLK = 150 MHz / xclk_div. The firmware default is 12 (12.5 MHz,
        # 16.8 fps); a smaller divider speeds up the whole sensor: 7 gives ~28.7 fps.
        self.ser.write(f"x{int(xclk_div):02x}".encode())
        self.ser.flush()
        self._wait_line(("OK", "ERR"))
        time.sleep(0.3)
        self._set_ae_target(ae_target)
        # JPEG quantisation scale: 2 = best quality and largest frames ... 63 = smallest.
        # The Pico's USB carries about 1 MB/s, so the frame size limits the frame rate:
        # 30 fps needs frames under roughly 30 kB.
        self.ser.write(f"j{max(2, min(int(jpeg_scale), 63)):02x}".encode())
        self.ser.flush()
        self._wait_line(("OK", "ERR"))
        self.frame_bytes = 0  # size of the last frame as sent by the Pico
        self.ser.write(b"v")
        self.ser.flush()

    def _write_reg(self, reg: int, value: int) -> None:
        self.ser.write(f"w{reg:04x}{value & 0xFF:02x}".encode())
        self.ser.flush()
        self._wait_line(("OK", "ERR"))

    def _set_ae_target(self, target: int) -> None:
        """Auto-exposure brightness target (OV5640 0x3a0f.., firmware default 48).

        A bright lamp in view makes the sensor expose for the lamp and leaves the
        person dark; a higher target brightens the whole picture.
        """
        t = max(16, min(int(target), 110))
        for reg, value in ((0x3A0F, t), (0x3A10, t - 8), (0x3A1B, t), (0x3A1E, t - 10),
                           (0x3A11, min(2 * t, 0xF0)), (0x3A1F, max(t // 2 - 10, 8))):
            self._write_reg(reg, value)

    def _wait_line(self, prefixes: tuple, tries: int = 20) -> str:
        for _ in range(tries):
            text = self.ser.readline().decode("utf-8", errors="replace").strip()
            if not text:
                return ""
            if text.startswith(prefixes):
                return text
        return ""

    def _stop_stream(self) -> None:
        """Stop a running stream (harmless when idle) and drain the input."""
        self.ser.write(b"q")
        self.ser.flush()
        old, self.ser.timeout = self.ser.timeout, 0.2
        quiet_since, deadline = time.monotonic(), time.monotonic() + 5
        while time.monotonic() < deadline and time.monotonic() - quiet_since < 0.6:
            if self.ser.read(65536):
                quiet_since = time.monotonic()
        self.ser.timeout = old

    def _read_exact(self, n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.ser.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)

    def read(self):
        try:
            for _ in range(50):  # skip status lines and damaged frames
                text = self.ser.readline().decode("utf-8", errors="replace").strip()
                if not text or text.startswith(("VIDEO_END", "CAPTURE_ERROR")):
                    return False, None
                if not text.startswith("VFRAME "):
                    continue
                try:
                    _, w, h, fmt, count = text.split()
                    w, h, count = int(w), int(h), int(count)
                except ValueError:
                    continue
                raw = self._read_exact(count)
                if raw is None:
                    return False, None
                self.frame_bytes = count
                if fmt == "JPEG":
                    end = raw.rfind(b"\xff\xd9")  # the sensor pads after end-of-image
                    data = raw[: end + 2] if end >= 0 else raw
                    bgr = self._cv2.imdecode(np.frombuffer(data, dtype=np.uint8), self._cv2.IMREAD_COLOR)
                    if bgr is None:
                        continue
                else:  # RGB565, big-endian
                    v = np.frombuffer(raw, dtype=">u2").astype(np.uint32).reshape(h, w)
                    bgr = np.dstack(((v & 0x1F) * 255 // 31, ((v >> 5) & 0x3F) * 255 // 63,
                                     ((v >> 11) & 0x1F) * 255 // 31)).astype(np.uint8)
                return True, bgr
        except OSError:  # includes serial.SerialException: device unplugged
            pass
        return False, None

    def release(self) -> None:
        try:
            self._stop_stream()
        except OSError:
            pass
        try:
            self.ser.close()
        except OSError:
            pass
