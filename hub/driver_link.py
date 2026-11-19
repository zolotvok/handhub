# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""Hub <-> runtime UDP protocol. The packet layout is defined once, here."""
from __future__ import annotations

import socket
import struct
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

HAND_FMT = "II3f4f2f5f2f63f"
PACKET = struct.Struct("<4sIfI" + HAND_FMT * 2)
STATUS = struct.Struct("<4sII3f4f")
HAPTIC = struct.Struct("<4sIfff")
MAGIC_PACKET, MAGIC_STATUS, MAGIC_HAPTIC = b"HHB1", b"HHD1", b"HHH1"
FLAG_FOLLOW_HMD_POSITION = 1

assert PACKET.size == 664 and STATUS.size == 40 and HAPTIC.size == 20


@dataclass
class HandState:
    tracked: bool = False
    buttons: int = 0
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rot: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))  # w, x, y, z
    trigger: float = 0.0
    grip: float = 0.0
    curl: list = field(default_factory=lambda: [0.0] * 5)
    joy: tuple = (0.0, 0.0)
    lm: np.ndarray = field(default_factory=lambda: np.zeros((21, 3)))  # controller-local, metres

    def fields(self) -> list:
        return [
            int(self.tracked), int(self.buttons), *map(float, self.pos), *map(float, self.rot),
            float(self.trigger), float(self.grip), *map(float, self.curl), *map(float, self.joy),
            *map(float, np.asarray(self.lm, float).reshape(63)),
        ]


def pack_packet(seq: int, yaw_offset: float, flags: int, left: HandState, right: HandState) -> bytes:
    return PACKET.pack(MAGIC_PACKET, seq & 0xFFFFFFFF, float(yaw_offset), flags, *left.fields(), *right.fields())


def unpack_packet(data: bytes) -> dict:
    v = PACKET.unpack(data)
    if v[0] != MAGIC_PACKET:
        raise ValueError("bad magic")
    hands = []
    for i in range(2):
        h = v[4 + i * 81: 4 + (i + 1) * 81]
        hands.append(HandState(
            tracked=bool(h[0]), buttons=h[1], pos=np.array(h[2:5]), rot=np.array(h[5:9]),
            trigger=h[9], grip=h[10], curl=list(h[11:16]), joy=tuple(h[16:18]),
            lm=np.array(h[18:81]).reshape(21, 3),
        ))
    return {"seq": v[1], "yaw_offset": v[2], "flags": v[3], "hands": hands}


class DriverLink:
    """Sends hand packets, receives Status (HMD pose) and Haptic packets."""

    def __init__(self, host: str, port: int, status_port: int):
        self.addr = (host, port)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx_error: str | None = None
        try:
            self.rx.bind(("127.0.0.1", status_port))
        except OSError as e:
            self.rx_error = str(e)
        self.rx.setblocking(False)
        self.seq = 0
        self.last_status_t = 0.0
        self.hmd_valid = False
        self.hmd_pos = np.array([0.0, 1.6, 0.0])
        self.hmd_rot = np.array([1.0, 0.0, 0.0, 0.0])
        self.haptics: list = []
        self.hmd_history: deque = deque(maxlen=120)  # (monotonic time, quaternion w x y z)

    def send(self, yaw_offset: float, flags: int, left: HandState, right: HandState) -> None:
        self.seq += 1
        try:
            self.tx.sendto(pack_packet(self.seq, yaw_offset, flags, left, right), self.addr)
        except OSError:
            pass

    def poll(self) -> None:
        while True:
            try:
                data, _ = self.rx.recvfrom(256)
            except (BlockingIOError, OSError):
                return
            if len(data) == STATUS.size and data[:4] == MAGIC_STATUS:
                v = STATUS.unpack(data)
                self.last_status_t = time.monotonic()
                self.hmd_valid = bool(v[2])
                if self.hmd_valid:
                    self.hmd_pos, self.hmd_rot = np.array(v[3:6]), np.array(v[6:10])
                    self.hmd_history.append((self.last_status_t, self.hmd_rot))
            elif len(data) == HAPTIC.size and data[:4] == MAGIC_HAPTIC:
                v = HAPTIC.unpack(data)
                self.haptics.append({"hand": v[1], "duration_s": v[2], "frequency_hz": v[3], "amplitude": v[4]})
                del self.haptics[:-16]

    def hmd_rot_at(self, t: float, max_gap: float = 0.15):
        """The headset orientation reported nearest to monotonic time t, or None."""
        best = min(self.hmd_history, key=lambda e: abs(e[0] - t), default=None)
        return None if best is None or abs(best[0] - t) > max_gap else best[1]

    @property
    def connected(self) -> bool:
        return time.monotonic() - self.last_status_t < 1.0

    def close(self) -> None:
        self.tx.close()
        self.rx.close()
