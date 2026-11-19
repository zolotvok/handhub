# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
import socket
import struct
import time

import numpy as np
import pytest

from hub.driver_link import (HAPTIC, HAND_FMT, MAGIC_HAPTIC, MAGIC_STATUS, PACKET, STATUS, DriverLink,
                             HandState, pack_packet, unpack_packet)


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_hand(seed):
    rng = np.random.default_rng(seed)
    return HandState(
        tracked=True, buttons=5 + seed, pos=rng.normal(size=3), rot=np.array([0.5, 0.5, 0.5, 0.5]),
        trigger=0.25, grip=0.75, curl=[0.1, 0.2, 0.3, 0.4, 0.5], joy=(0.1, -0.2),
        lm=rng.normal(size=(21, 3)),
    )


def test_struct_sizes():
    assert PACKET.size == 664
    assert STATUS.size == 40
    assert HAPTIC.size == 20
    assert struct.calcsize("<II3f4f2f5f2f63f") == 324
    assert HAND_FMT == "II3f4f2f5f2f63f"


def test_pack_unpack_round_trip():
    left, right = make_hand(1), make_hand(2)
    data = pack_packet(7, 0.5, 1, left, right)
    assert len(data) == 664
    assert data[:4] == b"HHB1"
    assert struct.unpack_from("<I", data, 4)[0] == 7

    out = unpack_packet(data)
    assert out["seq"] == 7 and out["flags"] == 1
    assert out["yaw_offset"] == pytest.approx(0.5)
    for got, want in zip(out["hands"], (left, right)):
        assert got.tracked and got.buttons == want.buttons
        assert np.allclose(got.pos, want.pos, atol=1e-6)
        assert np.allclose(got.rot, want.rot, atol=1e-6)
        assert got.trigger == pytest.approx(0.25) and got.grip == pytest.approx(0.75)
        assert np.allclose(got.curl, want.curl, atol=1e-6)
        assert np.allclose(got.joy, want.joy, atol=1e-6)
        assert np.allclose(got.lm, want.lm, atol=1e-6)


def test_unpack_rejects_bad_magic():
    data = bytearray(pack_packet(1, 0.0, 0, make_hand(1), make_hand(2)))
    data[:4] = b"XXXX"
    with pytest.raises(ValueError):
        unpack_packet(bytes(data))


@pytest.fixture
def link():
    port, status_port = free_port(), free_port()
    lk = DriverLink("127.0.0.1", port, status_port)
    yield lk, port, status_port
    lk.close()


def test_send_delivers_664_byte_datagram(link):
    lk, port, _ = link
    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        listener.bind(("127.0.0.1", port))
        listener.settimeout(2)
        lk.send(0.0, 0, HandState(), HandState())
        data, _ = listener.recvfrom(1024)
        assert len(data) == 664
        assert data[:4] == b"HHB1"
    finally:
        listener.close()


def test_status_updates_hmd_and_connection(link):
    lk, _, status_port = link
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        assert not lk.connected
        pkt = STATUS.pack(MAGIC_STATUS, 1, 1, 0.1, 1.5, -0.2, 1.0, 0.0, 0.0, 0.0)
        deadline = time.monotonic() + 2
        while not lk.hmd_valid and time.monotonic() < deadline:
            sender.sendto(pkt, ("127.0.0.1", status_port))
            time.sleep(0.02)
            lk.poll()
        assert lk.connected
        assert lk.hmd_valid
        assert np.allclose(lk.hmd_pos, [0.1, 1.5, -0.2], atol=1e-6)
        assert np.allclose(lk.hmd_rot, [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    finally:
        sender.close()


def test_haptic_is_recorded(link):
    lk, _, status_port = link
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        pkt = HAPTIC.pack(MAGIC_HAPTIC, 1, 0.2, 170.0, 0.5)
        deadline = time.monotonic() + 2
        while not lk.haptics and time.monotonic() < deadline:
            sender.sendto(pkt, ("127.0.0.1", status_port))
            time.sleep(0.02)
            lk.poll()
        assert lk.haptics
        h = lk.haptics[-1]
        assert h["hand"] == 1
        assert h["duration_s"] == pytest.approx(0.2)
        assert h["frequency_hz"] == pytest.approx(170.0)
        assert h["amplitude"] == pytest.approx(0.5)
    finally:
        sender.close()
