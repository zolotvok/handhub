# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
import json
import socket
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hub.config import DEFAULTS, Config
from hub.fusion import Hub
from hub.server import create_app

INDEX_HTML = Path(__file__).resolve().parents[1] / "hub" / "web" / "index.html"


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def env(tmp_path):
    path = tmp_path / "c.json"
    config = Config(path)
    config.update({"driver": {"port": free_port(), "status_port": free_port()}})
    hub = Hub(config=config, start_camera=False)
    client = TestClient(create_app(hub))
    yield client, path
    hub.link.close()


@pytest.mark.skipif(not INDEX_HTML.exists(), reason="hub/web/index.html not written yet")
def test_index_returns_html(env):
    client, _ = env
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_get_config_returns_defaults(env):
    client, _ = env
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == set(DEFAULTS)
    assert body["camera"] == DEFAULTS["camera"]  # driver ports are overridden by the fixture


def test_put_config_persists(env):
    client, path = env
    r = client.put("/api/config", json={"smoothing": {"beta": 0.05}})
    assert r.status_code == 200
    assert client.get("/api/config").json()["smoothing"]["beta"] == 0.05
    assert json.loads(path.read_text())["smoothing"]["beta"] == 0.05


def test_get_cameras_lists_devices(env):
    client, _ = env
    r = client.get("/api/cameras")
    assert r.status_code == 200
    assert isinstance(r.json()["devices"], list)


def test_calibrate_rest_without_hand_is_not_ok(env):
    client, _ = env
    r = client.post("/api/calibrate/rest")
    assert r.status_code == 200
    assert r.json()["ok"] is False


def test_calibrate_intrinsics_is_not_implemented(env):
    client, _ = env
    assert client.post("/api/calibrate/intrinsics").status_code == 501


def test_websocket_sends_state(env):
    client, _ = env
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
    assert "hands" in msg
