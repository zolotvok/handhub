# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""Web server: config/calibration API, MJPEG preview, state WebSocket, static UI."""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .cameras.v4l2 import list_devices
from .cameras.scrcpy_cam import list_adb_devices
from .fusion import PRIMARY, Hub

WEB_DIR = Path(__file__).parent / "web"


def create_app(hub: Hub) -> FastAPI:
    app = FastAPI(title="HandHub")

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/config")
    def get_config():
        return hub.config.get()

    @app.put("/api/config")
    def put_config(patch: dict = Body(...)):
        return hub.config.update(patch)

    @app.post("/api/record/raw")
    def record_raw(seconds: float = 60.0):
        return hub.record_raw_start(seconds)

    @app.get("/api/record/raw")
    def record_raw_status():
        return {"recording": hub.raw_log is not None, "last": hub.raw_last}

    @app.post("/api/game/recentre")
    def game_recentre():
        return hub.request_recentre()

    @app.get("/api/debug/views")
    def debug_views():
        return hub.debug_views()

    @app.get("/api/cameras")
    def cameras():
        return {"devices": list_devices(), "active": hub.config.get()["camera"]["device"],
                "phones": list_adb_devices()}

    @app.post("/api/calibrate/{kind}")
    def calibrate(kind: str):
        if kind == "rest":
            return hub.calibrate_rest()
        if kind == "cameras":
            return hub.calibrate_cameras_start()
        if kind == "headset":
            return hub.calibrate_headcam_start()
        if kind == "level":
            return hub.calibrate_level()
        if kind == "head":
            return hub.calibrate_head()
        if kind == "align":
            return hub.calibrate_align()
        if kind in ("intrinsics", "extrinsics"):
            raise HTTPException(501, f"{kind} calibration arrives in stage 2")
        raise HTTPException(404, "unknown calibration")

    @app.get("/stream/{cam}.mjpg")
    async def stream(cam: str):
        source = hub.cams.get(cam) or hub.cams[PRIMARY]  # "0" and unknown names: the webcam

        async def frames():
            seq = -1
            while True:
                if source.jpeg is not None and source.jpeg_seq != seq:
                    seq, data = source.jpeg_seq, source.jpeg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                await asyncio.sleep(0.03)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            while True:
                await sock.send_json(hub.snapshot())
                await asyncio.sleep(1 / 30)
        except (WebSocketDisconnect, RuntimeError):
            pass

    app.mount("/", StaticFiles(directory=WEB_DIR), name="static")
    return app
