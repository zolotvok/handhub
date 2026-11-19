# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""`handhub` / `python -m hub`: start the hub and its web UI."""
from __future__ import annotations

import argparse

import uvicorn

from .fusion import Hub
from .server import create_app


def main() -> None:
    ap = argparse.ArgumentParser(prog="handhub")
    ap.add_argument("--host", default="127.0.0.1",
                    help="UI bind address; use 0.0.0.0 to open the UI from a phone (no authentication)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-camera", action="store_true", help="run without opening a camera")
    args = ap.parse_args()

    hub = Hub(start_camera=not args.no_camera)
    hub.start()
    try:
        # Preview streams and WebSockets never finish on their own: don't wait for them on exit.
        uvicorn.run(create_app(hub), host=args.host, port=args.port, log_level="warning",
                    timeout_graceful_shutdown=1)
    finally:
        hub.stop()


if __name__ == "__main__":
    main()
