# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Schmiz Otto
"""Hub configuration: defaults deep-merged with ~/.config/handhub/config.json."""
from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path

CONFIG_PATH = Path(os.environ.get("HANDHUB_CONFIG", Path.home() / ".config/handhub/config.json"))
CACHE_DIR = Path(os.environ.get("HANDHUB_CACHE", Path.home() / ".cache/handhub"))
# Recordings and logs (they contain camera pictures of the room: keep them out of the repository).
DATA_DIR = Path(os.environ.get("HANDHUB_DATA", Path.home() / ".local/share/handhub"))

DEFAULTS: dict = {
    "camera": {
        "enabled": True,
        "device": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
        "fixed_framerate": True,  # stop the camera halving its frame rate in low light
        "flip_input": False,
        "light_correction": "local",  # "off" | "gamma" (brighten) | "local" (brighten + even out)
        "rotate_deg": 0,
        "head_hz": 15,  # head detections per second; the webcam sees the face, so it runs fastest
        "hfov_deg": 70.0,
        "swap_hands": False,
        "mirror_preview": True,
        # Camera pose in the body frame (see hub/geometry.py): position C and angles on
        # top of the "facing the user" base rotation.
        "pos": [0.0, -0.2, -0.6],
        "yaw_deg": 0.0,
        "pitch_deg": 0.0,
        "roll_deg": 0.0,
    },
    # Additional cameras. Until their position relative to the webcam is calibrated
    # they are shown and tracked, but not yet used for the hand pose.
    "extra_cameras": {
        "phone": {  # Android phone over adb (scrcpy server as the streamer)
            "type": "scrcpy", "enabled": False, "serial": "", "camera_id": "0",
            "width": 1280, "height": 720, "fps": 30,
            "flip_input": False,
            "light_correction": "local",
            "torch": False,  # switch the phone's flash LED on as a lamp while streaming
            "rotate_deg": 0,  # 90 = phone standing upright (portrait)
            "head_hz": 10,
            "hfov_deg": 100.0, "swap_hands": False, "mirror_preview": False,
            # Position relative to the webcam, set by the camera calibration.
            "calibrated": False, "rot": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], "tvec": [0.0, 0.0, 0.0],
        },
        "pico": {  # Pico 2 + OV5640 over USB serial
            "type": "pico", "enabled": False, "port": "/dev/ttyACM0",
            "xclk_div": 12,  # sensor clock = 150 MHz / this; 12 = 16.8 fps (firmware default), 7 = ~28.7 fps
            "width": 640, "height": 480, "fps": 17,
            "rotate_deg": 0,
            "head_hz": 5,
            "flip_input": True,  # this firmware's sensor setup delivers a mirrored picture
            "light_correction": "gamma",  # its compressed picture turns blocky under "local"
            "jpeg_scale": 16,  # sensor JPEG compression: 2 = best ... 63 = smallest frames (faster over USB)
            "ae_target": 48,  # auto-exposure brightness target; raise it if a lamp in view darkens the picture
            "hfov_deg": 62.0, "swap_hands": False, "mirror_preview": False,
            "calibrated": False, "rot": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], "tvec": [0.0, 0.0, 0.0],
        },
        "headset": {  # the headset phone's own camera: it moves with the head, so its place
            # comes from the tracked head and the phone's motion sensors, not from calibration
            "type": "scrcpy", "enabled": False, "serial": "", "camera_id": "0", "zoom": 0.67,
            "width": 1280, "height": 720, "fps": 30,
            "flip_input": False, "light_correction": "gamma", "torch": False,
            "rotate_deg": 0, "head_hz": 0, "roi_hands": False,
            "hfov_deg": 100.0, "swap_hands": False, "mirror_preview": False,
            "moving": True, "calibrated": False,
            # Set by the headset camera calibration (hub/headcam.py): "rot" = how the camera
            # sits in the headset, "align" = runtime world to webcam frame, "tvec" = the
            # camera's offset from the eyes in headset axes.
            "rot": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], "tvec": [0.0, 0.0, 0.0],
            "align": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        },
        "ir": {  # the laptop's infrared camera, beside the webcam: sees by its own light, so it
            # does not care about room lighting or a hand the colour of the wall behind it
            "type": "v4l2", "enabled": False, "device": 2,
            "width": 640, "height": 360, "fps": 30, "fixed_framerate": False,
            "ir_strobe": True,  # its emitter lights every second frame: use only the lit ones
            "flip_input": False, "light_correction": "gamma", "rotate_deg": 0, "head_hz": 5,
            "hfov_deg": 86.0, "swap_hands": False, "mirror_preview": True,
            # Rigidly beside the webcam, looking the same way (measured: 0.2 degrees apart).
            "fixed": True, "calibrated": True,
            "rot": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], "tvec": [0.0, 0.0, 0.0],
        },
    },
    "tracking": {
        "head_tracking": True,  # hands are placed relative to the head the cameras see
        "free_hand": "",  # "left" | "right": one-handed play, the other hand holds the headset to the face
        "ignore_hand_near_head_m": 0.0,  # > 0: a hand this close to the head is holding the headset; ignore it
        "user_height_m": 0.0,  # the user's height; fixes the size of the 3D reconstruction (0 = unknown)
        "hand_scale": 1.0,  # per-user hand size relative to MediaPipe's average hand
        "lost_hold_ms": 300,
        "min_confidence": 0.35,  # below this, button states are held
        "grip_offset": [0.0, 0.0, 0.0],  # metres, controller frame
        "roll_offset_deg": 0.0,
        "pitch_offset_deg": 0.0,
    },
    "smoothing": {"min_cutoff": 1.0, "beta": 0.02, "d_cutoff": 1.0},
    "gestures": {
        "profile": "gesture",  # "gesture" | "finger"
        "pinch_far": 0.5,
        "pinch_near": 0.15,
        "trigger_click": 0.9,
        "trigger_release": 0.7,
        "curl_open": 1.8,
        "curl_closed": 1.0,
        "grip_click": 0.8,
        "grip_release": 0.6,
        "ab_pinch": 0.25,
        "ab_max_grip": 0.5,
    },
    "driver": {
        "host": "127.0.0.1",
        "port": 39570,
        "status_port": 39571,
        "follow_hmd_position": True,
        "yaw_offset": 0.0,
        "min_send_hz": 60,
    },
}


def deep_merge(base: dict, over: dict) -> dict:
    """Return base updated with over; only keys that exist in base are kept."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k not in out:
            continue
        if isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        elif isinstance(out[k], dict) or isinstance(v, dict):
            continue
        elif isinstance(out[k], list):
            if (isinstance(v, list) and len(v) == len(out[k])
                    and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)):
                out[k] = [float(x) for x in v]
        elif isinstance(out[k], bool):
            out[k] = bool(v)
        elif isinstance(out[k], (int, float)) and isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = type(out[k])(v)
        elif isinstance(out[k], str) and isinstance(v, str):
            out[k] = v
    return out


class Config:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else CONFIG_PATH
        self._lock = threading.Lock()
        self.data = copy.deepcopy(DEFAULTS)
        self.load()

    def load(self) -> None:
        try:
            loaded = json.loads(self.path.read_text())
            if not isinstance(loaded, dict):
                raise OSError("top level is not an object")
            self.data = deep_merge(DEFAULTS, loaded)
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, OSError) as e:
            print(f"[config] ignoring unreadable {self.path}: {e}")

    def get(self) -> dict:
        with self._lock:
            return copy.deepcopy(self.data)

    def update(self, patch: dict) -> dict:
        with self._lock:
            self.data = deep_merge(self.data, patch)
            self._save()
            return copy.deepcopy(self.data)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)
